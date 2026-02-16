"""
Modal-based GPU kernel profiler using Nsight Compute (ncu) for CUDA
and torch.profiler for Triton kernels.
"""

import os
import sys
import json
import csv
import io
import time
import modal
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field

# Force UTF-8 for stdout/stderr on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GpuTimelineSample:
    """A single NVML sample taken during kernel execution."""

    timestamp_ms: float = 0.0  # ms since execution start
    gpu_utilization_percent: float = 0.0
    memory_utilization_percent: float = 0.0
    memory_used_mb: float = 0.0
    power_draw_w: float = 0.0
    temperature_c: float = 0.0
    sm_clock_mhz: int = 0
    mem_clock_mhz: int = 0


@dataclass
class KernelProfilingMetrics:
    """Per-kernel-launch profiling data."""

    kernel_name: str = ""
    execution_time_us: float = 0.0
    sm_efficiency_percent: float = 0.0
    memory_throughput_gbs: float = 0.0
    achieved_occupancy_percent: float = 0.0
    registers_per_thread: int = 0
    block_size: Tuple[int, int, int] = (0, 0, 0)
    grid_size: Tuple[int, int, int] = (0, 0, 0)
    shared_memory_bytes: int = 0
    limiting_factor: str = "unknown"  # warps | registers | shared_memory | blocks | unknown


@dataclass
class ProfilingResult:
    """Top-level profiling result returned to the VS Code extension."""

    successful: bool = False
    error_message: str = ""
    kernel_metrics: List[Dict[str, Any]] = field(default_factory=list)
    gpu_name: str = ""
    compute_capability: str = ""
    raw_ncu_output: str = ""
    profiling_tool_used: str = ""  # "ncu" | "torch_profiler" | "nvcc_fallback"
    timeline_samples: List[Dict[str, Any]] = field(default_factory=list)
    gpu_peak_memory_bw_gbs: float = 0.0  # peak bandwidth for reference line
    gpu_power_limit_w: float = 0.0  # TDP for reference line


# ---------------------------------------------------------------------------
# Docker images  (reuse the same base images from kernel_runner.py)
# ---------------------------------------------------------------------------

cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.1.1-devel-ubuntu24.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("build-essential")
    .uv_pip_install("nvidia-ml-py", "numpy")
)

triton_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "torch",
    "triton",
    "numpy",
    "nvidia-ml-py",
)

app = modal.App("kernel-orbit-profiler")


# ---------------------------------------------------------------------------
# GPU info helper  (same as kernel_runner.py)
# ---------------------------------------------------------------------------


def get_gpu_info() -> Dict[str, Any]:
    """Get detailed GPU information using nvidia-ml-py."""
    try:
        from pynvml import (
            nvmlInit,
            nvmlShutdown,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetName,
            nvmlDeviceGetMemoryInfo,
            nvmlDeviceGetCudaComputeCapability,
        )

        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        name = nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")

        info = {
            "name": name,
            "memory_total_mb": nvmlDeviceGetMemoryInfo(handle).total / (1024**2),
            "compute_capability": ".".join(
                map(str, nvmlDeviceGetCudaComputeCapability(handle))
            ),
        }
        nvmlShutdown()
        return info
    except Exception as e:
        return {"error": str(e), "name": "Unknown", "compute_capability": "Unknown"}


# ---------------------------------------------------------------------------
# NCU CSV parser
# ---------------------------------------------------------------------------

# Mapping from ncu metric names → our fields
_NCU_METRIC_MAP = {
    "gpu__time_duration.sum": "execution_time_us",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_efficiency_percent",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "memory_throughput_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved_occupancy_percent",
    "launch__registers_per_thread": "registers_per_thread",
    "launch__block_size": "block_size_flat",
    "launch__grid_size": "grid_size_flat",
    "launch__shared_mem_per_block_dynamic": "shared_memory_dynamic",
    "launch__shared_mem_per_block_static": "shared_memory_static",
    "launch__occupancy_limit_warps": "limit_warps",
    "launch__occupancy_limit_registers": "limit_registers",
    "launch__occupancy_limit_shared_mem": "limit_shared_mem",
    "launch__occupancy_limit_blocks": "limit_blocks",
}


def parse_ncu_csv(csv_text: str, gpu_mem_bw_gbs: float = 0.0) -> List[Dict[str, Any]]:
    """
    Parse ncu --csv output into a list of KernelProfilingMetrics dicts.
    ``gpu_mem_bw_gbs`` is the peak memory bandwidth in GB/s for this GPU
    (used to convert memory throughput % → absolute GB/s).
    """
    # ncu CSV sometimes has leading junk lines before the header.
    # Find the header row that starts with "ID" or contains "Kernel Name".
    lines = csv_text.strip().splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('"ID"') or line.startswith("ID") or "Kernel Name" in line:
            header_idx = i
            break

    if header_idx is None:
        return []

    csv_body = "\n".join(lines[header_idx:])
    reader = csv.DictReader(io.StringIO(csv_body))

    # Collect raw metric values keyed by (ID, Kernel Name)
    kernel_raw: Dict[str, Dict[str, Any]] = {}  # key = "id|kernel_name"
    for row in reader:
        kid = row.get("ID", "0")
        kname = row.get("Kernel Name", "unknown")
        key = f"{kid}|{kname}"
        if key not in kernel_raw:
            kernel_raw[key] = {"kernel_name": kname}

        metric_name = row.get("Metric Name", "")
        metric_value = row.get("Metric Value", "")
        mapped = _NCU_METRIC_MAP.get(metric_name)
        if mapped:
            try:
                kernel_raw[key][mapped] = float(metric_value.replace(",", ""))
            except (ValueError, TypeError):
                kernel_raw[key][mapped] = metric_value

    results: List[Dict[str, Any]] = []
    for raw in kernel_raw.values():
        # Determine limiting factor
        limiters = {
            "warps": raw.get("limit_warps", 0),
            "registers": raw.get("limit_registers", 0),
            "shared_memory": raw.get("limit_shared_mem", 0),
            "blocks": raw.get("limit_blocks", 0),
        }
        # The limiter with the smallest value is the bottleneck
        numeric_limiters = {
            k: float(v) for k, v in limiters.items() if isinstance(v, (int, float))
        }
        if numeric_limiters:
            limiting = min(numeric_limiters, key=numeric_limiters.get)  # type: ignore[arg-type]
        else:
            limiting = "unknown"

        # Convert memory throughput % to GB/s
        mem_pct = float(raw.get("memory_throughput_pct", 0))
        mem_gbs = (mem_pct / 100.0) * gpu_mem_bw_gbs if gpu_mem_bw_gbs > 0 else mem_pct

        block_flat = int(raw.get("block_size_flat", 0))
        grid_flat = int(raw.get("grid_size_flat", 0))

        shared_dyn = int(raw.get("shared_memory_dynamic", 0))
        shared_static = int(raw.get("shared_memory_static", 0))

        # Convert execution time: ncu reports in nanoseconds → microseconds
        exec_time_ns = float(raw.get("execution_time_us", 0))
        exec_time_us = exec_time_ns / 1000.0

        m = KernelProfilingMetrics(
            kernel_name=raw.get("kernel_name", "unknown"),
            execution_time_us=exec_time_us,
            sm_efficiency_percent=float(raw.get("sm_efficiency_percent", 0)),
            memory_throughput_gbs=mem_gbs,
            achieved_occupancy_percent=float(
                raw.get("achieved_occupancy_percent", 0)
            ),
            registers_per_thread=int(raw.get("registers_per_thread", 0)),
            block_size=(block_flat, 1, 1),
            grid_size=(grid_flat, 1, 1),
            shared_memory_bytes=shared_dyn + shared_static,
            limiting_factor=limiting,
        )
        results.append(asdict(m))
    return results


# ---------------------------------------------------------------------------
# CUDA profiling implementation
# ---------------------------------------------------------------------------


def _parse_resource_usage(compiler_output: str) -> Dict[str, Any]:
    """
    Parse ``nvcc --resource-usage`` output to extract per-kernel resource info.

    Example lines:
      ptxas info    : Used 18 registers, 384 bytes smem, 0 bytes cmem[0]
      ptxas info    : Function properties for matrixMulTiled
      ptxas info    : Used 24 registers, 2048+0 bytes smem, 380 bytes cmem[0], ...
    """
    import re

    kernels: Dict[str, Dict[str, Any]] = {}
    current_func: str | None = None

    for line in compiler_output.splitlines():
        # Detect "Function properties for <name>"
        func_match = re.search(r"Function properties for\s+(\S+)", line)
        if func_match:
            current_func = func_match.group(1)
            continue

        # Detect "Used N registers, ..."
        reg_match = re.search(r"Used\s+(\d+)\s+registers", line)
        if reg_match:
            regs = int(reg_match.group(1))
            # Shared memory: "N bytes smem" or "N+M bytes smem"
            smem = 0
            smem_match = re.search(r"(\d+)(?:\+(\d+))?\s+bytes\s+smem", line)
            if smem_match:
                smem = int(smem_match.group(1))
                if smem_match.group(2):
                    smem += int(smem_match.group(2))

            name = current_func or "unknown"
            kernels[name] = {"registers": regs, "shared_memory_bytes": smem}
            current_func = None  # reset for next kernel

    return kernels


def _sample_gpu_timeline(
    exe_file_path: str,
    interval_ms: float = 50.0,
) -> Tuple[subprocess.CompletedProcess, List[Dict[str, Any]]]:
    """
    Run the kernel binary while sampling GPU metrics via NVML in a background
    thread.  Returns (CompletedProcess, list-of-sample-dicts).
    """
    import threading
    import time as _time

    samples: List[Dict[str, Any]] = []
    stop_event = threading.Event()

    def _sampler():
        try:
            from pynvml import (
                nvmlInit,
                nvmlShutdown,
                nvmlDeviceGetHandleByIndex,
                nvmlDeviceGetUtilizationRates,
                nvmlDeviceGetMemoryInfo,
                nvmlDeviceGetPowerUsage,
                nvmlDeviceGetTemperature,
                nvmlDeviceGetClockInfo,
                NVML_TEMPERATURE_GPU,
                NVML_CLOCK_SM,
                NVML_CLOCK_MEM,
            )

            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0)
            t0 = _time.perf_counter()

            while not stop_event.is_set():
                try:
                    util = nvmlDeviceGetUtilizationRates(handle)
                    mem = nvmlDeviceGetMemoryInfo(handle)
                    power_mw = nvmlDeviceGetPowerUsage(handle)  # milliwatts
                    temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
                    sm_clk = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_SM)
                    mem_clk = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_MEM)

                    sample = GpuTimelineSample(
                        timestamp_ms=(_time.perf_counter() - t0) * 1000.0,
                        gpu_utilization_percent=float(util.gpu),
                        memory_utilization_percent=float(util.memory),
                        memory_used_mb=mem.used / (1024**2),
                        power_draw_w=power_mw / 1000.0,
                        temperature_c=float(temp),
                        sm_clock_mhz=int(sm_clk),
                        mem_clock_mhz=int(mem_clk),
                    )
                    samples.append(asdict(sample))
                except Exception:
                    pass  # skip one sample if NVML hiccups

                stop_event.wait(interval_ms / 1000.0)

            nvmlShutdown()
        except Exception:
            pass  # NVML not available — samples stays empty

    sampler_thread = threading.Thread(target=_sampler, daemon=True)
    sampler_thread.start()

    proc = subprocess.run(
        [exe_file_path], capture_output=True, text=True, timeout=60
    )

    stop_event.set()
    sampler_thread.join(timeout=2.0)

    return proc, samples


def _parse_launch_config(
    kernel_source: str, kernel_stdout: str
) -> Dict[str, Dict[str, Any]]:
    """
    Try to extract grid/block dimensions from kernel stdout and source.

    Returns a dict keyed by kernel name (or "*" for a global config that
    applies to all kernels) mapping to ``{"block": (x,y,z), "grid": (x,y,z)}``.
    """
    import re

    configs: Dict[str, Dict[str, Any]] = {}

    # --- Parse from stdout ---
    # Common patterns:
    #   "Grid: (64, 64), Block: (16, 16)"
    #   "grid: (64,64,1), block: (16,16,1)"
    #   "Grid dim: 64x64, Block dim: 16x16"
    gb_re = re.compile(
        r"[Gg]rid[^:]*:\s*\(?(\d+)[,x×\s]+(\d+)(?:[,x×\s]+(\d+))?\)?[,;\s]+"
        r"[Bb]lock[^:]*:\s*\(?(\d+)[,x×\s]+(\d+)(?:[,x×\s]+(\d+))?\)?"
    )
    m = gb_re.search(kernel_stdout)
    if m:
        gx, gy = int(m.group(1)), int(m.group(2))
        gz = int(m.group(3)) if m.group(3) else 1
        bx, by = int(m.group(4)), int(m.group(5))
        bz = int(m.group(6)) if m.group(6) else 1
        configs["*"] = {"block": (bx, by, bz), "grid": (gx, gy, gz)}

    # --- Also try parsing dim3 from source as fallback ---
    if "*" not in configs:
        # Look for dim3 blockDim(x, y [, z])
        block_re = re.compile(
            r"dim3\s+(?:block|blockDim|block_dim|threads)\s*\(\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*(\d+))?\s*\)",
            re.IGNORECASE,
        )
        grid_re = re.compile(
            r"dim3\s+(?:grid|gridDim|grid_dim|blocks)\s*\(\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*(\d+))?\s*\)",
            re.IGNORECASE,
        )
        # Also look for <<<dim3(x,y), dim3(x,y)>>> or <<<gridDim, blockDim>>>
        bm = block_re.search(kernel_source)
        gm = grid_re.search(kernel_source)
        block = grid = None
        if bm:
            block = (int(bm.group(1)), int(bm.group(2)), int(bm.group(3)) if bm.group(3) else 1)
        if gm:
            grid = (int(gm.group(1)), int(gm.group(2)), int(gm.group(3)) if gm.group(3) else 1)

        # Fallback: single-number block/grid like <<<N/256, 256>>>
        if block is None:
            single_block = re.search(r"<<<[^>]*,\s*(\d+)\s*>>>", kernel_source)
            if single_block:
                bs = int(single_block.group(1))
                block = (bs, 1, 1)
        if grid is None:
            single_grid = re.search(r"<<<\s*(\d+)\s*,", kernel_source)
            if single_grid:
                gs = int(single_grid.group(1))
                grid = (gs, 1, 1)

        if block or grid:
            configs["*"] = {
                "block": block or (0, 0, 0),
                "grid": grid or (0, 0, 0),
            }

    return configs


def _parse_kernel_times_from_stdout(kernel_stdout: str) -> Dict[str, float]:
    """
    Try to extract per-kernel execution times from kernel stdout.

    Looks for common patterns like:
      "Naive kernel: 4.5878 ms"
      "Tiled kernel: 2.8534 ms"
      "kernel_name: 1.234 ms"
      "kernel_name time: 1.234 ms"
    """
    import re

    times: Dict[str, float] = {}

    # Pattern: "<name> kernel: <time> ms" or "<name>: <time> ms"
    time_re = re.compile(
        r"(\w[\w\s]*?)\s*(?:kernel)?\s*(?:time)?\s*:\s*([\d.]+)\s*ms",
        re.IGNORECASE,
    )
    for m in time_re.finditer(kernel_stdout):
        name = m.group(1).strip().lower()
        t_ms = float(m.group(2))
        times[name] = t_ms * 1000.0  # convert ms → μs

    return times


def _match_kernel_time(
    mangled_name: str, parsed_times: Dict[str, float]
) -> Optional[float]:
    """
    Try to match a mangled kernel name (e.g. ``_Z14matrixMulTiledPKfS0_Pfi``)
    against the friendly names extracted from stdout (e.g. ``tiled``).
    """
    lower = mangled_name.lower()
    for friendly, time_us in parsed_times.items():
        # Check if the friendly name (or significant words) appear inside
        # the mangled name.
        words = friendly.split()
        if any(w in lower for w in words):
            return time_us
    return None


def _profile_cuda_fallback(
    kernel_source: str, gpu_type: str, exe_file_path: str, compiler_output: str
) -> Dict[str, Any]:
    """
    Fallback profiling when ncu is unavailable or fails.

    Uses:
    - ``nvcc --resource-usage`` output for register/shared-memory info
    - Direct execution with timing for execution time
    - NVML for GPU utilisation metrics
    - Static GPU specs for theoretical occupancy estimation
    - Kernel stdout for launch config & per-kernel timing
    """
    import re
    import statistics

    result = ProfilingResult()
    result.profiling_tool_used = "nvcc_fallback"

    try:
        gpu_info = get_gpu_info()
        result.gpu_name = gpu_info.get("name", "Unknown")
        result.compute_capability = gpu_info.get("compute_capability", "Unknown")

        # --------------- resource-usage parsing ---------------
        resource_info = _parse_resource_usage(compiler_output)

        # --------------- GPU bandwidth lookup for derived metrics ---------------
        from kernel_profiler import _GPU_MEM_BW, _GPU_POWER_LIMIT
        peak_mem_bw = _GPU_MEM_BW.get(gpu_type, 0.0)
        power_limit = _GPU_POWER_LIMIT.get(gpu_type, 0.0)
        result.gpu_peak_memory_bw_gbs = peak_mem_bw
        result.gpu_power_limit_w = power_limit

        # --------------- execution timing with NVML sampling ---------------
        import time as _time

        times_us: List[float] = []
        kernel_stdout = ""
        all_samples: List[Dict[str, Any]] = []

        for run_idx in range(5):
            t0 = _time.perf_counter()
            run_proc, samples = _sample_gpu_timeline(
                exe_file_path, interval_ms=25.0
            )
            t1 = _time.perf_counter()
            times_us.append((t1 - t0) * 1_000_000)  # seconds → μs
            kernel_stdout = run_proc.stdout + run_proc.stderr

            # Keep the samples from the run with the most data points
            if len(samples) > len(all_samples):
                all_samples = samples

        result.timeline_samples = all_samples
        avg_time = statistics.mean(times_us)

        # --------------- Derive memory throughput from NVML ---------------
        # NVML memory utilization % × peak bandwidth ≈ actual throughput
        avg_mem_util = 0.0
        avg_gpu_util = 0.0
        if all_samples:
            avg_mem_util = statistics.mean(
                s["memory_utilization_percent"] for s in all_samples
            )
            avg_gpu_util = statistics.mean(
                s["gpu_utilization_percent"] for s in all_samples
            )
        derived_mem_throughput_gbs = (avg_mem_util / 100.0) * peak_mem_bw if peak_mem_bw > 0 else -1.0
        derived_sm_activity = avg_gpu_util  # not the same as SM efficiency, but closest proxy

        # --------------- GPU specs for occupancy estimation ---------------
        _GPU_SPECS: Dict[str, Dict[str, int]] = {
            "T4":        {"sm_count": 40,  "max_threads_per_sm": 1024, "max_regs_per_sm": 65536, "max_smem_per_sm": 65536},
            "L4":        {"sm_count": 60,  "max_threads_per_sm": 1536, "max_regs_per_sm": 65536, "max_smem_per_sm": 102400},
            "A10G":      {"sm_count": 80,  "max_threads_per_sm": 1536, "max_regs_per_sm": 65536, "max_smem_per_sm": 102400},
            "A100-40GB": {"sm_count": 108, "max_threads_per_sm": 2048, "max_regs_per_sm": 65536, "max_smem_per_sm": 167936},
            "A100-80GB": {"sm_count": 108, "max_threads_per_sm": 2048, "max_regs_per_sm": 65536, "max_smem_per_sm": 167936},
            "L40S":      {"sm_count": 142, "max_threads_per_sm": 1536, "max_regs_per_sm": 65536, "max_smem_per_sm": 102400},
            "H100":      {"sm_count": 132, "max_threads_per_sm": 2048, "max_regs_per_sm": 65536, "max_smem_per_sm": 233472},
            "H200":      {"sm_count": 132, "max_threads_per_sm": 2048, "max_regs_per_sm": 65536, "max_smem_per_sm": 233472},
            "B200":      {"sm_count": 160, "max_threads_per_sm": 2048, "max_regs_per_sm": 65536, "max_smem_per_sm": 233472},
        }

        specs = _GPU_SPECS.get(gpu_type, _GPU_SPECS["T4"])

        # --------------- detect launch config from kernel stdout ---------------
        launch_configs = _parse_launch_config(kernel_source, kernel_stdout)
        global_config = launch_configs.get("*", {})
        default_block = global_config.get("block", (0, 0, 0))
        default_grid = global_config.get("grid", (0, 0, 0))

        # --------------- per-kernel timing from stdout ---------------
        parsed_times = _parse_kernel_times_from_stdout(kernel_stdout)

        # Also try to detect __global__ function names from source.
        global_re = re.compile(r"__global__\s+\w+\s+(\w+)\s*\(")
        kernel_names = global_re.findall(kernel_source)

        # --------------- build metrics list ---------------
        metrics_list: List[Dict[str, Any]] = []

        raw_lines = [f"== Fallback profiling (nvcc --resource-usage + timing) =="]
        raw_lines.append(f"GPU: {result.gpu_name} (CC {result.compute_capability})")
        raw_lines.append(f"Execution times (μs): {[f'{t:.1f}' for t in times_us]}")
        raw_lines.append(f"Average: {avg_time:.1f} μs")
        if default_block != (0, 0, 0) or default_grid != (0, 0, 0):
            raw_lines.append(f"Block: {default_block}, Grid: {default_grid}")
        if all_samples:
            raw_lines.append(f"NVML samples: {len(all_samples)} (avg GPU util: {avg_gpu_util:.1f}%, avg mem util: {avg_mem_util:.1f}%)")
            if derived_mem_throughput_gbs > 0:
                raw_lines.append(f"Est. memory throughput: {derived_mem_throughput_gbs:.1f} GB/s (of {peak_mem_bw:.0f} GB/s peak)")
        raw_lines.append("")

        if resource_info:
            for kname, rinfo in resource_info.items():
                regs = rinfo.get("registers", 0)
                smem = rinfo.get("shared_memory_bytes", 0)

                # Estimate occupancy
                max_threads = specs["max_threads_per_sm"]
                # Threads limited by registers: each thread uses `regs` registers
                if regs > 0:
                    threads_by_regs = (specs["max_regs_per_sm"] // regs)
                    # Round down to warp granularity (32)
                    threads_by_regs = (threads_by_regs // 32) * 32
                else:
                    threads_by_regs = max_threads

                occupancy_pct = min(100.0, (min(threads_by_regs, max_threads) / max_threads) * 100)

                # Determine limiting factor
                limiting = "unknown"
                if threads_by_regs < max_threads:
                    limiting = "registers"
                if smem > 0 and smem > specs["max_smem_per_sm"]:
                    limiting = "shared_memory"

                # Try to get per-kernel time from stdout; fall back to even split
                kernel_time = _match_kernel_time(kname, parsed_times)
                if kernel_time is None:
                    kernel_time = avg_time / max(len(resource_info), 1)

                # Get launch config (per-kernel or global)
                kconfig = launch_configs.get(kname, global_config)
                block = kconfig.get("block", default_block)
                grid = kconfig.get("grid", default_grid)

                m = KernelProfilingMetrics(
                    kernel_name=kname,
                    execution_time_us=kernel_time,
                    sm_efficiency_percent=derived_sm_activity if all_samples else -1.0,
                    memory_throughput_gbs=derived_mem_throughput_gbs,
                    achieved_occupancy_percent=occupancy_pct,
                    registers_per_thread=regs,
                    block_size=tuple(block),
                    grid_size=tuple(grid),
                    shared_memory_bytes=smem,
                    limiting_factor=limiting,
                )
                metrics_list.append(asdict(m))

                raw_lines.append(f"Kernel: {kname}")
                raw_lines.append(f"  Registers/thread: {regs}")
                raw_lines.append(f"  Shared memory: {smem} bytes")
                raw_lines.append(f"  Est. occupancy: {occupancy_pct:.1f}%")
                raw_lines.append(f"  Block: {block}, Grid: {grid}")
                raw_lines.append(f"  Limiting factor: {limiting}")
                if kernel_time != avg_time / max(len(resource_info), 1):
                    raw_lines.append(f"  Per-kernel time (from stdout): {kernel_time:.1f} μs")
                raw_lines.append("")
        elif kernel_names:
            # No resource-usage data but we know kernel names
            for kname in kernel_names:
                kernel_time = _match_kernel_time(kname, parsed_times)
                if kernel_time is None:
                    kernel_time = avg_time / max(len(kernel_names), 1)
                kconfig = launch_configs.get(kname, global_config)
                m = KernelProfilingMetrics(
                    kernel_name=kname,
                    execution_time_us=kernel_time,
                    sm_efficiency_percent=derived_sm_activity if all_samples else -1.0,
                    memory_throughput_gbs=derived_mem_throughput_gbs,
                    block_size=tuple(kconfig.get("block", default_block)),
                    grid_size=tuple(kconfig.get("grid", default_grid)),
                )
                metrics_list.append(asdict(m))
                raw_lines.append(f"Kernel: {kname}  (no resource-usage data)")
        else:
            # Completely generic fallback — single unnamed entry
            m = KernelProfilingMetrics(
                kernel_name="(entire program)",
                execution_time_us=avg_time,
                sm_efficiency_percent=derived_sm_activity if all_samples else -1.0,
                memory_throughput_gbs=derived_mem_throughput_gbs,
                block_size=tuple(default_block),
                grid_size=tuple(default_grid),
            )
            metrics_list.append(asdict(m))

        if kernel_stdout.strip():
            raw_lines.append("--- Kernel stdout ---")
            raw_lines.append(kernel_stdout.strip())

        result.kernel_metrics = metrics_list
        result.raw_ncu_output = "\n".join(raw_lines)
        result.successful = True

    except Exception as e:
        import traceback
        result.successful = False
        result.error_message = f"Fallback profiling failed: {str(e)}\n{traceback.format_exc()}"

    return asdict(result)


def _profile_cuda_impl(kernel_source: str, gpu_type: str) -> Dict[str, Any]:
    """
    Profile a CUDA kernel.

    Strategy:
    1. Compile with ``nvcc --resource-usage -O3 -lineinfo``
    2. Attempt ``ncu --set basic --csv`` profiling
    3. If ncu fails (container permission issue, missing counters, etc.)
       → fall back to resource-usage + timing based profiling
    """
    import shutil

    result = ProfilingResult()
    result.profiling_tool_used = "ncu"

    try:
        gpu_info = get_gpu_info()
        result.gpu_name = gpu_info.get("name", "Unknown")
        result.compute_capability = gpu_info.get("compute_capability", "Unknown")

        from kernel_profiler import _GPU_MEM_BW
        gpu_mem_bw = _GPU_MEM_BW.get(gpu_type, 0.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            source_file = Path(tmpdir) / "kernel.cu"
            exe_file = Path(tmpdir) / "kernel"
            csv_file = Path(tmpdir) / "profile.csv"

            source_file.write_text(kernel_source)

            # Compile with --resource-usage so we always have register/smem info
            compile_proc = subprocess.run(
                [
                    "nvcc",
                    str(source_file),
                    "-o",
                    str(exe_file),
                    "-O3",
                    "-lineinfo",
                    "--resource-usage",
                ],
                capture_output=True,
                text=True,
            )

            if compile_proc.returncode != 0:
                result.successful = False
                result.error_message = (
                    f"Compilation failed:\n{compile_proc.stdout}{compile_proc.stderr}"
                )
                return asdict(result)

            compiler_output = compile_proc.stderr  # nvcc prints resource-usage to stderr

            # ---- Attempt ncu profiling ----
            ncu_path = shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
            if not Path(ncu_path).exists():
                ncu_path = shutil.which("ncu")

            if not ncu_path:
                # ncu not found → go straight to fallback
                return _profile_cuda_fallback(
                    kernel_source, gpu_type, str(exe_file), compiler_output
                )

            ncu_proc = subprocess.run(
                [
                    ncu_path,
                    "--set", "basic",
                    "--csv",
                    "--log-file", str(csv_file),
                    str(exe_file),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            raw_output = ""
            if csv_file.exists():
                raw_output = csv_file.read_text()
            if not raw_output:
                raw_output = ncu_proc.stdout or ncu_proc.stderr or ""
            result.raw_ncu_output = raw_output

            # Detect ncu failure: check return code AND whether we got real metrics
            metrics: List[Dict[str, Any]] = []
            if raw_output:
                metrics = parse_ncu_csv(raw_output, gpu_mem_bw)
            if not metrics and ncu_proc.stdout:
                metrics = parse_ncu_csv(ncu_proc.stdout, gpu_mem_bw)

            ncu_has_errors = "==ERROR==" in raw_output or "==ERROR==" in (
                ncu_proc.stderr or ""
            )

            if metrics:
                # ncu succeeded, use its data
                result.kernel_metrics = metrics
                result.successful = True
                return asdict(result)

            # ncu ran but produced no usable metrics → fallback
            fallback_result = _profile_cuda_fallback(
                kernel_source, gpu_type, str(exe_file), compiler_output
            )
            # Preserve the raw ncu output so the user can see what happened
            if raw_output:
                fallback_result["raw_ncu_output"] = (
                    f"== ncu failed (exit {ncu_proc.returncode}), using fallback profiler ==\n"
                    f"{raw_output}\n\n"
                    f"{fallback_result.get('raw_ncu_output', '')}"
                )
            return fallback_result

    except subprocess.TimeoutExpired:
        result.successful = False
        result.error_message = "ncu profiling timed out after 300 seconds"
    except Exception as e:
        import traceback
        result.successful = False
        result.error_message = f"{str(e)}\n{traceback.format_exc()}"

    return asdict(result)


# ---------------------------------------------------------------------------
# Triton profiling implementation
# ---------------------------------------------------------------------------


def _profile_triton_impl(kernel_source: str, gpu_type: str) -> Dict[str, Any]:
    """Profile a Triton kernel using torch.profiler."""
    import torch
    import contextlib
    import io as _io

    result = ProfilingResult()
    result.profiling_tool_used = "torch_profiler"

    try:
        import triton
        import triton.language as tl

        gpu_info = get_gpu_info()
        result.gpu_name = gpu_info.get("name", "Unknown")
        result.compute_capability = gpu_info.get("compute_capability", "Unknown")

        kernel_globals = {
            "torch": torch,
            "triton": triton,
            "tl": tl,
            "np": __import__("numpy"),
            "numpy": __import__("numpy"),
        }

        stdout_capture = _io.StringIO()

        # first exec the script to define kernels
        with contextlib.redirect_stdout(stdout_capture):
            exec(kernel_source, kernel_globals)

        # detect a callable entry point
        run_fn = None
        for name in ["benchmark", "benchmark_kernel", "main", "run", "test"]:
            if name in kernel_globals and callable(kernel_globals[name]):
                run_fn = kernel_globals[name]
                break

        if run_fn is None:
            # no entry point — report what we captured from exec
            result.successful = True
            result.raw_ncu_output = stdout_capture.getvalue()
            return asdict(result)

        # warmup
        torch.cuda.synchronize()
        for _ in range(3):
            with contextlib.redirect_stdout(stdout_capture):
                run_fn()
            torch.cuda.synchronize()

        # Profile
        from torch.profiler import profile, ProfilerActivity

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=False,
        ) as prof:
            with contextlib.redirect_stdout(stdout_capture):
                run_fn()
            torch.cuda.synchronize()

        result.raw_ncu_output = prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=50
        )

        # Extract per-kernel metrics
        device_props = torch.cuda.get_device_properties(0)
        metrics_list: List[Dict[str, Any]] = []
        for evt in prof.key_averages():
            if evt.device_type is not None and evt.self_cuda_time_total > 0:
                cuda_time_us = evt.self_cuda_time_total  # already in μs
                m = KernelProfilingMetrics(
                    kernel_name=evt.key,
                    execution_time_us=cuda_time_us,
                    sm_efficiency_percent=0.0,  # not directly available from torch profiler
                    memory_throughput_gbs=0.0,
                    achieved_occupancy_percent=0.0,
                    registers_per_thread=0,
                    block_size=(0, 0, 0),
                    grid_size=(0, 0, 0),
                    shared_memory_bytes=0,
                    limiting_factor="unknown",
                )
                metrics_list.append(asdict(m))

        result.kernel_metrics = metrics_list
        result.successful = True

    except Exception as e:
        import traceback

        result.successful = False
        result.error_message = f"{str(e)}\n{traceback.format_exc()}"

    return asdict(result)


# ---------------------------------------------------------------------------
# GPU memory bandwidth lookup  (GB/s peak, for % → absolute conversion)
# ---------------------------------------------------------------------------

_GPU_MEM_BW: Dict[str, float] = {
    "T4": 320.0,
    "L4": 300.0,
    "A10G": 600.0,
    "A100-40GB": 1555.0,
    "A100-80GB": 2039.0,
    "L40S": 864.0,
    "H100": 3350.0,
    "H200": 4800.0,
    "B200": 8000.0,
}

# TDP (watts) — used as reference line on power draw chart
_GPU_POWER_LIMIT: Dict[str, float] = {
    "T4": 70.0,
    "L4": 72.0,
    "A10G": 150.0,
    "A100-40GB": 250.0,
    "A100-80GB": 300.0,
    "L40S": 350.0,
    "H100": 700.0,
    "H200": 700.0,
    "B200": 1000.0,
}


# ---------------------------------------------------------------------------
# Per-GPU Modal functions  (CUDA profiling)
# ---------------------------------------------------------------------------


@app.function(gpu="T4", image=cuda_image, timeout=600)
def profile_cuda_on_t4(kernel_source: str) -> Dict:
    return _profile_cuda_impl(kernel_source, "T4")


@app.function(gpu="L4", image=cuda_image, timeout=600)
def profile_cuda_on_l4(kernel_source: str) -> Dict:
    return _profile_cuda_impl(kernel_source, "L4")


@app.function(gpu="A10G", image=cuda_image, timeout=600)
def profile_cuda_on_a10g(kernel_source: str) -> Dict:
    return _profile_cuda_impl(kernel_source, "A10G")


@app.function(gpu="A100-40GB", image=cuda_image, timeout=600)
def profile_cuda_on_a100_40gb(kernel_source: str) -> Dict:
    return _profile_cuda_impl(kernel_source, "A100-40GB")


@app.function(gpu="A100-80GB", image=cuda_image, timeout=600)
def profile_cuda_on_a100_80gb(kernel_source: str) -> Dict:
    return _profile_cuda_impl(kernel_source, "A100-80GB")


@app.function(gpu="L40S", image=cuda_image, timeout=600)
def profile_cuda_on_l40s(kernel_source: str) -> Dict:
    return _profile_cuda_impl(kernel_source, "L40S")


@app.function(gpu="H100", image=cuda_image, timeout=600)
def profile_cuda_on_h100(kernel_source: str) -> Dict:
    return _profile_cuda_impl(kernel_source, "H100")


@app.function(gpu="H200", image=cuda_image, timeout=600)
def profile_cuda_on_h200(kernel_source: str) -> Dict:
    return _profile_cuda_impl(kernel_source, "H200")


@app.function(gpu="B200", image=cuda_image, timeout=600)
def profile_cuda_on_b200(kernel_source: str) -> Dict:
    return _profile_cuda_impl(kernel_source, "B200")


# ---------------------------------------------------------------------------
# Per-GPU Modal functions  (Triton profiling)
# ---------------------------------------------------------------------------


@app.function(gpu="T4", image=triton_image, timeout=600)
def profile_triton_on_t4(kernel_source: str) -> Dict:
    return _profile_triton_impl(kernel_source, "T4")


@app.function(gpu="L4", image=triton_image, timeout=600)
def profile_triton_on_l4(kernel_source: str) -> Dict:
    return _profile_triton_impl(kernel_source, "L4")


@app.function(gpu="A10G", image=triton_image, timeout=600)
def profile_triton_on_a10g(kernel_source: str) -> Dict:
    return _profile_triton_impl(kernel_source, "A10G")


@app.function(gpu="A100-40GB", image=triton_image, timeout=600)
def profile_triton_on_a100_40gb(kernel_source: str) -> Dict:
    return _profile_triton_impl(kernel_source, "A100-40GB")


@app.function(gpu="A100-80GB", image=triton_image, timeout=600)
def profile_triton_on_a100_80gb(kernel_source: str) -> Dict:
    return _profile_triton_impl(kernel_source, "A100-80GB")


@app.function(gpu="L40S", image=triton_image, timeout=600)
def profile_triton_on_l40s(kernel_source: str) -> Dict:
    return _profile_triton_impl(kernel_source, "L40S")


@app.function(gpu="H100", image=triton_image, timeout=600)
def profile_triton_on_h100(kernel_source: str) -> Dict:
    return _profile_triton_impl(kernel_source, "H100")


@app.function(gpu="H200", image=triton_image, timeout=600)
def profile_triton_on_h200(kernel_source: str) -> Dict:
    return _profile_triton_impl(kernel_source, "H200")


@app.function(gpu="B200", image=triton_image, timeout=600)
def profile_triton_on_b200(kernel_source: str) -> Dict:
    return _profile_triton_impl(kernel_source, "B200")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    kernel_file: str = None,
    kernel_type: str = "auto",
    gpu: str = "T4",
    output_json: str = None,
):
    """
    Profile a CUDA or Triton kernel on Modal.

    Args:
        kernel_file: Path to the kernel file (.cu or .py)
        kernel_type: "cuda", "triton", or "auto"
        gpu: GPU type (T4, L4, A10G, A100-40GB, ...)
        output_json: Output JSON file path
    """
    if not kernel_file:
        print("Error: kernel_file is required")
        sys.exit(1)

    kernel_path = Path(kernel_file)
    kernel_source = kernel_path.read_text()

    if kernel_type == "auto":
        if kernel_path.suffix == ".cu":
            kernel_type = "cuda"
        elif kernel_path.suffix == ".py":
            kernel_type = "triton"
        else:
            print(f"Cannot auto-detect kernel type for {kernel_path.suffix}")
            sys.exit(1)

    print(f"Profiling {kernel_type.upper()} kernel on Modal with {gpu}...")
    print(f"   File: {kernel_file}")
    print()

    cuda_runners = {
        "T4": profile_cuda_on_t4,
        "L4": profile_cuda_on_l4,
        "A10G": profile_cuda_on_a10g,
        "A100-40GB": profile_cuda_on_a100_40gb,
        "A100-80GB": profile_cuda_on_a100_80gb,
        "L40S": profile_cuda_on_l40s,
        "H100": profile_cuda_on_h100,
        "H200": profile_cuda_on_h200,
        "B200": profile_cuda_on_b200,
    }

    triton_runners = {
        "T4": profile_triton_on_t4,
        "L4": profile_triton_on_l4,
        "A10G": profile_triton_on_a10g,
        "A100-40GB": profile_triton_on_a100_40gb,
        "A100-80GB": profile_triton_on_a100_80gb,
        "L40S": profile_triton_on_l40s,
        "H100": profile_triton_on_h100,
        "H200": profile_triton_on_h200,
        "B200": profile_triton_on_b200,
    }

    runners = cuda_runners if kernel_type == "cuda" else triton_runners
    runner = runners.get(gpu)
    if not runner:
        print(f"Error: No profiler runner for {kernel_type} on {gpu}")
        sys.exit(1)

    result = runner.remote(kernel_source)

    if result["successful"]:
        print("Profiling completed successfully!")
        print(f"  GPU: {result['gpu_name']}")
        print(f"  Tool: {result['profiling_tool_used']}")
        print(f"  Kernels found: {len(result['kernel_metrics'])}")
        print()
        for km in result["kernel_metrics"]:
            print(f"  Kernel: {km['kernel_name']}")
            print(f"    Execution Time: {km['execution_time_us']:.2f} μs")
            print(f"    SM Efficiency:  {km['sm_efficiency_percent']:.1f}%")
            print(f"    Memory Throughput: {km['memory_throughput_gbs']:.1f} GB/s")
            print(f"    Occupancy: {km['achieved_occupancy_percent']:.1f}%")
            print(f"    Registers/Thread: {km['registers_per_thread']}")
            print(f"    Shared Memory: {km['shared_memory_bytes']} bytes")
            print(f"    Limiting Factor: {km['limiting_factor']}")
            print()
    else:
        print("Profiling failed!")
        print(f"  Error: {result['error_message']}")

    if output_json:
        Path(output_json).write_text(json.dumps(result, indent=2))
        print(f"\nResults saved to {output_json}")

    return result
