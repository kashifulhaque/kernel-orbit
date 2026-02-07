"""
Unified Modal runner that handles both CUDA and Triton kernels.
"""

import sys
import json
import time
import modal
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict, field


AVAILABLE_GPUS = [
  {"id": "T4", "name": "NVIDIA T4", "memory_gb": 16, "architecture": "Turing"},
  {"id": "L4", "name": "NVIDIA L4", "memory_gb": 24, "architecture": "Ada Lovelace"},
  {"id": "A10G", "name": "NVIDIA A10G", "memory_gb": 24, "architecture": "Ampere"},
  {
    "id": "A100-40GB",
    "name": "NVIDIA A100 (40GB)",
    "memory_gb": 40,
    "architecture": "Ampere",
  },
  {
    "id": "A100-80GB",
    "name": "NVIDIA A100 (80GB)",
    "memory_gb": 80,
    "architecture": "Ampere",
  },
  {
    "id": "L40S",
    "name": "NVIDIA L40S",
    "memory_gb": 48,
    "architecture": "Ada Lovelace",
  },
  {"id": "H100", "name": "NVIDIA H100", "memory_gb": 80, "architecture": "Hopper"},
  {"id": "H200", "name": "NVIDIA H200", "memory_gb": 141, "architecture": "Hopper"},
  {
    "id": "B200",
    "name": "NVIDIA B200",
    "memory_gb": 192,
    "architecture": "Blackwell",
  },
]


cuda_image = (
  modal.Image.from_registry("nvidia/cuda:13.1.1-devel-ubuntu24.04", add_python="3.12")
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


def get_cuda_image():
  return cuda_image


def get_triton_image():
  return triton_image


app = modal.App("kernel-orbit")


@dataclass
class KernelResult:
  """Unified result structure for both CUDA and Triton kernels."""

  # Status
  successful: bool = False
  kernel_type: str = ""  # "cuda" or "triton"
  error_message: str = ""

  # Timing (in milliseconds)
  compilation_time_ms: float = 0.0
  warmup_time_ms: float = 0.0
  execution_time_ms: float = 0.0
  execution_time_std_ms: float = 0.0
  min_execution_time_ms: float = 0.0
  max_execution_time_ms: float = 0.0
  total_time_ms: float = 0.0

  # Memory (in MB)
  gpu_memory_used_mb: float = 0.0
  gpu_memory_reserved_mb: float = 0.0
  peak_memory_mb: float = 0.0

  # GPU Information
  gpu_name: str = ""
  gpu_type_requested: str = ""
  gpu_compute_capability: str = ""
  gpu_memory_total_mb: float = 0.0
  gpu_temperature_c: float = 0.0
  gpu_power_draw_w: float = 0.0
  gpu_utilization_percent: float = 0.0
  gpu_count: int = 1

  # Benchmark Configuration
  warmup_runs: int = 0
  benchmark_runs: int = 0

  # Outputs
  kernel_output: str = ""
  compiler_output: str = ""
  profiler_output: str = ""

  # All timing samples
  timing_samples_ms: List[float] = field(default_factory=list)


def get_gpu_info() -> Dict[str, Any]:
  """Get detailed GPU information using nvidia-ml-py."""
  try:
    from pynvml import (
      nvmlInit,
      nvmlShutdown,
      nvmlDeviceGetHandleByIndex,
      nvmlDeviceGetName,
      nvmlDeviceGetMemoryInfo,
      nvmlDeviceGetTemperature,
      NVML_TEMPERATURE_GPU,
      nvmlDeviceGetPowerUsage,
      nvmlDeviceGetCudaComputeCapability,
      nvmlDeviceGetUtilizationRates,
    )

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)

    name = nvmlDeviceGetName(handle)
    if isinstance(name, bytes):
      name = name.decode("utf-8")

    info = {
      "name": name,
      "memory_total_mb": nvmlDeviceGetMemoryInfo(handle).total / (1024**2),
      "memory_used_mb": nvmlDeviceGetMemoryInfo(handle).used / (1024**2),
      "memory_free_mb": nvmlDeviceGetMemoryInfo(handle).free / (1024**2),
      "temperature_c": nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU),
      "power_draw_w": nvmlDeviceGetPowerUsage(handle) / 1000.0,
      "compute_capability": ".".join(
        map(str, nvmlDeviceGetCudaComputeCapability(handle))
      ),
      "utilization_percent": nvmlDeviceGetUtilizationRates(handle).gpu,
    }
    nvmlShutdown()
    return info
  except Exception as e:
    return {"error": str(e), "name": "Unknown"}


@app.function(gpu="T4", image=get_cuda_image(), timeout=600)
def run_cuda_on_t4(
  kernel_source: str, warmup_runs: int, benchmark_runs: int, enable_profiling: bool
) -> Dict:
  return _run_cuda_kernel_impl(
    kernel_source, "T4", warmup_runs, benchmark_runs, enable_profiling
  )


@app.function(gpu="L4", image=get_cuda_image(), timeout=600)
def run_cuda_on_l4(
  kernel_source: str, warmup_runs: int, benchmark_runs: int, enable_profiling: bool
) -> Dict:
  return _run_cuda_kernel_impl(
    kernel_source, "L4", warmup_runs, benchmark_runs, enable_profiling
  )


@app.function(gpu="A10G", image=get_cuda_image(), timeout=600)
def run_cuda_on_a10g(
  kernel_source: str, warmup_runs: int, benchmark_runs: int, enable_profiling: bool
) -> Dict:
  return _run_cuda_kernel_impl(
    kernel_source, "A10G", warmup_runs, benchmark_runs, enable_profiling
  )


@app.function(gpu="A100-40GB", image=get_cuda_image(), timeout=600)
def run_cuda_on_a100_40gb(
  kernel_source: str, warmup_runs: int, benchmark_runs: int, enable_profiling: bool
) -> Dict:
  return _run_cuda_kernel_impl(
    kernel_source, "A100-40GB", warmup_runs, benchmark_runs, enable_profiling
  )


@app.function(gpu="A100-80GB", image=get_cuda_image(), timeout=600)
def run_cuda_on_a100_80gb(
  kernel_source: str, warmup_runs: int, benchmark_runs: int, enable_profiling: bool
) -> Dict:
  return _run_cuda_kernel_impl(
    kernel_source, "A100-80GB", warmup_runs, benchmark_runs, enable_profiling
  )


@app.function(gpu="L40S", image=get_cuda_image(), timeout=600)
def run_cuda_on_l40s(
  kernel_source: str, warmup_runs: int, benchmark_runs: int, enable_profiling: bool
) -> Dict:
  return _run_cuda_kernel_impl(
    kernel_source, "L40S", warmup_runs, benchmark_runs, enable_profiling
  )


@app.function(gpu="H100", image=get_cuda_image(), timeout=600)
def run_cuda_on_h100(
  kernel_source: str, warmup_runs: int, benchmark_runs: int, enable_profiling: bool
) -> Dict:
  return _run_cuda_kernel_impl(
    kernel_source, "H100", warmup_runs, benchmark_runs, enable_profiling
  )


def _run_cuda_kernel_impl(
  kernel_source: str,
  gpu_type: str,
  warmup_runs: int,
  benchmark_runs: int,
  enable_profiling: bool,
) -> Dict[str, Any]:
  """Implementation for running CUDA kernels."""
  import statistics

  result = KernelResult()
  result.kernel_type = "cuda"
  result.gpu_type_requested = gpu_type
  result.warmup_runs = warmup_runs
  result.benchmark_runs = benchmark_runs

  total_start = time.perf_counter()

  try:
    gpu_info = get_gpu_info()
    result.gpu_name = gpu_info.get("name", "Unknown")
    result.gpu_compute_capability = gpu_info.get("compute_capability", "Unknown")
    result.gpu_memory_total_mb = gpu_info.get("memory_total_mb", 0)
    result.gpu_temperature_c = gpu_info.get("temperature_c", 0)
    result.gpu_power_draw_w = gpu_info.get("power_draw_w", 0)

    with tempfile.TemporaryDirectory() as tmpdir:
      source_file = Path(tmpdir) / "kernel.cu"
      exe_file = Path(tmpdir) / "kernel"

      source_file.write_text(kernel_source)

      compile_start = time.perf_counter()
      compile_result = subprocess.run(
        ["nvcc", str(source_file), "-o", str(exe_file), "-O3", "-lineinfo"],
        capture_output=True,
        text=True,
      )
      result.compilation_time_ms = (time.perf_counter() - compile_start) * 1000
      result.compiler_output = compile_result.stdout + compile_result.stderr

      if compile_result.returncode != 0:
        result.successful = False
        result.error_message = f"Compilation failed:\n{result.compiler_output}"
        result.total_time_ms = (time.perf_counter() - total_start) * 1000
        return asdict(result)

      warmup_start = time.perf_counter()
      for _ in range(warmup_runs):
        subprocess.run([str(exe_file)], capture_output=True)
      result.warmup_time_ms = (time.perf_counter() - warmup_start) * 1000

      times = []
      kernel_output = ""
      for _ in range(benchmark_runs):
        start = time.perf_counter()
        run_result = subprocess.run([str(exe_file)], capture_output=True, text=True)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        kernel_output = run_result.stdout + run_result.stderr

      result.timing_samples_ms = times
      result.kernel_output = kernel_output

      if times:
        result.execution_time_ms = statistics.mean(times)
        result.execution_time_std_ms = statistics.stdev(times) if len(times) > 1 else 0
        result.min_execution_time_ms = min(times)
        result.max_execution_time_ms = max(times)

      gpu_info_after = get_gpu_info()
      result.gpu_memory_used_mb = gpu_info_after.get("memory_used_mb", 0)
      result.gpu_utilization_percent = gpu_info_after.get("utilization_percent", 0)

      if enable_profiling:
        try:
          prof_result = subprocess.run(
            [
              "nvidia-smi",
              "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu,power.draw",
              "--format=csv",
            ],
            capture_output=True,
            text=True,
            timeout=10,
          )
          result.profiler_output = prof_result.stdout
        except:
          pass

      result.successful = True

  except Exception as e:
    import traceback

    result.successful = False
    result.error_message = f"{str(e)}\n{traceback.format_exc()}"

  result.total_time_ms = (time.perf_counter() - total_start) * 1000
  return asdict(result)


@app.function(gpu="T4", image=get_triton_image(), timeout=600)
def run_triton_on_t4(
  kernel_source: str, warmup_runs: int, benchmark_runs: int, enable_profiling: bool
) -> Dict:
  return _run_triton_kernel_impl(
    kernel_source, "T4", warmup_runs, benchmark_runs, enable_profiling
  )


@app.function(gpu="L4", image=get_triton_image(), timeout=600)
def run_triton_on_l4(
  kernel_source: str, warmup_runs: int, benchmark_runs: int, enable_profiling: bool
) -> Dict:
  return _run_triton_kernel_impl(
    kernel_source, "L4", warmup_runs, benchmark_runs, enable_profiling
  )


@app.function(gpu="A10G", image=get_triton_image(), timeout=600)
def run_triton_on_a10g(
  kernel_source: str, warmup_runs: int, benchmark_runs: int, enable_profiling: bool
) -> Dict:
  return _run_triton_kernel_impl(
    kernel_source, "A10G", warmup_runs, benchmark_runs, enable_profiling
  )


@app.function(gpu="A100-40GB", image=get_triton_image(), timeout=600)
def run_triton_on_a100_40gb(
  kernel_source: str, warmup_runs: int, benchmark_runs: int, enable_profiling: bool
) -> Dict:
  return _run_triton_kernel_impl(
    kernel_source, "A100", warmup_runs, benchmark_runs, enable_profiling
  )


@app.function(gpu="A100-80GB", image=get_triton_image(), timeout=600)
def run_triton_on_a100_80gb(
  kernel_source: str, warmup_runs: int, benchmark_runs: int, enable_profiling: bool
) -> Dict:
  return _run_triton_kernel_impl(
    kernel_source, "A100-80GB", warmup_runs, benchmark_runs, enable_profiling
  )


@app.function(gpu="L40S", image=get_triton_image(), timeout=600)
def run_triton_on_l40s(
  kernel_source: str, warmup_runs: int, benchmark_runs: int, enable_profiling: bool
) -> Dict:
  return _run_triton_kernel_impl(
    kernel_source, "L40S", warmup_runs, benchmark_runs, enable_profiling
  )


@app.function(gpu="H100", image=get_triton_image(), timeout=600)
def run_triton_on_h100(
  kernel_source: str, warmup_runs: int, benchmark_runs: int, enable_profiling: bool
) -> Dict:
  return _run_triton_kernel_impl(
    kernel_source, "H100", warmup_runs, benchmark_runs, enable_profiling
  )


def _run_triton_kernel_impl(
  kernel_source: str,
  gpu_type: str,
  warmup_runs: int,
  benchmark_runs: int,
  enable_profiling: bool,
) -> Dict[str, Any]:
  """Implementation for running Triton kernels."""
  import torch
  import statistics
  import io
  import contextlib

  result = KernelResult()
  result.kernel_type = "triton"
  result.gpu_type_requested = gpu_type
  result.warmup_runs = warmup_runs
  result.benchmark_runs = benchmark_runs

  total_start = time.perf_counter()

  try:
    import triton
    import triton.language as tl

    gpu_info = get_gpu_info()
    result.gpu_name = gpu_info.get("name", "Unknown")
    result.gpu_compute_capability = gpu_info.get("compute_capability", "Unknown")
    result.gpu_memory_total_mb = gpu_info.get("memory_total_mb", 0)
    result.gpu_temperature_c = gpu_info.get("temperature_c", 0)
    result.gpu_power_draw_w = gpu_info.get("power_draw_w", 0)

    torch.cuda.reset_peak_memory_stats()

    kernel_globals = {
      "torch": torch,
      "triton": triton,
      "tl": tl,
      "np": __import__("numpy"),
      "numpy": __import__("numpy"),
    }

    stdout_capture = io.StringIO()

    with contextlib.redirect_stdout(stdout_capture):
      exec(kernel_source, kernel_globals)

    run_fn = None
    for name in ["benchmark", "benchmark_kernel", "main", "run", "test"]:
      if name in kernel_globals and callable(kernel_globals[name]):
        run_fn = kernel_globals[name]
        break

    if run_fn is None:
      result.kernel_output = stdout_capture.getvalue()
      result.successful = True
      result.total_time_ms = (time.perf_counter() - total_start) * 1000

      result.gpu_memory_used_mb = torch.cuda.memory_allocated() / (1024**2)
      result.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)

      return asdict(result)

    warmup_start = time.perf_counter()
    for _ in range(warmup_runs):
      with contextlib.redirect_stdout(stdout_capture):
        run_fn()
      torch.cuda.synchronize()
    result.warmup_time_ms = (time.perf_counter() - warmup_start) * 1000

    times = []
    for _ in range(benchmark_runs):
      torch.cuda.synchronize()
      start = time.perf_counter()
      with contextlib.redirect_stdout(stdout_capture):
        run_fn()
      torch.cuda.synchronize()
      elapsed = (time.perf_counter() - start) * 1000
      times.append(elapsed)

    result.timing_samples_ms = times
    result.kernel_output = stdout_capture.getvalue()

    if times:
      result.execution_time_ms = statistics.mean(times)
      result.execution_time_std_ms = statistics.stdev(times) if len(times) > 1 else 0
      result.min_execution_time_ms = min(times)
      result.max_execution_time_ms = max(times)

    result.gpu_memory_used_mb = torch.cuda.memory_allocated() / (1024**2)
    result.gpu_memory_reserved_mb = torch.cuda.memory_reserved() / (1024**2)
    result.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)

    gpu_info_after = get_gpu_info()
    result.gpu_utilization_percent = gpu_info_after.get("utilization_percent", 0)

    result.successful = True

  except Exception as e:
    import traceback

    result.successful = False
    result.error_message = f"{str(e)}\n{traceback.format_exc()}"

  result.total_time_ms = (time.perf_counter() - total_start) * 1000
  return asdict(result)


@app.function(image=modal.Image.debian_slim())
def list_available_gpus() -> List[Dict[str, Any]]:
  """Return list of available GPU configurations."""
  return AVAILABLE_GPUS


@app.function(gpu="T4", image=cuda_image, timeout=60)
def warmup_cuda_t4() -> str:
  """Warmup CUDA image on T4."""
  return "CUDA image ready on T4"


@app.function(gpu="T4", image=triton_image, timeout=60)
def warmup_triton_t4() -> str:
  """Warmup Triton image on T4."""
  return "Triton image ready on T4"


@app.local_entrypoint()
def main(
  kernel_file: str = None,
  kernel_type: str = "auto",
  gpu: str = "T4",
  warmup: int = 3,
  benchmark: int = 10,
  profile: bool = True,
  output_json: str = None,
  list_gpus: bool = False,
  warmup_images: bool = False,
):
  """
  Run a CUDA or Triton kernel on Modal.

  Args:
      kernel_file: Path to the kernel file (.cu or .py)
      kernel_type: "cuda", "triton", or "auto" (detect from extension)
      gpu: GPU type
      warmup: Number of warmup runs
      benchmark: Number of benchmark runs
      profile: Enable profiling
      output_json: Output JSON file path
      list_gpus: Just list available GPUs
      warmup_images: Pre-build Docker images (run once for faster subsequent runs)
  """
  if warmup_images:
    print("Pre-building Modal images (this only needs to be done once)...")
    print()
    print("Building CUDA image...")
    result1 = warmup_cuda_t4.remote()
    print(f"  {result1}")
    print("Building Triton image...")
    result2 = warmup_triton_t4.remote()
    print(f"  {result2}")
    print()
    print("Images are now cached! Subsequent kernel runs will be much faster.")
    return

  if list_gpus:
    gpus = list_available_gpus.remote()
    print(json.dumps(gpus, indent=2))
    return

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

  print(f"Running {kernel_type.upper()} kernel on Modal with {gpu}...")
  print(f"   File: {kernel_file}")
  print(f"   Warmup runs: {warmup}")
  print(f"   Benchmark runs: {benchmark}")
  print()

  gpu_runners = {
    "cuda": {
      "T4": run_cuda_on_t4,
      "L4": run_cuda_on_l4,
      "A10G": run_cuda_on_a10g,
      "A100-40GB": run_cuda_on_a100_40gb,
      "A100-80GB": run_cuda_on_a100_80gb,
      "L40S": run_cuda_on_l40s,
      "H100": run_cuda_on_h100,
    },
    "triton": {
      "T4": run_triton_on_t4,
      "L4": run_triton_on_l4,
      "A10G": run_triton_on_a10g,
      "A100-40GB": run_triton_on_a100_40gb,
      "A100-80GB": run_triton_on_a100_80gb,
      "L40S": run_triton_on_l40s,
      "H100": run_triton_on_h100,
    },
  }

  runner = gpu_runners.get(kernel_type, {}).get(gpu)
  if not runner:
    print(f"Error: No runner for {kernel_type} on {gpu}")
    sys.exit(1)

  result = runner.remote(kernel_source, warmup, benchmark, profile)

  if result["successful"]:
    print("Kernel executed successfully!")
    print("GPU Information:")
    print(f"  Name: {result['gpu_name']}")
    print(f"  Requested: {result['gpu_type_requested']}")
    print(f"  Compute Capability: {result['gpu_compute_capability']}")
    print(f"  Total Memory: {result['gpu_memory_total_mb']:.0f} MB")
    print()
    print("Timing:")
    if result.get("compilation_time_ms"):
      print(f"  Compilation: {result['compilation_time_ms']:.2f} ms")
    print(f"  Warmup ({result['warmup_runs']} runs): {result['warmup_time_ms']:.2f} ms")
    print(
      f"  Execution ({result['benchmark_runs']} runs): {result['execution_time_ms']:.2f} ± {result['execution_time_std_ms']:.2f} ms"
    )
    print(f"  Min: {result['min_execution_time_ms']:.2f} ms")
    print(f"  Max: {result['max_execution_time_ms']:.2f} ms")
    print(f"  Total: {result['total_time_ms']:.2f} ms")
    print()
    print("Memory:")
    print(f"  Used: {result['gpu_memory_used_mb']:.2f} MB")
    print(f"  Peak: {result['peak_memory_mb']:.2f} MB")
    print()
    print("GPU Status:")
    print(f"  Utilization: {result['gpu_utilization_percent']}%")
    print(f"  Temperature: {result['gpu_temperature_c']}°C")
    print(f"  Power Draw: {result['gpu_power_draw_w']:.1f} W")

    if result["kernel_output"]:
      print()
      print("KERNEL OUTPUT")
      print(result["kernel_output"])

    if result.get("profiler_output"):
      print()
      print("PROFILER OUTPUT")
      print(result["profiler_output"])
  else:
    print("Kernel execution failed!")
    print()
    print("Error:")
    print(result["error_message"])

    if result.get("compiler_output"):
      print()
      print("Compiler Output:")
      print(result["compiler_output"])

  if output_json:
    Path(output_json).write_text(json.dumps(result, indent=2))
    print(f"\nResults saved to {output_json}")

  return result
