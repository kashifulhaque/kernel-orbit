"""
Modal script to run Triton Python kernels with comprehensive profiling and metrics.
"""

import json
import time
import modal
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List


def get_triton_image():
  """Create a Modal image with Triton and PyTorch."""
  return modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "triton",
    "numpy",
    "pynvml",
    "py3nvml",
  )


app = modal.App("triton-kernel-runner")


@dataclass
class TritonKernelMetrics:
  """Comprehensive metrics from Triton kernel execution."""

  # Timing
  warmup_time_ms: float = 0.0
  execution_time_ms: float = 0.0
  execution_time_std_ms: float = 0.0
  min_execution_time_ms: float = 0.0
  max_execution_time_ms: float = 0.0
  total_time_ms: float = 0.0

  # Memory
  gpu_memory_allocated_mb: float = 0.0
  gpu_memory_reserved_mb: float = 0.0
  peak_memory_mb: float = 0.0

  # GPU Info
  gpu_name: str = ""
  gpu_compute_capability: str = ""
  gpu_memory_total_mb: float = 0.0
  gpu_temperature_c: float = 0.0
  gpu_power_draw_w: float = 0.0
  gpu_utilization_percent: float = 0.0

  # Triton Kernel Info
  kernel_name: str = ""
  num_warps: int = 0
  num_stages: int = 0
  shared_memory_bytes: int = 0

  # Performance metrics
  memory_throughput_gb_s: float = 0.0
  compute_throughput_tflops: float = 0.0

  # Triton autotuning
  best_config: str = ""
  all_configs_tested: List[str] = None

  # Execution details
  warmup_runs: int = 0
  benchmark_runs: int = 0
  successful: bool = False
  error_message: str = ""

  # Output
  kernel_output: str = ""
  profiler_output: str = ""

  def __post_init__(self):
    if self.all_configs_tested is None:
      self.all_configs_tested = []


def get_gpu_info() -> Dict[str, Any]:
  """Get detailed GPU information using pynvml."""
  try:
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    info = {
      "name": pynvml.nvmlDeviceGetName(handle),
      "memory_total_mb": pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**2),
      "memory_used_mb": pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**2),
      "memory_free_mb": pynvml.nvmlDeviceGetMemoryInfo(handle).free / (1024**2),
      "temperature_c": pynvml.nvmlDeviceGetTemperature(
        handle, pynvml.NVML_TEMPERATURE_GPU
      ),
      "power_draw_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
      "compute_capability": ".".join(
        map(str, pynvml.nvmlDeviceGetCudaComputeCapability(handle))
      ),
      "utilization_percent": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
    }
    pynvml.nvmlShutdown()
    return info
  except Exception as e:
    return {"error": str(e)}


def get_torch_memory_stats() -> Dict[str, float]:
  """Get PyTorch CUDA memory statistics."""
  try:
    import torch

    if torch.cuda.is_available():
      return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
        "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
        "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
      }
  except:
    pass
  return {}


@app.function(gpu="T4", image=get_triton_image(), timeout=600)
def run_triton_kernel(
  kernel_source: str,
  gpu_type: str = "T4",
  gpu_count: int = 1,
  warmup_runs: int = 3,
  benchmark_runs: int = 10,
  enable_profiling: bool = True,
  input_sizes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
  """
  Run a Triton kernel on Modal with comprehensive metrics.

  The kernel source should define:
  1. A Triton kernel function decorated with @triton.jit
  2. A `benchmark_kernel()` function that returns timing info

  Or the extension will auto-generate a benchmark wrapper.
  """
  import torch
  import triton
  import statistics
  import io
  import contextlib

  metrics = TritonKernelMetrics()
  metrics.warmup_runs = warmup_runs
  metrics.benchmark_runs = benchmark_runs

  total_start = time.perf_counter()

  try:
    gpu_info = get_gpu_info()
    metrics.gpu_name = gpu_info.get("name", "Unknown")
    metrics.gpu_compute_capability = gpu_info.get("compute_capability", "Unknown")
    metrics.gpu_memory_total_mb = gpu_info.get("memory_total_mb", 0)
    metrics.gpu_temperature_c = gpu_info.get("temperature_c", 0)
    metrics.gpu_power_draw_w = gpu_info.get("power_draw_w", 0)

    torch.cuda.reset_peak_memory_stats()

    kernel_globals = {
      "torch": torch,
      "triton": triton,
      "tl": triton.language,
      "np": __import__("numpy"),
    }

    stdout_capture = io.StringIO()

    with contextlib.redirect_stdout(stdout_capture):
      exec(kernel_source, kernel_globals)

    if "benchmark_kernel" in kernel_globals:
      benchmark_fn = kernel_globals["benchmark_kernel"]
    elif "main" in kernel_globals:
      benchmark_fn = kernel_globals["main"]
    else:
      kernel_fn = None
      for name, obj in kernel_globals.items():
        if hasattr(obj, "__wrapped__") or (
          hasattr(obj, "run") and "triton" in str(type(obj))
        ):
          kernel_fn = obj
          metrics.kernel_name = name
          break

      if kernel_fn is None:
        with contextlib.redirect_stdout(stdout_capture):
          exec(kernel_source, kernel_globals)
        metrics.kernel_output = stdout_capture.getvalue()
        metrics.successful = True
        metrics.total_time_ms = (time.perf_counter() - total_start) * 1000
        return asdict(metrics)

    warmup_start = time.perf_counter()
    for _ in range(warmup_runs):
      with contextlib.redirect_stdout(stdout_capture):
        if "benchmark_fn" in dir():
          benchmark_fn()
    metrics.warmup_time_ms = (time.perf_counter() - warmup_start) * 1000
    torch.cuda.synchronize()

    times = []
    for _ in range(benchmark_runs):
      torch.cuda.synchronize()
      start = time.perf_counter()
      with contextlib.redirect_stdout(stdout_capture):
        if "benchmark_fn" in dir():
          result = benchmark_fn()
      torch.cuda.synchronize()
      elapsed = (time.perf_counter() - start) * 1000
      times.append(elapsed)

    if times:
      metrics.execution_time_ms = statistics.mean(times)
      metrics.execution_time_std_ms = statistics.stdev(times) if len(times) > 1 else 0
      metrics.min_execution_time_ms = min(times)
      metrics.max_execution_time_ms = max(times)

    mem_stats = get_torch_memory_stats()
    metrics.gpu_memory_allocated_mb = mem_stats.get("allocated_mb", 0)
    metrics.gpu_memory_reserved_mb = mem_stats.get("reserved_mb", 0)
    metrics.peak_memory_mb = mem_stats.get("max_allocated_mb", 0)

    gpu_info_after = get_gpu_info()
    metrics.gpu_utilization_percent = gpu_info_after.get("utilization_percent", 0)

    metrics.kernel_output = stdout_capture.getvalue()
    metrics.successful = True

  except Exception as e:
    import traceback

    metrics.successful = False
    metrics.error_message = f"{str(e)}\n\n{traceback.format_exc()}"

  metrics.total_time_ms = (time.perf_counter() - total_start) * 1000

  return asdict(metrics)


@app.local_entrypoint()
def main(
  kernel_file: str,
  gpu: str = "T4",
  gpu_count: int = 1,
  warmup: int = 3,
  benchmark: int = 10,
  profile: bool = True,
  output_json: str = None,
):
  """
  Run a Triton kernel on Modal.

  Args:
      kernel_file: Path to the .py file containing Triton kernel
      gpu: GPU type (T4, L4, A10G, A100-40GB, A100-80GB, L40S, H100, H200, B200)
      gpu_count: Number of GPUs
      warmup: Warmup runs
      benchmark: Benchmark runs
      profile: Enable profiling
      output_json: Optional output JSON file path
  """
  kernel_source = Path(kernel_file).read_text()

  gpu_spec = f"{gpu}:{gpu_count}" if gpu_count > 1 else gpu

  print(f"Running Triton kernel on Modal with {gpu_spec}...")
  print(f"   Warmup runs: {warmup}")
  print(f"   Benchmark runs: {benchmark}")
  print()

  result = run_triton_kernel.remote(
    kernel_source=kernel_source,
    gpu_type=gpu,
    gpu_count=gpu_count,
    warmup_runs=warmup,
    benchmark_runs=benchmark,
    enable_profiling=profile,
  )

  if result["successful"]:
    print("Kernel executed successfully!")
    print()
    print("Results:")
    print(f"   GPU: {result['gpu_name']}")
    print(f"   Compute Capability: {result['gpu_compute_capability']}")
    print(f"   Warmup Time: {result['warmup_time_ms']:.2f} ms")
    print(
      f"   Execution Time: {result['execution_time_ms']:.2f} ± {result['execution_time_std_ms']:.2f} ms"
    )
    print(
      f"   Min/Max: {result['min_execution_time_ms']:.2f} / {result['max_execution_time_ms']:.2f} ms"
    )
    print(f"   GPU Memory Used: {result['gpu_memory_allocated_mb']:.2f} MB")
    print(f"   Peak Memory: {result['peak_memory_mb']:.2f} MB")
    print(f"   Total Time: {result['total_time_ms']:.2f} ms")

    if result["kernel_output"]:
      print()
      print("Kernel Output:")
      print(result["kernel_output"])
  else:
    print("Kernel execution failed!")
    print(f"   Error: {result['error_message']}")

  if output_json:
    Path(output_json).write_text(json.dumps(result, indent=2))
    print(f"\nResults saved to {output_json}")

  return result


if __name__ == "__main__":
  print("Use: modal run run_triton_kernel.py --kernel-file <file> [options]")
