"""
Modal script to run CUDA C++ kernels with comprehensive profiling and metrics.
"""

import os
import sys
import json
import time
import modal
import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

# Force UTF-8 for stdout/stderr on Windows to avoid charmap encoding errors
if sys.platform == "win32":
  os.environ.setdefault("PYTHONUTF8", "1")
  if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
  if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def get_cuda_image(cuda_version: str = "13.1.1"):
  return (
    modal.Image.from_registry(
      f"nvidia/cuda:{cuda_version}-devel-ubuntu24.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("build-essential", "cmake", "ninja-build")
    .uv_pip_install("nvidia-ml-py", "numpy")
  )


app = modal.App("cuda-kernel-runner")


@dataclass
class KernelMetrics:
  """Comprehensive metrics from kernel execution."""

  # Timing
  compilation_time_ms: float = 0.0
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

  # Kernel Info
  kernel_name: str = ""
  grid_size: str = ""
  block_size: str = ""
  shared_memory_bytes: int = 0
  registers_per_thread: int = 0

  # Performance
  theoretical_occupancy: float = 0.0
  achieved_occupancy: float = 0.0
  memory_throughput_gb_s: float = 0.0
  compute_throughput_gflops: float = 0.0

  # Execution details
  warmup_runs: int = 0
  benchmark_runs: int = 0
  successful: bool = False
  error_message: str = ""

  # Profiler output
  profiler_output: str = ""
  nvcc_output: str = ""
  kernel_output: str = ""


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


def compile_cuda_kernel(
  source_path: str, output_path: str, extra_flags: List[str] = None
) -> tuple:
  """Compile CUDA kernel and return compilation time and output."""
  flags = [
    "nvcc",
    source_path,
    "-o",
    output_path,
    "-O3",
    "-lineinfo",
    "--generate-line-info",
    "-Xcompiler",
    "-fPIC",
  ]

  if extra_flags:
    flags.extend(extra_flags)

  start_time = time.perf_counter()
  result = subprocess.run(flags, capture_output=True, text=True)
  compilation_time = (time.perf_counter() - start_time) * 1000

  return compilation_time, result.returncode, result.stdout + result.stderr


def run_cuda_executable(exe_path: str, warmup_runs: int, benchmark_runs: int) -> tuple:
  """Run compiled CUDA executable and collect timing."""
  times = []
  output = ""

  warmup_start = time.perf_counter()
  for _ in range(warmup_runs):
    subprocess.run([exe_path], capture_output=True)
  warmup_time = (time.perf_counter() - warmup_start) * 1000

  for _ in range(benchmark_runs):
    start = time.perf_counter()
    result = subprocess.run([exe_path], capture_output=True, text=True)
    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)
    output = result.stdout + result.stderr

  return times, warmup_time, output


def run_nvprof_analysis(exe_path: str) -> str:
  """Run nsys profiling on the executable."""
  try:
    result = subprocess.run(
      ["nsys", "nvprof", "--print-gpu-trace", exe_path],
      capture_output=True,
      text=True,
      timeout=60,
    )
    return result.stdout + result.stderr
  except subprocess.TimeoutExpired:
    return "Profiling timed out"
  except FileNotFoundError:
    try:
      result = subprocess.run(
        ["nvidia-smi", "dmon", "-s", "pucvmet", "-d", "1", "-c", "5"],
        capture_output=True,
        text=True,
        timeout=30,
      )
      return f"nvidia-smi monitoring:\n{result.stdout}"
    except:
      return "Profiler not available"


@app.function(
  gpu=modal.gpu.T4(),  # Will be overridden by caller
  image=get_cuda_image(),
  timeout=600,
)
def run_cuda_kernel(
  kernel_source: str,
  gpu_type: str = "T4",
  gpu_count: int = 1,
  warmup_runs: int = 3,
  benchmark_runs: int = 10,
  enable_profiling: bool = True,
  extra_compile_flags: Optional[List[str]] = None,
  input_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
  """
  Compile and run a CUDA kernel on Modal with comprehensive metrics.

  Args:
      kernel_source: The CUDA C++ source code
      gpu_type: GPU type string (T4, L4, A100, H100, etc.)
      gpu_count: Number of GPUs
      warmup_runs: Number of warmup iterations
      benchmark_runs: Number of benchmark iterations
      enable_profiling: Whether to run profiler
      extra_compile_flags: Additional nvcc flags
      input_data: Optional input data configuration

  Returns:
      Dictionary containing all metrics and outputs
  """
  import statistics

  metrics = KernelMetrics()
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

    with tempfile.TemporaryDirectory() as tmpdir:
      source_file = Path(tmpdir) / "kernel.cu"
      exe_file = Path(tmpdir) / "kernel"

      source_file.write_text(kernel_source)

      compilation_time, retcode, nvcc_output = compile_cuda_kernel(
        str(source_file), str(exe_file), extra_compile_flags
      )
      metrics.compilation_time_ms = compilation_time
      metrics.nvcc_output = nvcc_output

      if retcode != 0:
        metrics.successful = False
        metrics.error_message = f"Compilation failed:\n{nvcc_output}"
        return asdict(metrics)

      times, warmup_time, kernel_output = run_cuda_executable(
        str(exe_file), warmup_runs, benchmark_runs
      )
      metrics.warmup_time_ms = warmup_time
      metrics.kernel_output = kernel_output

      if times:
        metrics.execution_time_ms = statistics.mean(times)
        metrics.execution_time_std_ms = statistics.stdev(times) if len(times) > 1 else 0
        metrics.min_execution_time_ms = min(times)
        metrics.max_execution_time_ms = max(times)

      gpu_info_after = get_gpu_info()
      metrics.gpu_memory_allocated_mb = gpu_info_after.get("memory_used_mb", 0)
      metrics.gpu_utilization_percent = gpu_info_after.get("utilization_percent", 0)

      if enable_profiling:
        metrics.profiler_output = run_nvprof_analysis(str(exe_file))

      metrics.successful = True

  except Exception as e:
    metrics.successful = False
    metrics.error_message = str(e)

  metrics.total_time_ms = (time.perf_counter() - total_start) * 1000

  return asdict(metrics)


def create_gpu_runner(gpu_spec: str):
  """Create a Modal function with specific GPU configuration."""

  @app.function(gpu=gpu_spec, image=get_cuda_image(), timeout=600)
  def gpu_runner(
    kernel_source: str,
    warmup_runs: int = 3,
    benchmark_runs: int = 10,
    enable_profiling: bool = True,
    extra_compile_flags: Optional[List[str]] = None,
  ) -> Dict[str, Any]:
    return run_cuda_kernel.local(
      kernel_source=kernel_source,
      warmup_runs=warmup_runs,
      benchmark_runs=benchmark_runs,
      enable_profiling=enable_profiling,
      extra_compile_flags=extra_compile_flags,
    )

  return gpu_runner


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
  Run a CUDA kernel on Modal.

  Args:
      kernel_file: Path to the .cu file
      gpu: GPU type (T4, L4, A10G, A100-40GB, A100-80GB, L40S, H100, H200, B200)
      gpu_count: Number of GPUs
      warmup: Warmup runs
      benchmark: Benchmark runs
      profile: Enable profiling
      output_json: Optional output JSON file path
  """
  kernel_source = Path(kernel_file).read_text()

  gpu_spec = f"{gpu}:{gpu_count}" if gpu_count > 1 else gpu

  print(f"Running kernel on Modal with {gpu_spec}...")
  print(f"   Warmup runs: {warmup}")
  print(f"   Benchmark runs: {benchmark}")
  print(f"   Profiling: {'enabled' if profile else 'disabled'}")
  print()

  result = run_cuda_kernel.remote(
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
    print(f"   Compilation Time: {result['compilation_time_ms']:.2f} ms")
    print(f"   Warmup Time: {result['warmup_time_ms']:.2f} ms")
    print(
      f"   Execution Time: {result['execution_time_ms']:.2f} ± {result['execution_time_std_ms']:.2f} ms"
    )
    print(
      f"   Min/Max: {result['min_execution_time_ms']:.2f} / {result['max_execution_time_ms']:.2f} ms"
    )
    print(f"   GPU Memory Used: {result['gpu_memory_allocated_mb']:.2f} MB")
    print(f"   Total Time: {result['total_time_ms']:.2f} ms")

    if result["kernel_output"]:
      print()
      print("Kernel Output:")
      print(result["kernel_output"])

    if result["profiler_output"]:
      print()
      print("Profiler Output:")
      print(result["profiler_output"])
  else:
    print("Kernel execution failed!")
    print(f"   Error: {result['error_message']}")
    if result["nvcc_output"]:
      print()
      print("NVCC Output:")
      print(result["nvcc_output"])

  if output_json:
    Path(output_json).write_text(json.dumps(result, indent=2))
    print(f"\nResults saved to {output_json}")

  return result


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Run CUDA kernel on Modal")
  parser.add_argument("kernel_file", help="Path to .cu file")
  parser.add_argument("--gpu", default="T4", help="GPU type")
  parser.add_argument("--gpu-count", type=int, default=1, help="GPU count")
  parser.add_argument("--warmup", type=int, default=3, help="Warmup runs")
  parser.add_argument("--benchmark", type=int, default=10, help="Benchmark runs")
  parser.add_argument("--no-profile", action="store_true", help="Disable profiling")
  parser.add_argument("--output-json", help="Output JSON file")

  args = parser.parse_args()

  print("Use: modal run run_cuda_kernel.py --kernel-file <file> [options]")
