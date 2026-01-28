"""
Example Triton Matrix Multiplication Kernel
Demonstrates a more complex Triton kernel with autotuning
"""

import torch
import triton
import triton.language as tl


# Define autotuning configurations
@triton.autotune(
  configs=[
    triton.Config(
      {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
      },
      num_stages=3,
      num_warps=8,
    ),
    triton.Config(
      {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
      },
      num_stages=4,
      num_warps=4,
    ),
    triton.Config(
      {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
      },
      num_stages=4,
      num_warps=4,
    ),
    triton.Config(
      {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
      },
      num_stages=4,
      num_warps=4,
    ),
    triton.Config(
      {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
      },
      num_stages=4,
      num_warps=4,
    ),
    triton.Config(
      {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
      },
      num_stages=4,
      num_warps=4,
    ),
    triton.Config(
      {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
      },
      num_stages=5,
      num_warps=2,
    ),
    triton.Config(
      {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
      },
      num_stages=5,
      num_warps=2,
    ),
  ],
  key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
  # Pointers to matrices
  a_ptr,
  b_ptr,
  c_ptr,
  # Matrix dimensions
  M,
  N,
  K,
  # Strides
  stride_am,
  stride_ak,
  stride_bk,
  stride_bn,
  stride_cm,
  stride_cn,
  # Block sizes (from autotuning)
  BLOCK_SIZE_M: tl.constexpr,
  BLOCK_SIZE_N: tl.constexpr,
  BLOCK_SIZE_K: tl.constexpr,
  GROUP_SIZE_M: tl.constexpr,
):
  """
  Compute C = A @ B

  A: (M, K)
  B: (K, N)
  C: (M, N)
  """
  # Program ID
  pid = tl.program_id(axis=0)

  # Number of blocks in each dimension
  num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
  num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
  num_pid_in_group = GROUP_SIZE_M * num_pid_n

  # Group ID and position within group
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * GROUP_SIZE_M
  group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

  # Block indices
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m

  # Offsets for A and B
  offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
  offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
  offs_k = tl.arange(0, BLOCK_SIZE_K)

  # Pointers for A and B tiles
  a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
  b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

  # Initialize accumulator
  accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

  # Main loop over K dimension
  for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    # Load A and B tiles
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

    # Compute matrix multiply and accumulate
    accumulator += tl.dot(a, b)

    # Advance pointers
    a_ptrs += BLOCK_SIZE_K * stride_ak
    b_ptrs += BLOCK_SIZE_K * stride_bk

  # Convert to output dtype
  c = accumulator.to(tl.float16)

  # Compute output pointers
  offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
  offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
  c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

  # Store result
  tl.store(c_ptrs, c, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """
  Compute matrix multiplication using Triton kernel.
  """
  assert a.shape[1] == b.shape[0], "Incompatible dimensions"
  assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"

  M, K = a.shape
  K, N = b.shape

  # Allocate output
  c = torch.empty((M, N), device=a.device, dtype=torch.float16)

  # Define grid
  grid = lambda META: (
    triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
  )

  # Launch kernel
  matmul_kernel[grid](
    a,
    b,
    c,
    M,
    N,
    K,
    a.stride(0),
    a.stride(1),
    b.stride(0),
    b.stride(1),
    c.stride(0),
    c.stride(1),
  )

  return c


def benchmark():
  """
  Benchmark matrix multiplication kernel.
  """
  print("Matrix Multiplication Benchmark")
  print("=" * 50)

  # Test sizes
  sizes = [
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
  ]

  for M, N, K in sizes:
    print(f"\nSize: ({M}, {K}) @ ({K}, {N})")

    # Create matrices
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    # Warmup
    triton_output = matmul(a, b)
    torch.cuda.synchronize()

    # Benchmark Triton
    import time

    n_runs = 100

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
      triton_output = matmul(a, b)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / n_runs * 1000

    # Benchmark PyTorch
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
      torch_output = torch.matmul(a, b)
    torch.cuda.synchronize()
    torch_time = (time.perf_counter() - start) / n_runs * 1000

    # Verify correctness
    torch_output = torch.matmul(a, b)
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
      print("  ✅ Verification: PASSED")
    else:
      max_diff = (triton_output - torch_output).abs().max()
      print(f"  ⚠️ Verification: max diff = {max_diff}")

    # Calculate TFLOPS
    flops = 2 * M * N * K
    triton_tflops = flops / (triton_time / 1000) / 1e12
    torch_tflops = flops / (torch_time / 1000) / 1e12

    print(f"  Triton:  {triton_time:.4f} ms ({triton_tflops:.2f} TFLOPS)")
    print(f"  PyTorch: {torch_time:.4f} ms ({torch_tflops:.2f} TFLOPS)")
    print(f"  Ratio:   {torch_time / triton_time:.2f}x")

  print()
  print("✅ Benchmark completed!")

  return triton_output


# Entry point
if __name__ == "__main__":
  benchmark()
