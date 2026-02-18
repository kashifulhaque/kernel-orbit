"""
Example Triton Vector Addition Kernel
This file demonstrates a simple Triton kernel that can be run on Modal.com
"""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
  x_ptr,  # Pointer to first input vector
  y_ptr,  # Pointer to second input vector
  output_ptr,  # Pointer to output vector
  n_elements,  # Number of elements
  BLOCK_SIZE: tl.constexpr,  # Block size (compile-time constant)
):
  """
  Triton kernel for element-wise vector addition.
  Each program instance processes BLOCK_SIZE elements.
  """
  # Get the program ID (which block we're processing)
  pid = tl.program_id(axis=0)

  # Calculate the starting offset for this block
  block_start = pid * BLOCK_SIZE

  # Create a range of offsets for this block
  offsets = block_start + tl.arange(0, BLOCK_SIZE)

  # Create a mask for bounds checking
  mask = offsets < n_elements

  # Load data from global memory
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)

  # Compute addition
  output = x + y

  # Store result to global memory
  tl.store(output_ptr + offsets, output, mask=mask)


def add_vectors(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  """
  Add two vectors using the Triton kernel.
  """
  assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA"
  assert x.shape == y.shape, "Tensors must have the same shape"

  output = torch.empty_like(x)
  n_elements = output.numel()

  # Define the grid (number of blocks to launch)
  BLOCK_SIZE = 1024
  grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

  # Launch the kernel
  add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

  return output


def benchmark():
  """
  Benchmark function that runs the vector addition kernel.
  This is the entry point for the Modal runner.
  """
  # Configuration
  n_elements = 1 << 20  # 1M elements

  print(f"Vector Addition Benchmark")
  print(f"=" * 50)
  print(f"Elements: {n_elements:,}")
  print(f"Memory per vector: {n_elements * 4 / 1024 / 1024:.2f} MB")
  print()

  # Create input tensors on GPU
  x = torch.rand(n_elements, device="cuda", dtype=torch.float32)
  y = torch.rand(n_elements, device="cuda", dtype=torch.float32)

  # Warmup
  output = add_vectors(x, y)
  torch.cuda.synchronize()

  # Benchmark Triton
  import time

  n_runs = 100
  torch.cuda.synchronize()
  start = time.perf_counter()
  for _ in range(n_runs):
    output = add_vectors(x, y)
  torch.cuda.synchronize()
  triton_time = (time.perf_counter() - start) / n_runs * 1000

  # Benchmark PyTorch
  torch.cuda.synchronize()
  start = time.perf_counter()
  for _ in range(n_runs):
    expected = x + y
  torch.cuda.synchronize()
  torch_time = (time.perf_counter() - start) / n_runs * 1000

  # Verify correctness
  expected = x + y
  if torch.allclose(output, expected):
    print("Verification: PASSED")
  else:
    max_diff = (output - expected).abs().max()
    print(f"Verification: FAILED (max diff: {max_diff})")

  print()
  print("Timing Results:")
  print(f"  Triton:  {triton_time:.4f} ms")
  print(f"  PyTorch: {torch_time:.4f} ms")
  print(f"  Ratio:   {torch_time / triton_time:.2f}x")

  # Calculate bandwidth
  bytes_transferred = 3 * n_elements * 4  # 2 reads + 1 write
  bandwidth_triton = bytes_transferred / (triton_time / 1000) / 1e9
  bandwidth_torch = bytes_transferred / (torch_time / 1000) / 1e9

  print()
  print("Memory Bandwidth:")
  print(f"  Triton:  {bandwidth_triton:.2f} GB/s")
  print(f"  PyTorch: {bandwidth_torch:.2f} GB/s")

  print()
  print(f"Sample output: {output[:5].tolist()}")

  return output


# Entry point for Modal runner
if __name__ == "__main__":
  benchmark()
