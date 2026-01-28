/**
 * Example CUDA Vector Addition Kernel
 * This file demonstrates a simple CUDA kernel that can be run on Modal.com
 */

#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel for vector addition
__global__ void vectorAdd(const float *a, const float *b, float *c, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
  {
    c[idx] = a[idx] + b[idx];
  }
}

// Helper function to check CUDA errors
#define CUDA_CHECK(call)                                               \
  do                                                                   \
  {                                                                    \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess)                                            \
    {                                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

int main()
{
  // Vector size
  const int N = 1 << 20; // 1M elements
  const size_t size = N * sizeof(float);

  printf("Vector Addition: %d elements\n", N);
  printf("Memory per vector: %.2f MB\n", size / (1024.0 * 1024.0));

  // Host vectors
  float *h_a = (float *)malloc(size);
  float *h_b = (float *)malloc(size);
  float *h_c = (float *)malloc(size);

  // Initialize vectors
  for (int i = 0; i < N; i++)
  {
    h_a[i] = sinf(i) * sinf(i);
    h_b[i] = cosf(i) * cosf(i);
  }

  // Device vectors
  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, size));
  CUDA_CHECK(cudaMalloc(&d_b, size));
  CUDA_CHECK(cudaMalloc(&d_c, size));

  // Copy to device
  CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

  // Launch kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  printf("Grid: %d blocks, %d threads/block\n", blocksPerGrid, threadsPerBlock);

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // Record start time
  CUDA_CHECK(cudaEventRecord(start));

  // Launch kernel
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

  // Record stop time
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  // Calculate elapsed time
  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Kernel execution time: %.4f ms\n", milliseconds);

  // Copy result back
  CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

  // Verify result
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
  {
    float expected = h_a[i] + h_b[i];
    maxError = fmaxf(maxError, fabsf(h_c[i] - expected));
  }
  printf("Max error: %e\n", maxError);

  // Calculate bandwidth
  float bandwidth = 3.0f * size / (milliseconds / 1000.0f) / 1e9;
  printf("Effective bandwidth: %.2f GB/s\n", bandwidth);

  // Cleanup
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  free(h_a);
  free(h_b);
  free(h_c);

  printf("\n✅ Vector addition completed successfully!\n");

  return 0;
}
