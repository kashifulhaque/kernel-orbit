/**
 * Example CUDA Matrix Multiplication Kernel
 * Demonstrates a more complex kernel with shared memory optimization
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// Naive matrix multiplication kernel
__global__ void matrixMulNaive(const float *A, const float *B, float *C, int N)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < N)
  {
    float sum = 0.0f;
    for (int k = 0; k < N; k++)
    {
      sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

// Tiled matrix multiplication with shared memory
__global__ void matrixMulTiled(const float *A, const float *B, float *C, int N)
{
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  float sum = 0.0f;

  for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++)
  {
    // Load tiles into shared memory
    if (row < N && t * TILE_SIZE + threadIdx.x < N)
    {
      As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
    }
    else
    {
      As[threadIdx.y][threadIdx.x] = 0.0f;
    }

    if (col < N && t * TILE_SIZE + threadIdx.y < N)
    {
      Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
    }
    else
    {
      Bs[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Compute partial sum
    for (int k = 0; k < TILE_SIZE; k++)
    {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < N && col < N)
  {
    C[row * N + col] = sum;
  }
}

#define CUDA_CHECK(call)                                            \
  do                                                                \
  {                                                                 \
    cudaError_t err = call;                                         \
    if (err != cudaSuccess)                                         \
    {                                                               \
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

int main()
{
  const int N = 1024;
  const size_t size = N * N * sizeof(float);

  printf("Matrix Multiplication: %d x %d\n", N, N);
  printf("Memory per matrix: %.2f MB\n", size / (1024.0 * 1024.0));

  // Host matrices
  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);
  float *h_C = (float *)malloc(size);

  // Initialize matrices
  for (int i = 0; i < N * N; i++)
  {
    h_A[i] = (float)(rand() % 100) / 100.0f;
    h_B[i] = (float)(rand() % 100) / 100.0f;
  }

  // Device matrices
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, size));
  CUDA_CHECK(cudaMalloc(&d_B, size));
  CUDA_CHECK(cudaMalloc(&d_C, size));

  // Copy to device
  CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

  // Grid and block dimensions
  dim3 blockDim(TILE_SIZE, TILE_SIZE);
  dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

  printf("Grid: (%d, %d), Block: (%d, %d)\n",
         gridDim.x, gridDim.y, blockDim.x, blockDim.y);

  // Timing events
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // Test naive kernel
  CUDA_CHECK(cudaEventRecord(start));
  matrixMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float naiveTime;
  CUDA_CHECK(cudaEventElapsedTime(&naiveTime, start, stop));
  printf("Naive kernel: %.4f ms\n", naiveTime);

  // Test tiled kernel
  CUDA_CHECK(cudaEventRecord(start));
  matrixMulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float tiledTime;
  CUDA_CHECK(cudaEventElapsedTime(&tiledTime, start, stop));
  printf("Tiled kernel: %.4f ms\n", tiledTime);

  // Calculate speedup
  printf("Speedup: %.2fx\n", naiveTime / tiledTime);

  // Calculate GFLOPS
  double flops = 2.0 * N * N * N;
  double gflops = flops / (tiledTime / 1000.0) / 1e9;
  printf("Performance: %.2f GFLOPS\n", gflops);

  // Copy result back
  CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

  printf("Result sample: C[0][0] = %.4f\n", h_C[0]);

  // Cleanup
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  free(h_A);
  free(h_B);
  free(h_C);

  printf("\n✅ Matrix multiplication completed successfully!\n");

  return 0;
}
