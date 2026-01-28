# Kernel Orbit

Kernel Orbit lets you run CUDA and Triton kernels on [modal.com](https://modal.com) GPUs directly from VS Code, complete with benchmarking, profiling, and detailed metrics in one glance.

## Features

- **Run CUDA C++ Kernels** - Compile and run `.cu` files on Modal's GPU infrastructure
- **Run Triton Kernels** - Execute OpenAI Triton Python kernels with full profiling
- **GPU Selection** - Choose from T4, L4, A10G, A100, L40S, H100, H200, B200
- **Comprehensive Metrics** - Execution time, memory usage, temperature, power draw
- **Benchmarking** - Configurable warmup and benchmark runs with statistical analysis
- **Profiling** - CUDA profiler output and nvidia-smi metrics
- **Export Results** - Save results as JSON, CSV, or Markdown
- **Keyboard Shortcuts** - `Cmd+Shift+R` (Mac) / `Ctrl+Shift+R` (Win/Linux)

## Requirements

- **Python 3.8+** with `uv` (recommended) or `pip`
- **Modal account** - Sign up at [modal.com](https://modal.com)
- **Modal CLI** - Installed and authenticated

## Installation

### 1. Install the Extension

Install from the VS Code Marketplace or clone this repository.

### 2. Install Modal

```bash
# Using uv (recommended)
uv add modal

# Or using pip
pip install modal
```

### 3. Authenticate with Modal

```bash
modal token set
```

This will open a browser window to authenticate with your Modal account.

## Usage

### Running a Kernel

1. Open a `.cu` (CUDA) or `.py` (Triton) file
2. Press `Cmd+Shift+R` (Mac) or `Ctrl+Shift+R` (Windows/Linux)
3. Or right-click and select **"Run Kernel on Modal"**
4. View results in the Results Panel

### Selecting a GPU

1. Click the GPU indicator in the status bar (bottom right)
2. Or run command: **"Kernel Orbit: Select GPU Type"**
3. Choose from available GPUs:
   - **T4** - 16 GB, Turing (budget-friendly)
   - **L4** - 24 GB, Ada Lovelace
   - **A10G** - 24 GB, Ampere
   - **A100** - 40/80 GB, Ampere (ML training)
   - **L40S** - 48 GB, Ada Lovelace
   - **H100** - 80 GB, Hopper (top performance)
   - **H200** - 141 GB, Hopper
   - **B200** - 192 GB, Blackwell (latest)

### Results Panel

After running a kernel, the Results Panel shows:

- **GPU Information** - Name, compute capability, memory
- **Timing** - Execution time with min/max/std deviation
- **Memory Usage** - GPU memory allocated and peak usage
- **GPU Status** - Temperature, power draw, utilization
- **Kernel Output** - stdout/stderr from your kernel
- **Profiler Output** - nvidia-smi metrics

### Exporting Results

Click the export buttons in the Results Panel:
- **JSON** - Full structured data
- **CSV** - Spreadsheet-compatible
- **Markdown** - Documentation-friendly

## Example Kernels

### CUDA Vector Add

```cuda
// vector_add.cu
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 1000000;
    size_t size = n * sizeof(float);
    
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    
    // Allocate host memory
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);
    
    // Initialize vectors
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // Copy to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    
    // Copy result back
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // Verify
    printf("Result: c[0] = %f, c[999999] = %f\n", h_c[0], h_c[999999]);
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}
```

### Triton Vector Add

```python
# vector_add.py
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def benchmark():
    n = 1000000
    x = torch.rand(n, device='cuda')
    y = torch.rand(n, device='cuda')
    output = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
    
    # Verify
    expected = x + y
    assert torch.allclose(output, expected)
    print(f"Result verified! First 5 elements: {output[:5]}")
    
    return output

# Entry point
benchmark()
```

## Extension Settings

Configure in VS Code Settings (`Cmd+,`):

| Setting | Default | Description |
|---------|---------|-------------|
| `modalKernel.defaultGpu` | `T4` | Default GPU type |
| `modalKernel.gpuCount` | `1` | Number of GPUs per container |
| `modalKernel.timeout` | `300` | Execution timeout (seconds) |
| `modalKernel.warmupRuns` | `3` | Warmup iterations before benchmarking |
| `modalKernel.benchmarkRuns` | `10` | Number of benchmark iterations |
| `modalKernel.enableProfiling` | `true` | Enable CUDA profiling |
| `modalKernel.pythonPath` | `python3` | Path to Python interpreter |
| `modalKernel.autoSave` | `true` | Save file before running |

## Commands

| Command | Description | Shortcut |
|---------|-------------|----------|
| `Kernel Orbit: Run Kernel on Modal` | Run current file on Modal | `Cmd+Shift+R` / `Ctrl+Shift+R` |
| `Kernel Orbit: Select GPU Type` | Choose GPU configuration | - |
| `Kernel Orbit: Show Results Panel` | Open results webview | - |
| `Kernel Orbit: Export Results` | Export results to file | - |
| `Kernel Orbit: Setup Modal Environment` | Install and configure Modal | - |
| `Kernel Orbit: Check Modal Status` | Verify Modal installation | - |

## Troubleshooting

### Modal Not Installed

```bash
uv add modal
```

### Modal Not Authenticated

```bash
modal token set
```

### Python Not Found

Set the correct Python path in settings:

```json
{
  "modalKernel.pythonPath": "/path/to/python3"
}
```

### Kernel Compilation Fails

- Check your CUDA syntax
- Ensure all includes are available
- Review compiler output in the Results Panel

## Development

```bash
# Clone the repository
git clone <repo-url>
cd modal-cuda-kernel-runner

# Install dependencies
npm install

# Compile
npm run compile

# Run in development
# Press F5 in VS Code
```

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.

## Credits

- [Modal.com](https://modal.com) - Serverless GPU infrastructure
- [OpenAI Triton](https://github.com/triton-lang/triton) - GPU programming language
