# Modal CUDA Kernel Runner - Development Instructions

## Project Overview
This is a VS Code extension that allows running CUDA and Triton kernels on Modal.com's GPU infrastructure.

## Project Structure
- `src/` - TypeScript source files
  - `extension.ts` - Main extension entry point
  - `modalRunner.ts` - Handles Modal CLI execution
  - `resultsPanel.ts` - Webview panel for results display
  - `types.ts` - TypeScript interfaces and types
- `modal_scripts/` - Python scripts for Modal execution
  - `kernel_runner.py` - Main unified runner
  - `run_cuda_kernel.py` - CUDA-specific runner
  - `run_triton_kernel.py` - Triton-specific runner
- `examples/` - Example kernel files

## Development Commands
- `npm run compile` - Compile TypeScript
- `npm run watch` - Watch mode for development
- `npm run lint` - Run ESLint
- Press `F5` to launch Extension Development Host

## Key Features
1. GPU selection via status bar and command palette
2. Automatic kernel type detection (.cu vs .py)
3. Results displayed in webview panel with export options
4. Configurable warmup and benchmark runs
5. Modal setup and authentication checking

## Adding New GPU Types
Edit `AVAILABLE_GPUS` in `src/types.ts` and add corresponding runner functions in `modal_scripts/kernel_runner.py`.

## Testing
1. Open a `.cu` or `.py` file
2. Press `Cmd+Shift+R` (Mac) or `Ctrl+Shift+R` (Win/Linux)
3. Select GPU if prompted
4. View results in the Results Panel
