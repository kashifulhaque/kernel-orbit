export interface GpuConfig {
  id: string;
  name: string;
  memoryGb: number;
  architecture: string;
}

export const AVAILABLE_GPUS: GpuConfig[] = [
  { id: "T4", name: "NVIDIA T4", memoryGb: 16, architecture: "Turing" },
  { id: "L4", name: "NVIDIA L4", memoryGb: 24, architecture: "Ada Lovelace" },
  { id: "A10G", name: "NVIDIA A10G", memoryGb: 24, architecture: "Ampere" },
  { id: "A100-40GB", name: "NVIDIA A100 (40GB)", memoryGb: 40, architecture: "Ampere" },
  { id: "A100-80GB", name: "NVIDIA A100 (80GB)", memoryGb: 80, architecture: "Ampere" },
  { id: "L40S", name: "NVIDIA L40S", memoryGb: 48, architecture: "Ada Lovelace" },
  { id: "H100", name: "NVIDIA H100", memoryGb: 80, architecture: "Hopper" },
  { id: "H200", name: "NVIDIA H200", memoryGb: 141, architecture: "Hopper" },
  { id: "B200", name: "NVIDIA B200", memoryGb: 192, architecture: "Blackwell" },
];

export type KernelSessionState = 'starting' | 'idle' | 'busy' | 'disconnected';

export interface KernelResult {
  successful: boolean;
  kernelType: string;
  errorMessage: string;

  // Timing (ms)
  compilationTimeMs: number;
  warmupTimeMs: number;
  executionTimeMs: number;
  executionTimeStdMs: number;
  minExecutionTimeMs: number;
  maxExecutionTimeMs: number;
  totalTimeMs: number;

  // Memory (MB)
  gpuMemoryUsedMb: number;
  gpuMemoryReservedMb: number;
  peakMemoryMb: number;

  // GPU Info
  gpuName: string;
  gpuTypeRequested: string;
  gpuComputeCapability: string;
  gpuMemoryTotalMb: number;
  gpuTemperatureC: number;
  gpuPowerDrawW: number;
  gpuUtilizationPercent: number;
  gpuCount: number;

  // Benchmark config
  warmupRuns: number;
  benchmarkRuns: number;

  // Outputs
  kernelOutput: string;
  compilerOutput: string;
  profilerOutput: string;

  // Timing samples
  timingSamplesMs: number[];
}

export interface RunHistoryItem {
  id: string;
  fileName: string;
  kernelType: string;
  gpuType: string;
  timestamp: Date;
  result: KernelResult | null;
  status: 'running' | 'completed' | 'failed';
}

/**
 * Result from executing a single notebook cell on Modal.
 * Keys match the snake_case JSON returned by notebook_runner.py.
 */
export interface NotebookCellResult {
  successful: boolean;
  interrupted?: boolean;
  stdout: string;
  stderr: string;
  error: string | null;
  error_traceback: string | null;
  images: string[];                                    // base64-encoded PNG images (matplotlib)
  html: string[];                                      // HTML outputs (pandas DataFrames, etc.)
  svg: string[];                                       // SVG outputs
  latex: string[];                                     // LaTeX outputs
  markdown: string[];                                  // Markdown outputs
  json_outputs: string[];                              // JSON outputs
  display_outputs: Array<{mime: string; data: string}>; // Mid-cell display() calls
  result_repr: string | null;                          // repr() of last expression
  gpu_name: string;
  gpu_type_requested: string;
  execution_time_ms: number;
}

export class ModalKernelState {
  private static instance: ModalKernelState;

  selectedGpu: string = 'T4';
  runHistory: RunHistoryItem[] = [];
  currentRun: RunHistoryItem | null = null;

  static getInstance(): ModalKernelState {
    if (!ModalKernelState.instance) {
      ModalKernelState.instance = new ModalKernelState();
    }
    return ModalKernelState.instance;
  }

  addToHistory(item: RunHistoryItem) {
    this.runHistory.unshift(item);
    if (this.runHistory.length > 50) {
      this.runHistory.pop();
    }
  }
}
