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
  { id: "A100", name: "NVIDIA A100 (40GB)", memoryGb: 40, architecture: "Ampere" },
  { id: "A100-40GB", name: "NVIDIA A100 (40GB)", memoryGb: 40, architecture: "Ampere" },
  { id: "A100-80GB", name: "NVIDIA A100 (80GB)", memoryGb: 80, architecture: "Ampere" },
  { id: "L40S", name: "NVIDIA L40S", memoryGb: 48, architecture: "Ada Lovelace" },
  { id: "H100", name: "NVIDIA H100", memoryGb: 80, architecture: "Hopper" },
  { id: "H200", name: "NVIDIA H200", memoryGb: 141, architecture: "Hopper" },
  { id: "B200", name: "NVIDIA B200", memoryGb: 192, architecture: "Blackwell" },
];

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
    // Keep only last 50 runs
    if (this.runHistory.length > 50) {
      this.runHistory.pop();
    }
  }
}
