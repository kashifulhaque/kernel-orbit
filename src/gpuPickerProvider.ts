import * as vscode from 'vscode';
import { AVAILABLE_GPUS, GpuConfig, ModalKernelState } from './types';

/** Chronological architecture ordering */
const ARCHITECTURE_ORDER = ['Turing', 'Ampere', 'Ada Lovelace', 'Hopper', 'Blackwell'];

class ArchitectureTreeItem extends vscode.TreeItem {
  constructor(
    public readonly architecture: string,
    gpuCount: number
  ) {
    super(architecture.toUpperCase(), vscode.TreeItemCollapsibleState.Expanded);
    this.description = `${gpuCount}`;
    this.contextValue = 'gpuArchitecture';
  }
}

class GpuTreeItem extends vscode.TreeItem {
  constructor(public readonly gpu: GpuConfig, selected: boolean) {
    super(gpu.name, vscode.TreeItemCollapsibleState.None);

    this.description = `CC: ${gpu.computeCapability} · ${gpu.memoryGb}GB · ${gpu.smCount} SMs`;

    this.tooltip = new vscode.MarkdownString([
      `**${gpu.name}**`,
      '',
      `| Spec | Value |`,
      `|------|-------|`,
      `| Architecture | ${gpu.architecture} |`,
      `| Compute Capability | ${gpu.computeCapability} |`,
      `| Memory | ${gpu.memoryGb} GB |`,
      `| SMs | ${gpu.smCount} |`,
      `| CUDA Cores | ${gpu.cudaCores.toLocaleString()} |`,
      `| Memory Bandwidth | ${gpu.memoryBandwidthGBs.toLocaleString()} GB/s |`,
    ].join('\n'));

    this.iconPath = new vscode.ThemeIcon(selected ? 'check' : 'circle-outline');
    this.contextValue = 'gpuItem';

    this.command = {
      command: 'modalKernel.selectGpuFromTree',
      title: 'Select GPU',
      arguments: [gpu.id],
    };
  }
}

export class GpuPickerProvider implements vscode.TreeDataProvider<vscode.TreeItem> {
  private _onDidChangeTreeData = new vscode.EventEmitter<vscode.TreeItem | undefined | void>();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  private state: ModalKernelState;

  constructor() {
    this.state = ModalKernelState.getInstance();
  }

  refresh(): void {
    this._onDidChangeTreeData.fire();
  }

  getTreeItem(element: vscode.TreeItem): vscode.TreeItem {
    return element;
  }

  getChildren(element?: vscode.TreeItem): vscode.TreeItem[] {
    if (!element) {
      // Root: return architecture groups in chronological order
      const archMap = new Map<string, GpuConfig[]>();
      for (const gpu of AVAILABLE_GPUS) {
        const list = archMap.get(gpu.architecture) || [];
        list.push(gpu);
        archMap.set(gpu.architecture, list);
      }

      return ARCHITECTURE_ORDER
        .filter(arch => archMap.has(arch))
        .map(arch => new ArchitectureTreeItem(arch, archMap.get(arch)!.length));
    }

    if (element instanceof ArchitectureTreeItem) {
      const gpus = AVAILABLE_GPUS.filter(g => g.architecture === element.architecture);
      return gpus.map(gpu => new GpuTreeItem(gpu, gpu.id === this.state.selectedGpu));
    }

    return [];
  }
}
