import * as vscode from 'vscode';
import * as path from 'path';
import { ModalRunner } from './modalRunner';
import { ResultsPanel } from './resultsPanel';
import { ModalNotebookController } from './notebookController';
import { SessionTreeProvider } from './sessionTreeProvider';
import { ModalKernelState, AVAILABLE_GPUS, GpuConfig, KernelSessionState } from './types';

let modalRunner: ModalRunner;
let notebookController: ModalNotebookController;
let statusBarItem: vscode.StatusBarItem;

export function activate(context: vscode.ExtensionContext) {
  console.log('Kernel Orbit is now active!');

  modalRunner = new ModalRunner(context.extensionPath);

  notebookController = new ModalNotebookController(modalRunner);
  context.subscriptions.push({ dispose: () => notebookController.dispose() });

  const sessionTree = new SessionTreeProvider(notebookController);
  context.subscriptions.push(
    vscode.window.registerTreeDataProvider('modalKernel.sessions', sessionTree)
  );

  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  statusBarItem.command = 'modalKernel.selectGpu';
  updateStatusBar();
  statusBarItem.show();
  context.subscriptions.push(statusBarItem);

  context.subscriptions.push(
    vscode.commands.registerCommand('modalKernel.runKernel', () => runKernelCommand(context)),
    vscode.commands.registerCommand('modalKernel.selectGpu', () => selectGpuCommand()),
    vscode.commands.registerCommand('modalKernel.showResults', () => showResultsCommand(context)),
    vscode.commands.registerCommand('modalKernel.exportResults', () => exportResultsCommand()),
    vscode.commands.registerCommand('modalKernel.setupModal', () => setupModalCommand()),
    vscode.commands.registerCommand('modalKernel.checkStatus', () => checkStatusCommand()),
    vscode.commands.registerCommand('modalKernel.reloadCredentials', () => reloadCredentialsCommand()),
    vscode.commands.registerCommand('modalKernel.warmupImages', () => warmupImagesCommand()),
    vscode.commands.registerCommand('modalKernel.restartNotebookKernel', () => restartNotebookKernelCommand()),
    vscode.commands.registerCommand('modalKernel.terminateNotebookSession', () => terminateNotebookSessionCommand()),
    vscode.commands.registerCommand('modalKernel.killSession', (item: any) => {
      if (item?.notebookUri) {
        notebookController.terminateSession(item.notebookUri);
      }
    })
  );
  context.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor(() => updateStatusBar()),
    vscode.window.onDidChangeActiveNotebookEditor(() => updateStatusBar()),
    notebookController.onSessionsChanged(() => updateStatusBar())
  );
}

function updateStatusBar() {
  const state = ModalKernelState.getInstance();
  const gpu = AVAILABLE_GPUS.find(g => g.id === state.selectedGpu);

  const editor = vscode.window.activeTextEditor;
  if (editor) {
    const ext = path.extname(editor.document.fileName).toLowerCase();
    if (ext === '.cu' || ext === '.cuh' || ext === '.py') {
      statusBarItem.text = `$(server) ${gpu?.name || state.selectedGpu}`;
      statusBarItem.tooltip = `Click to change GPU. Current: ${state.selectedGpu}`;
      statusBarItem.show();
      return;
    }
  }

  const notebookEditor = vscode.window.activeNotebookEditor;
  if (notebookEditor) {
    const uri = notebookEditor.notebook.uri.toString();
    const sessionState = notebookController.getSessionState(uri);
    const gpuLabel = gpu?.name || state.selectedGpu;

    if (sessionState) {
      switch (sessionState) {
        case 'starting':
          statusBarItem.text = `$(sync~spin) Starting ${gpuLabel}…`;
          statusBarItem.tooltip = `Provisioning GPU container…\nGPU: ${state.selectedGpu}`;
          break;
        case 'busy':
          statusBarItem.text = `$(loading~spin) Running on ${gpuLabel}`;
          statusBarItem.tooltip = `Executing cell on remote GPU\nGPU: ${state.selectedGpu}`;
          break;
        case 'idle':
          statusBarItem.text = `$(check) ${gpuLabel}`;
          statusBarItem.tooltip = `GPU session ready\nClick to change GPU. Current: ${state.selectedGpu}`;
          break;
        case 'disconnected':
          statusBarItem.text = `$(error) ${gpuLabel} (disconnected)`;
          statusBarItem.tooltip = `Session disconnected. Run a cell to reconnect.\nGPU: ${state.selectedGpu}`;
          break;
      }
    } else {
      statusBarItem.text = `$(server) ${gpuLabel}`;
      statusBarItem.tooltip = `Click to change GPU. Current: ${state.selectedGpu}`;
    }
    statusBarItem.show();
    return;
  }

  statusBarItem.hide();
}

async function runKernelCommand(context: vscode.ExtensionContext) {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage('No active editor. Open a .cu or .py file to run.');
    return;
  }

  const filePath = editor.document.fileName;
  const ext = path.extname(filePath).toLowerCase();

  if (ext !== '.cu' && ext !== '.cuh' && ext !== '.py') {
    vscode.window.showWarningMessage('Unsupported file type. Open a .cu or .py file to run.');
    return;
  }

  const setupStatus = await modalRunner.checkModalSetup();
  if (!setupStatus.installed) {
    const action = await vscode.window.showErrorMessage(
      'Modal is not installed.',
      'Setup Modal',
      'Cancel'
    );
    if (action === 'Setup Modal') {
      await modalRunner.setupModal();
    }
    return;
  }

  if (!setupStatus.authenticated) {
    const hasEnvFile = setupStatus.pythonInfo?.hasEnvCredentials;
    const errorMsg = hasEnvFile
      ? 'Modal credentials in .env appear invalid. Check your MODAL_TOKEN_ID and MODAL_TOKEN_SECRET.'
      : 'Modal is not authenticated. Add credentials to .env file or run "modal token set".';

    const action = await vscode.window.showErrorMessage(
      errorMsg,
      'Create .env',
      'Open Terminal',
      'Open Modal Dashboard'
    );

    if (action === 'Create .env') {
      await modalRunner.createEnvFile();
      modalRunner.clearCache();
    } else if (action === 'Open Terminal') {
      const terminal = vscode.window.createTerminal('Modal Auth');
      terminal.show();
      terminal.sendText('# Run this command to authenticate with Modal:');
      terminal.sendText('# uv run modal token set --token-id YOUR_TOKEN_ID --token-secret YOUR_TOKEN_SECRET');
      terminal.sendText('# OR');
      terminal.sendText('# python -m modal token set --token-id YOUR_TOKEN_ID --token-secret YOUR_TOKEN_SECRET');
      terminal.sendText('# Get your tokens from: https://modal.com/settings');
    } else if (action === 'Open Modal Dashboard') {
      vscode.env.openExternal(vscode.Uri.parse('https://modal.com/settings'));
    }
    return;
  }

  const config = vscode.workspace.getConfiguration('modalKernel');
  if (config.get<boolean>('autoSave', true)) {
    await editor.document.save();
  }

  const state = ModalKernelState.getInstance();
  const gpuType = state.selectedGpu;

  const resultsPanel = ResultsPanel.createOrShow(context.extensionUri);
  resultsPanel.showLoading(`Running kernel on ${gpuType}...`);

  try {
    const result = await modalRunner.runKernelWithProgress(filePath, gpuType, {
      warmupRuns: config.get<number>('warmupRuns', 3),
      benchmarkRuns: config.get<number>('benchmarkRuns', 10),
      enableProfiling: config.get<boolean>('enableProfiling', true)
    });

    resultsPanel.updateResults(result, path.basename(filePath));

    if (result.successful) {
      vscode.window.showInformationMessage(
        `Kernel completed in ${result.executionTimeMs.toFixed(4)}ms on ${result.gpuName}`
      );
    } else {
      vscode.window.showErrorMessage(`Kernel failed: ${result.errorMessage.substring(0, 100)}...`);
    }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    resultsPanel.showError(errorMessage);
    vscode.window.showErrorMessage(`Failed to run kernel: ${errorMessage.substring(0, 100)}...`);
  }
}

async function selectGpuCommand() {
  const state = ModalKernelState.getInstance();

  const items: vscode.QuickPickItem[] = AVAILABLE_GPUS.map(gpu => ({
    label: gpu.id === state.selectedGpu ? `$(check) ${gpu.name}` : gpu.name,
    description: `${gpu.memoryGb} GB • ${gpu.architecture}`,
    detail: gpu.id
  }));

  const selected = await vscode.window.showQuickPick(items, {
    placeHolder: 'Select GPU type for kernel execution',
    title: 'GPU Selection'
  });

  if (selected && selected.detail) {
    state.selectedGpu = selected.detail;
    updateStatusBar();
    vscode.window.showInformationMessage(`GPU changed to ${selected.detail}`);
    notebookController.onGpuChanged(selected.detail);
  }
}

async function showResultsCommand(context: vscode.ExtensionContext) {
  ResultsPanel.createOrShow(context.extensionUri);
}

async function exportResultsCommand() {
  const state = ModalKernelState.getInstance();

  if (state.runHistory.length === 0) {
    vscode.window.showWarningMessage('No results to export. Run a kernel first.');
    return;
  }

  const formatOptions = ['JSON', 'CSV', 'Markdown'];
  const format = await vscode.window.showQuickPick(formatOptions, {
    placeHolder: 'Select export format'
  });

  if (!format) {
    return;
  }

  const latestRun = state.runHistory[0];
  if (!latestRun.result) {
    vscode.window.showWarningMessage('No results available to export.');
    return;
  }

  const extension = format.toLowerCase() === 'markdown' ? 'md' : format.toLowerCase();
  const uri = await vscode.window.showSaveDialog({
    defaultUri: vscode.Uri.file(`kernel_results.${extension}`),
    filters: {
      [format]: [extension]
    }
  });

  if (uri) {
    let content: string;
    const result = latestRun.result;

    switch (format) {
      case 'CSV':
        content = generateCSV(result);
        break;
      case 'Markdown':
        content = generateMarkdown(result, latestRun.fileName);
        break;
      default:
        content = JSON.stringify(result, null, 2);
    }

    await vscode.workspace.fs.writeFile(uri, Buffer.from(content, 'utf-8'));
    vscode.window.showInformationMessage(`Results exported to ${uri.fsPath}`);
  }
}

function generateCSV(result: any): string {
  const headers = [
    'GPU', 'Kernel Type', 'Execution Time (ms)', 'Std Dev (ms)',
    'Min (ms)', 'Max (ms)', 'Compilation (ms)', 'Memory Used (MB)',
    'Peak Memory (MB)', 'Temperature (C)', 'Power (W)', 'Successful'
  ];

  const values = [
    result.gpuName || result.gpuTypeRequested,
    result.kernelType,
    result.executionTimeMs?.toFixed(3) || '0',
    result.executionTimeStdMs?.toFixed(3) || '0',
    result.minExecutionTimeMs?.toFixed(3) || '0',
    result.maxExecutionTimeMs?.toFixed(3) || '0',
    result.compilationTimeMs?.toFixed(3) || '0',
    result.gpuMemoryUsedMb?.toFixed(2) || '0',
    result.peakMemoryMb?.toFixed(2) || '0',
    result.gpuTemperatureC?.toString() || '0',
    result.gpuPowerDrawW?.toFixed(1) || '0',
    result.successful?.toString() || 'false'
  ];

  return headers.join(',') + '\n' + values.join(',');
}

function generateMarkdown(result: any, fileName: string): string {
  return `# Kernel Execution Results

**File:** ${fileName}  
**Timestamp:** ${new Date().toISOString()}

## GPU Information
- **GPU:** ${result.gpuName || result.gpuTypeRequested}
- **Compute Capability:** ${result.gpuComputeCapability || 'N/A'}
- **Total Memory:** ${result.gpuMemoryTotalMb?.toFixed(0) || 0} MB

## Timing Results
| Metric | Value |
|--------|-------|
| Compilation Time | ${result.compilationTimeMs?.toFixed(2) || 0} ms |
| Warmup Time (${result.warmupRuns || 0} runs) | ${result.warmupTimeMs?.toFixed(2) || 0} ms |
| **Execution Time** | **${result.executionTimeMs?.toFixed(2) || 0} ± ${result.executionTimeStdMs?.toFixed(2) || 0} ms** |
| Min Execution Time | ${result.minExecutionTimeMs?.toFixed(2) || 0} ms |
| Max Execution Time | ${result.maxExecutionTimeMs?.toFixed(2) || 0} ms |
| Total Time | ${result.totalTimeMs?.toFixed(2) || 0} ms |

## Memory Usage
| Metric | Value |
|--------|-------|
| Memory Used | ${result.gpuMemoryUsedMb?.toFixed(2) || 0} MB |
| Peak Memory | ${result.peakMemoryMb?.toFixed(2) || 0} MB |

## GPU Status
| Metric | Value |
|--------|-------|
| Utilization | ${result.gpuUtilizationPercent || 0}% |
| Temperature | ${result.gpuTemperatureC || 0}°C |
| Power Draw | ${result.gpuPowerDrawW?.toFixed(1) || 0} W |

${result.kernelOutput ? `## Kernel Output\n\`\`\`\n${result.kernelOutput}\n\`\`\`` : ''}

${result.profilerOutput ? `## Profiler Output\n\`\`\`\n${result.profilerOutput}\n\`\`\`` : ''}
`;
}

async function setupModalCommand() {
  await modalRunner.setupModal();
}

async function reloadCredentialsCommand() {
  modalRunner.clearCache();
  const status = await modalRunner.checkModalSetup();

  if (status.pythonInfo?.hasEnvCredentials) {
    vscode.window.showInformationMessage(
      `Modal credentials reloaded from: ${status.pythonInfo.envFilePath}`
    );
  } else {
    vscode.window.showWarningMessage(
      'No Modal credentials found. Add MODAL_TOKEN_ID and MODAL_TOKEN_SECRET to .env file.'
    );
  }
}

async function warmupImagesCommand() {
  const result = await vscode.window.showInformationMessage(
    'Pre-build Modal Docker images? This runs once and makes subsequent kernel runs much faster.',
    'Warmup Images',
    'Cancel'
  );

  if (result !== 'Warmup Images') {
    return;
  }

  const terminal = vscode.window.createTerminal('Modal Image Warmup');
  terminal.show();

  const scriptsPath = modalRunner.getScriptsPath();

  terminal.sendText(`cd "${scriptsPath}"`);
  terminal.sendText('uv run modal run kernel_runner.py --warmup-images');
  terminal.sendText('');
  terminal.sendText('# After this completes, subsequent kernel runs will be much faster!');
}

async function checkStatusCommand() {
  const status = await modalRunner.checkModalSetup();
  const pythonInfo = status.pythonInfo;

  let details = '';
  if (pythonInfo) {
    details = `\n\nPython: ${pythonInfo.path}`;
    if (pythonInfo.venvType !== 'none') {
      details += `\nEnvironment: ${pythonInfo.venvType} (${pythonInfo.venvPath})`;
    }
  }

  if (status.installed && status.authenticated) {
    vscode.window.showInformationMessage(
      `Modal is installed and authenticated. Ready to run kernels!${details}`
    );
  } else if (status.installed) {
    vscode.window.showWarningMessage(
      `Modal is installed but not authenticated. Run "modal token set" to authenticate.${details}`
    );
  } else {
    const installCmd = 'uv add modal';

    const action = await vscode.window.showErrorMessage(
      `Modal is not installed.${details}`,
      'Setup Modal',
      'Copy Install Command'
    );

    if (action === 'Setup Modal') {
      await modalRunner.setupModal();
    } else if (action === 'Copy Install Command') {
      await vscode.env.clipboard.writeText(installCmd);
      vscode.window.showInformationMessage(`Copied: ${installCmd}`);
    }
  }
}

async function restartNotebookKernelCommand() {
  const notebookEditor = vscode.window.activeNotebookEditor;
  if (notebookEditor) {
    await notebookController.resetState(notebookEditor.notebook.uri.toString());
    vscode.window.showInformationMessage(
      'Modal GPU kernel restarted. Python state cleared; container still warm.'
    );
  } else {
    await notebookController.resetState();
    vscode.window.showInformationMessage('Modal GPU kernel state reset for all notebooks.');
  }
}

function terminateNotebookSessionCommand() {
  const notebookEditor = vscode.window.activeNotebookEditor;
  if (notebookEditor) {
    notebookController.terminateSession(notebookEditor.notebook.uri.toString());
    vscode.window.showInformationMessage(
      'Modal GPU session terminated. A new container will start on the next cell run.'
    );
  } else {
    notebookController.terminateSession();
    vscode.window.showInformationMessage('All Modal GPU sessions terminated.');
  }
}

export function deactivate() {
  if (notebookController) {
    notebookController.dispose();
  }
  if (modalRunner) {
    modalRunner.dispose();
  }
}
