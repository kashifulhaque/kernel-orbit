import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import { spawn, exec } from 'child_process';
import { promisify } from 'util';
import { KernelResult, ModalKernelState, RunHistoryItem, AVAILABLE_GPUS } from './types';

const execAsync = promisify(exec);

// Helpers
function snakeToCamel(str: string): string {
  return str.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
}

function convertKeysToCamelCase(obj: any): any {
  if (obj === null || obj === undefined) { return obj; }
  if (Array.isArray(obj)) { return obj.map(item => convertKeysToCamelCase(item)); }
  if (typeof obj === 'object') {
    const converted: any = {};
    for (const key in obj) {
      if (Object.prototype.hasOwnProperty.call(obj, key)) {
        converted[snakeToCamel(key)] = convertKeysToCamelCase(obj[key]);
      }
    }
    return converted;
  }
  return obj;
}

// Types
interface ModalCredentials {
  tokenId: string | null;
  tokenSecret: string | null;
  found: boolean;
  envFilePath?: string;
}

export class ModalRunner {
  private outputChannel: vscode.OutputChannel;
  private extensionPath: string;
  private cachedCredentials: ModalCredentials | null = null;

  constructor(extensionPath: string) {
    this.extensionPath = extensionPath;
    this.outputChannel = vscode.window.createOutputChannel('Kernel Orbit');
  }

  getScriptsPath(): string {
    return path.join(this.extensionPath, 'modal_scripts');
  }

  getOutputChannel(): vscode.OutputChannel {
    return this.outputChannel;
  }

  getWorkspaceRoot(): string | null {
    const folders = vscode.workspace.workspaceFolders;
    return folders && folders.length > 0 ? folders[0].uri.fsPath : null;
  }

  clearCache(): void {
    this.cachedCredentials = null;
  }

  private isLikelyTritonSource(source: string): boolean {
    const tritonMarkers = [
      /\bimport\s+triton\b/,
      /\bfrom\s+triton(?:\.[\w.]+)?\s+import\b/,
      /@triton\.jit\b/,
      /\btriton\.jit\b/,
      /\btriton\.autotune\b/,
      /\btriton\.heuristics\b/,
      /\btriton\.language\b/,
    ];
    return tritonMarkers.some((marker) => marker.test(source));
  }

  private terminateChildProcess(childProcess: ReturnType<typeof spawn>): void {
    if (!childProcess.pid) { return; }
    try {
      if (process.platform === 'win32') {
        spawn('taskkill', ['/pid', String(childProcess.pid), '/T', '/F'], { detached: true, stdio: 'ignore' }).unref();
        return;
      }
      childProcess.kill('SIGTERM');
      setTimeout(() => {
        if (!childProcess.killed) {
          try { childProcess.kill('SIGKILL'); } catch { /* ignore */ }
        }
      }, 3000);
    } catch {
      // Process may already be gone.
    }
  }

  getModalEnv(): NodeJS.ProcessEnv {
    const credentials = this.loadModalCredentials();
    const env: NodeJS.ProcessEnv = { ...process.env };

    if (credentials.found && credentials.tokenId && credentials.tokenSecret) {
      env['MODAL_TOKEN_ID'] = credentials.tokenId;
      env['MODAL_TOKEN_SECRET'] = credentials.tokenSecret;
    }

    env['PYTHONUNBUFFERED'] = '1';
    env['PYTHONDONTWRITEBYTECODE'] = '1';
    env['PYTHONIOENCODING'] = 'utf-8';
    env['PYTHONUTF8'] = '1';
    return env;
  }

  private parseEnvFile(filePath: string): Record<string, string> {
    const env: Record<string, string> = {};
    try {
      const content = fs.readFileSync(filePath, 'utf-8');
      for (const line of content.split('\n')) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('#')) { continue; }

        const match = trimmed.match(/^([^=]+)=(.*)$/);
        if (match) {
          const key = match[1].trim();
          let value = match[2].trim();
          if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
            value = value.slice(1, -1);
          }
          env[key] = value;
        }
      }
    } catch { /* file doesn't exist or can't be read */ }
    return env;
  }

  private loadModalCredentials(): ModalCredentials {
    if (this.cachedCredentials) {
      return this.cachedCredentials;
    }

    const dirsToCheck: string[] = [];

    // Directory of the active file + up to 3 parents.
    const activeEditor = vscode.window.activeTextEditor;
    if (activeEditor) {
      let dir = path.dirname(activeEditor.document.fileName);
      dirsToCheck.push(dir);
      for (let i = 0; i < 3 && dir !== path.dirname(dir); i++) {
        dir = path.dirname(dir);
        dirsToCheck.push(dir);
      }
    }

    // Workspace root(s).
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (workspaceFolders) {
      for (const folder of workspaceFolders) {
        if (!dirsToCheck.includes(folder.uri.fsPath)) {
          dirsToCheck.push(folder.uri.fsPath);
        }
      }
    }

    for (const dir of dirsToCheck) {
      const envPath = path.join(dir, '.env');
      if (!fs.existsSync(envPath)) { continue; }

      const envVars = this.parseEnvFile(envPath);
      const tokenId = envVars['MODAL_TOKEN_ID'] || null;
      const tokenSecret = envVars['MODAL_TOKEN_SECRET'] || null;

      if (tokenId && tokenSecret) {
        this.cachedCredentials = { tokenId, tokenSecret, found: true, envFilePath: envPath };
        this.outputChannel.appendLine(`Loaded Modal credentials from: ${envPath}`);
        return this.cachedCredentials;
      }
    }

    // Fallback: shell environment.
    if (process.env['MODAL_TOKEN_ID'] && process.env['MODAL_TOKEN_SECRET']) {
      this.cachedCredentials = {
        tokenId: process.env['MODAL_TOKEN_ID'],
        tokenSecret: process.env['MODAL_TOKEN_SECRET'],
        found: true,
        envFilePath: 'shell environment'
      };
      return this.cachedCredentials;
    }

    return { tokenId: null, tokenSecret: null, found: false };
  }

  async createEnvFile(): Promise<boolean> {
    const root = this.getWorkspaceRoot();
    if (!root) {
      vscode.window.showErrorMessage('No workspace folder open');
      return false;
    }

    const envPath = path.join(root, '.env');
    const template = `# Modal.com API Credentials
# Get your tokens from: https://modal.com/settings
MODAL_TOKEN_ID=your-token-id-here
MODAL_TOKEN_SECRET=your-token-secret-here
`;

    if (fs.existsSync(envPath)) {
      const existing = fs.readFileSync(envPath, 'utf-8');
      if (existing.includes('MODAL_TOKEN_ID') || existing.includes('MODAL_TOKEN_SECRET')) {
        vscode.window.showInformationMessage('.env file already contains Modal credentials');
      } else {
        fs.appendFileSync(envPath, `\n${template}`);
      }
    } else {
      fs.writeFileSync(envPath, template);
    }

    this.cachedCredentials = null;

    const doc = await vscode.workspace.openTextDocument(envPath);
    await vscode.window.showTextDocument(doc);
    vscode.window.showInformationMessage('Add your Modal token ID and secret to the .env file. Get them from modal.com/settings');
    return true;
  }

  async checkModalSetup(): Promise<{
    installed: boolean;
    authenticated: boolean;
    error?: string;
    pythonInfo?: any;
  }> {
    const credentials = this.loadModalCredentials();
    const root = this.getWorkspaceRoot();

    const pythonInfo = {
      venvType: 'uv',
      venvPath: root ? path.join(root, '.venv') : undefined,
      hasEnvCredentials: credentials.found,
      envFilePath: credentials.envFilePath
    };

    try {
      const { stdout } = await execAsync(
        'uv run python -c "import modal; print(modal.__version__)"',
        { env: this.getModalEnv(), cwd: root || undefined }
      );
      this.outputChannel.appendLine(`Modal version: ${stdout.trim()}`);
    } catch {
      return {
        installed: false,
        authenticated: false,
        error: 'Modal is not installed. Run: uv add modal',
        pythonInfo
      };
    }

    // '.env' takes priority
    if (credentials.found && credentials.tokenId && credentials.tokenSecret) {
      const validId = credentials.tokenId.startsWith('ak-');
      const validSecret = credentials.tokenSecret.startsWith('as-');

      if (validId && validSecret) {
        return { installed: true, authenticated: true, pythonInfo };
      }
      return {
        installed: true,
        authenticated: false,
        error: 'Modal credentials have invalid format. Token ID should start with "ak-" and secret with "as-"',
        pythonInfo
      };
    }

    // fallback: modal cli credentials in home dir
    const modalConfig = path.join(os.homedir(), '.modal', 'credentials.json');
    const modalToml = path.join(os.homedir(), '.modal.toml');
    if (fs.existsSync(modalConfig) || fs.existsSync(modalToml)) {
      return { installed: true, authenticated: true, pythonInfo };
    }

    return {
      installed: true,
      authenticated: false,
      error: 'modal is not authenticated. Add MODAL_TOKEN_ID and MODAL_TOKEN_SECRET to .env file, or run "uv run modal token set"',
      pythonInfo
    };
  }

  async setupModal(): Promise<boolean> {
    const root = this.getWorkspaceRoot();
    if (!root) {
      vscode.window.showErrorMessage('No workspace folder open');
      return false;
    }

    const terminal = vscode.window.createTerminal('Modal Setup');
    terminal.show();
    terminal.sendText(`cd "${root}"`);
    terminal.sendText('uv add modal');
    terminal.sendText('');
    terminal.sendText('# After installation, add your Modal credentials to .env file:');
    terminal.sendText('# MODAL_TOKEN_ID=your-token-id');
    terminal.sendText('# MODAL_TOKEN_SECRET=your-token-secret');
    terminal.sendText('# Get your tokens from: https://modal.com/settings');

    this.clearCache();

    const result = await vscode.window.showInformationMessage(
      'modal installation started. Would you like to create a .env file for credentials?',
      'Create .env',
      'Open Modal Dashboard',
      'Skip'
    );

    if (result === 'Create .env') {
      await this.createEnvFile();
    } else if (result === 'Open Modal Dashboard') {
      vscode.env.openExternal(vscode.Uri.parse('https://modal.com/settings'));
    }

    return true;
  }

  detectKernelType(filePath: string): 'cuda' | 'triton' | null {
    const ext = path.extname(filePath).toLowerCase();
    if (ext === '.cu' || ext === '.cuh') { return 'cuda'; }
    if (ext === '.py') {
      try {
        const source = fs.readFileSync(filePath, 'utf-8');
        return this.isLikelyTritonSource(source) ? 'triton' : null;
      } catch {
        return null;
      }
    }
    return null;
  }

  async runKernel(
    filePath: string,
    gpuType: string,
    options: {
      warmupRuns?: number;
      benchmarkRuns?: number;
      enableProfiling?: boolean;
      gpuCount?: number;
    } = {},
    cancellationToken?: vscode.CancellationToken
  ): Promise<KernelResult> {
    const config = vscode.workspace.getConfiguration('modalKernel');
    const warmupRuns = options.warmupRuns ?? config.get<number>('warmupRuns', 3);
    const benchmarkRuns = options.benchmarkRuns ?? config.get<number>('benchmarkRuns', 10);
    const enableProfiling = options.enableProfiling ?? config.get<boolean>('enableProfiling', true);

    const kernelType = this.detectKernelType(filePath);
    if (!kernelType) {
      if (path.extname(filePath).toLowerCase() === '.py') {
        throw new Error('Python file does not look like a Triton kernel. Add Triton markers (e.g. `import triton` or `@triton.jit`) or run a CUDA `.cu` file.');
      }
      throw new Error(`Unsupported file type: ${path.extname(filePath)}`);
    }

    const root = this.getWorkspaceRoot();
    const scriptsPath = this.getScriptsPath();
    const runnerScript = path.join(scriptsPath, 'kernel_runner.py');
    const outputFile = path.join(os.tmpdir(), `modal_result_${Date.now()}.json`);

    this.outputChannel.show(true);
    this.outputChannel.appendLine('');
    this.outputChannel.appendLine('='.repeat(60));
    this.outputChannel.appendLine(`Running ${kernelType.toUpperCase()} kernel on Modal...`);
    this.outputChannel.appendLine('='.repeat(60));
    this.outputChannel.appendLine(`File: ${filePath}`);
    this.outputChannel.appendLine(`GPU: ${gpuType}`);
    this.outputChannel.appendLine('');

    return new Promise((resolve, reject) => {
      const args = [
        'run', 'modal', 'run',
        runnerScript,
        '--kernel-file', filePath,
        '--kernel-type', kernelType,
        '--gpu', gpuType,
        '--warmup', warmupRuns.toString(),
        '--benchmark', benchmarkRuns.toString(),
        '--output-json', outputFile,
      ];

      if (!enableProfiling) {
        args.push('--no-profile');
      }

      this.outputChannel.appendLine(`Executing: uv ${args.join(' ')}`);

      const modalEnv = this.getModalEnv();
      const childProcess = spawn('uv', args, {
        cwd: root || scriptsPath,
        env: modalEnv,
      });

      let stderr = '';
      let settled = false;

      const cleanup = () => {
        cancellationDisposable?.dispose();
      };

      const rejectOnce = (error: Error) => {
        if (settled) { return; }
        settled = true;
        cleanup();
        try { fs.unlinkSync(outputFile); } catch { /* ignore */ }
        reject(error);
      };

      const cancellationDisposable = cancellationToken?.onCancellationRequested(() => {
        this.outputChannel.appendLine('Cancellation requested. Terminating Modal run...');
        this.terminateChildProcess(childProcess);
        rejectOnce(new vscode.CancellationError());
      });

      if (cancellationToken?.isCancellationRequested) {
        this.terminateChildProcess(childProcess);
        rejectOnce(new vscode.CancellationError());
        return;
      }

      childProcess.stdout.on('data', (data: Buffer) => {
        this.outputChannel.append(data.toString());
      });

      childProcess.stderr.on('data', (data: Buffer) => {
        const text = data.toString();
        stderr += text;
        this.outputChannel.append(text);
      });

      childProcess.on('close', (code: number | null) => {
        if (settled) { return; }
        settled = true;
        cleanup();

        if (code === 0) {
          try {
            const resultJson = fs.readFileSync(outputFile, 'utf-8');
            const result = convertKeysToCamelCase(JSON.parse(resultJson)) as KernelResult;
            try { fs.unlinkSync(outputFile); } catch { /* ignore */ }
            resolve(result);
          } catch (error) {
            reject(new Error(`Failed to parse results: ${error}`));
          }
        } else {
          try { fs.unlinkSync(outputFile); } catch { /* ignore */ }
          reject(new Error(`Modal run failed with code ${code}:\n${stderr}`));
        }
      });

      childProcess.on('error', (error: Error) => {
        rejectOnce(new Error(`Failed to start Modal: ${error.message}`));
      });
    });
  }

  async runKernelWithProgress(
    filePath: string,
    gpuType: string,
    options: {
      warmupRuns?: number;
      benchmarkRuns?: number;
      enableProfiling?: boolean;
    } = {}
  ): Promise<KernelResult> {
    return vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: `Running kernel on ${gpuType}...`,
        cancellable: true
      },
      async (progress, token) => {
        progress.report({ increment: 0, message: 'Starting Modal...' });

        const state = ModalKernelState.getInstance();
        const historyItem: RunHistoryItem = {
          id: `run_${Date.now()}`,
          fileName: path.basename(filePath),
          kernelType: this.detectKernelType(filePath) || 'unknown',
          gpuType,
          timestamp: new Date(),
          result: null,
          status: 'running'
        };
        state.currentRun = historyItem;
        state.addToHistory(historyItem);

        try {
          token.onCancellationRequested(() => {
            progress.report({ message: 'Cancelling run...' });
          });
          progress.report({ increment: 20, message: 'Uploading to modal...' });
          const result = await this.runKernel(filePath, gpuType, options, token);
          progress.report({ increment: 80, message: 'Complete!' });

          historyItem.result = result;
          historyItem.status = result.successful ? 'completed' : 'failed';
          state.currentRun = null;
          return result;
        } catch (error) {
          historyItem.status = error instanceof vscode.CancellationError ? 'cancelled' : 'failed';
          state.currentRun = null;
          throw error;
        }
      }
    );
  }

  dispose() {
    this.outputChannel.dispose();
  }
}
