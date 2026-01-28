import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import { spawn, exec } from 'child_process';
import { promisify } from 'util';
import { KernelResult, ModalKernelState, RunHistoryItem, AVAILABLE_GPUS } from './types';

const execAsync = promisify(exec);

/**
 * Convert snake_case keys to camelCase
 */
function snakeToCamel(str: string): string {
  return str.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
}

/**
 * Convert an object with snake_case keys to camelCase keys (recursive)
 */
function convertKeysToCamelCase(obj: any): any {
  if (obj === null || obj === undefined) {
    return obj;
  }

  if (Array.isArray(obj)) {
    return obj.map(item => convertKeysToCamelCase(item));
  }

  if (typeof obj === 'object') {
    const converted: any = {};
    for (const key in obj) {
      if (Object.prototype.hasOwnProperty.call(obj, key)) {
        const camelKey = snakeToCamel(key);
        converted[camelKey] = convertKeysToCamelCase(obj[key]);
      }
    }
    return converted;
  }

  return obj;
}

/**
 * Virtual environment detection result
 */
interface VenvInfo {
  found: boolean;
  type: 'uv' | 'venv' | 'none';
  pythonPath: string;
  venvPath?: string;
}

/**
 * Modal credentials from .env file
 */
interface ModalCredentials {
  tokenId: string | null;
  tokenSecret: string | null;
  found: boolean;
  envFilePath?: string;
}

/**
 * Manages execution of kernels on Modal.com
 */
export class ModalRunner {
  private outputChannel: vscode.OutputChannel;
  private extensionPath: string;
  private cachedVenvInfo: VenvInfo | null = null;
  private cachedCredentials: ModalCredentials | null = null;

  constructor(extensionPath: string) {
    this.extensionPath = extensionPath;
    this.outputChannel = vscode.window.createOutputChannel('Modal GPU Runner');
  }

  /**
   * Get the path to the Python scripts bundled with the extension
   */
  getScriptsPath(): string {
    return path.join(this.extensionPath, 'modal_scripts');
  }

  /**
   * Get the Python path (public accessor for warmup command)
   */
  getResolvedPythonPath(): string {
    return this.getPythonPath();
  }

  /**
   * Parse .env file and extract key-value pairs
   */
  private parseEnvFile(filePath: string): Record<string, string> {
    const env: Record<string, string> = {};

    try {
      const content = fs.readFileSync(filePath, 'utf-8');
      const lines = content.split('\n');

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('#')) {
          continue;
        }

        const match = trimmed.match(/^([^=]+)=(.*)$/);
        if (match) {
          const key = match[1].trim();
          let value = match[2].trim();

          if ((value.startsWith('"') && value.endsWith('"')) ||
            (value.startsWith("'") && value.endsWith("'"))) {
            value = value.slice(1, -1);
          }

          env[key] = value;
        }
      }
    } catch (error) {
      // File doesn't exist or can't be read
    }

    return env;
  }

  /**
   * Load Modal credentials from .env file in workspace or active file directory
   */
  private loadModalCredentials(): ModalCredentials {
    if (this.cachedCredentials) {
      return this.cachedCredentials;
    }

    const dirsToCheck: string[] = [];

    const activeEditor = vscode.window.activeTextEditor;
    if (activeEditor) {
      const fileDir = path.dirname(activeEditor.document.fileName);
      dirsToCheck.push(fileDir);

      let parentDir = path.dirname(fileDir);
      for (let i = 0; i < 3 && parentDir !== path.dirname(parentDir); i++) {
        dirsToCheck.push(parentDir);
        parentDir = path.dirname(parentDir);
      }
    }

    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (workspaceFolders) {
      for (const folder of workspaceFolders) {
        if (!dirsToCheck.includes(folder.uri.fsPath)) {
          dirsToCheck.push(folder.uri.fsPath);
        }
      }
    }

    this.outputChannel.appendLine(`Searching for .env in directories: ${dirsToCheck.join(', ')}`);

    for (const dir of dirsToCheck) {
      const envPath = path.join(dir, '.env');

      if (fs.existsSync(envPath)) {
        this.outputChannel.appendLine(`Found .env at: ${envPath}`);
        const envVars = this.parseEnvFile(envPath);

        const tokenId = envVars['MODAL_TOKEN_ID'] || null;
        const tokenSecret = envVars['MODAL_TOKEN_SECRET'] || null;

        if (tokenId && tokenSecret) {
          this.cachedCredentials = {
            tokenId,
            tokenSecret,
            found: true,
            envFilePath: envPath
          };
          this.outputChannel.appendLine(`Loaded Modal credentials from: ${envPath}`);
          return this.cachedCredentials;
        } else {
          this.outputChannel.appendLine(`  .env found but missing MODAL_TOKEN_ID or MODAL_TOKEN_SECRET`);
        }
      }
    }

    if (process.env['MODAL_TOKEN_ID'] && process.env['MODAL_TOKEN_SECRET']) {
      this.outputChannel.appendLine('Using Modal credentials from shell environment');
      this.cachedCredentials = {
        tokenId: process.env['MODAL_TOKEN_ID'],
        tokenSecret: process.env['MODAL_TOKEN_SECRET'],
        found: true,
        envFilePath: 'shell environment'
      };
      return this.cachedCredentials;
    }

    this.outputChannel.appendLine('No Modal credentials found in .env files or environment');
    return { tokenId: null, tokenSecret: null, found: false };
  }

  /**
   * Get environment variables for Modal execution (includes credentials from .env)
   */
  private getModalEnv(): NodeJS.ProcessEnv {
    const credentials = this.loadModalCredentials();
    const venvInfo = this.detectVirtualEnvironment();
    const env: NodeJS.ProcessEnv = { ...process.env };

    if (credentials.found && credentials.tokenId && credentials.tokenSecret) {
      env['MODAL_TOKEN_ID'] = credentials.tokenId;
      env['MODAL_TOKEN_SECRET'] = credentials.tokenSecret;
      this.outputChannel.appendLine(`Setting MODAL_TOKEN_ID=${credentials.tokenId.substring(0, 6)}...`);
    }

    if (venvInfo.found && venvInfo.venvPath) {
      const venvBinDir = process.platform === 'win32'
        ? path.join(venvInfo.venvPath, 'Scripts')
        : path.join(venvInfo.venvPath, 'bin');

      const currentPath = env['PATH'] || '';
      if (!currentPath.includes(venvBinDir)) {
        env['PATH'] = `${venvBinDir}${path.delimiter}${currentPath}`;
        this.outputChannel.appendLine(`Added ${venvBinDir} to PATH`);
      }

      env['VIRTUAL_ENV'] = venvInfo.venvPath;
    }

    env['PYTHONUNBUFFERED'] = '1';
    env['PYTHONDONTWRITEBYTECODE'] = '1';

    return env;
  }

  /**
   * Detect virtual environment in workspace
   * Checks for both uv and standard venv
   */
  private detectVirtualEnvironment(): VenvInfo {
    if (this.cachedVenvInfo) {
      return this.cachedVenvInfo;
    }

    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders || workspaceFolders.length === 0) {
      return { found: false, type: 'none', pythonPath: 'python3' };
    }

    for (const folder of workspaceFolders) {
      const workspacePath = folder.uri.fsPath;

      const venvPaths = [
        path.join(workspacePath, '.venv'),
        path.join(workspacePath, 'venv'),
        path.join(workspacePath, '.virtualenv'),
        path.join(workspacePath, 'env'),
      ];

      for (const venvPath of venvPaths) {
        const pythonPath = process.platform === 'win32'
          ? path.join(venvPath, 'Scripts', 'python.exe')
          : path.join(venvPath, 'bin', 'python');

        if (fs.existsSync(pythonPath)) {
          const isUv = this.isUvManagedVenv(venvPath);

          this.cachedVenvInfo = {
            found: true,
            type: isUv ? 'uv' : 'venv',
            pythonPath: pythonPath,
            venvPath: venvPath
          };

          this.outputChannel.appendLine(`Detected ${isUv ? 'uv' : 'standard'} venv at: ${venvPath}`);
          return this.cachedVenvInfo;
        }
      }
    }

    return { found: false, type: 'none', pythonPath: 'python3' };
  }

  /**
   * Check if a venv is managed by uv
   */
  private isUvManagedVenv(venvPath: string): boolean {
    const pyvenvCfg = path.join(venvPath, 'pyvenv.cfg');

    if (fs.existsSync(pyvenvCfg)) {
      try {
        const content = fs.readFileSync(pyvenvCfg, 'utf-8');
        if (content.includes('uv =') || content.includes('uv=')) {
          return true;
        }
      } catch {
        // Ignore read errors
      }
    }

    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (workspaceFolders) {
      for (const folder of workspaceFolders) {
        const uvLock = path.join(folder.uri.fsPath, 'uv.lock');
        if (fs.existsSync(uvLock)) {
          return true;
        }
      }
    }

    return false;
  }

  /**
   * Get Python executable - checks venv first, then falls back to settings
   */
  private getPythonPath(): string {
    const config = vscode.workspace.getConfiguration('modalKernel');
    const configuredPath = config.get<string>('pythonPath', '');

    if (configuredPath && configuredPath !== 'python3' && configuredPath !== 'python') {
      return configuredPath;
    }

    const venvInfo = this.detectVirtualEnvironment();
    if (venvInfo.found) {
      return venvInfo.pythonPath;
    }

    return configuredPath || 'python3';
  }

  /**
   * Get information about the current Python environment
   */
  async getPythonInfo(): Promise<{ path: string; version: string; venvType: string }> {
    const pythonPath = this.getPythonPath();
    const venvInfo = this.detectVirtualEnvironment();

    let version = 'unknown';
    try {
      const { stdout } = await execAsync(`"${pythonPath}" --version`);
      version = stdout.trim();
    } catch {
      // Ignore
    }

    return {
      path: pythonPath,
      version: version,
      venvType: venvInfo.type
    };
  }

  /**
   * Clear cached venv info (useful when workspace changes)
   */
  clearCache(): void {
    this.cachedVenvInfo = null;
    this.cachedCredentials = null;
  }

  /**
   * Create a .env file with Modal credential placeholders
   */
  async createEnvFile(): Promise<boolean> {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders || workspaceFolders.length === 0) {
      vscode.window.showErrorMessage('No workspace folder open');
      return false;
    }

    const envPath = path.join(workspaceFolders[0].uri.fsPath, '.env');

    if (fs.existsSync(envPath)) {
      const existing = fs.readFileSync(envPath, 'utf-8');

      if (existing.includes('MODAL_TOKEN_ID') || existing.includes('MODAL_TOKEN_SECRET')) {
        vscode.window.showInformationMessage('.env file already contains Modal credentials');
        const doc = await vscode.workspace.openTextDocument(envPath);
        await vscode.window.showTextDocument(doc);
        return true;
      }

      const toAppend = `
# Modal.com API Credentials
# Get your tokens from: https://modal.com/settings
MODAL_TOKEN_ID=your-token-id-here
MODAL_TOKEN_SECRET=your-token-secret-here
`;
      fs.appendFileSync(envPath, toAppend);
      this.outputChannel.appendLine(`Appended Modal credentials template to: ${envPath}`);
    } else {
      const content = `# Modal.com API Credentials
# Get your tokens from: https://modal.com/settings
MODAL_TOKEN_ID=your-token-id-here
MODAL_TOKEN_SECRET=your-token-secret-here
`;
      fs.writeFileSync(envPath, content);
      this.outputChannel.appendLine(`Created .env file at: ${envPath}`);
    }

    this.cachedCredentials = null;

    const doc = await vscode.workspace.openTextDocument(envPath);
    await vscode.window.showTextDocument(doc);

    vscode.window.showInformationMessage(
      'Add your Modal token ID and secret to the .env file. Get them from modal.com/settings'
    );

    return true;
  }

  /**
   * Check if Modal is installed and authenticated
   */
  async checkModalSetup(): Promise<{ installed: boolean; authenticated: boolean; error?: string; pythonInfo?: any }> {
    const pythonPath = this.getPythonPath();
    const venvInfo = this.detectVirtualEnvironment();
    const credentials = this.loadModalCredentials();

    const pythonInfo = {
      path: pythonPath,
      venvType: venvInfo.type,
      venvPath: venvInfo.venvPath,
      hasEnvCredentials: credentials.found,
      envFilePath: credentials.envFilePath
    };

    this.outputChannel.appendLine(`Using Python: ${pythonPath}`);
    this.outputChannel.appendLine(`Venv type: ${venvInfo.type}`);
    this.outputChannel.appendLine(`Credentials from .env: ${credentials.found ? 'Yes' : 'No'}`);
    if (credentials.found) {
      this.outputChannel.appendLine(`  Token ID starts with: ${credentials.tokenId?.substring(0, 6)}...`);
      this.outputChannel.appendLine(`  .env file path: ${credentials.envFilePath}`);
    }

    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (workspaceFolders) {
      this.outputChannel.appendLine(`Workspace folders: ${workspaceFolders.map(f => f.uri.fsPath).join(', ')}`);
    } else {
      this.outputChannel.appendLine(`Workspace folders: none detected`);
    }

    try {
      const modalEnv = this.getModalEnv();
      const { stdout } = await execAsync(`"${pythonPath}" -c "import modal; print(modal.__version__)"`, {
        env: modalEnv
      });
      this.outputChannel.appendLine(`Modal version: ${stdout.trim()}`);
    } catch (error) {
      const installCmd = venvInfo.type === 'uv'
        ? 'uv pip install modal'
        : 'pip install modal';
      return {
        installed: false,
        authenticated: false,
        error: `Modal is not installed. Run: ${installCmd}`,
        pythonInfo
      };
    }

    this.outputChannel.appendLine('Checking Modal authentication...');

    // If we have credentials from .env, validate their format and trust them
    // Actual validation happens when running the kernel
    if (credentials.found && credentials.tokenId && credentials.tokenSecret) {
      const validTokenIdPrefix = credentials.tokenId.startsWith('ak-');
      const validSecretPrefix = credentials.tokenSecret.startsWith('as-');

      if (validTokenIdPrefix && validSecretPrefix) {
        this.outputChannel.appendLine('Modal credentials found in .env with valid format');
        this.outputChannel.appendLine(`  Token ID: ${credentials.tokenId.substring(0, 10)}...`);
        return { installed: true, authenticated: true, pythonInfo };
      } else {
        this.outputChannel.appendLine(`Invalid credential format - Token ID starts with 'ak-': ${validTokenIdPrefix}, Secret starts with 'as-': ${validSecretPrefix}`);
        return {
          installed: true,
          authenticated: false,
          error: 'Modal credentials have invalid format. Token ID should start with "ak-" and secret should start with "as-"',
          pythonInfo
        };
      }
    }

    const modalConfigPath = path.join(os.homedir(), '.modal', 'credentials.json');
    const modalTokenPath = path.join(os.homedir(), '.modal.toml');

    if (fs.existsSync(modalConfigPath) || fs.existsSync(modalTokenPath)) {
      this.outputChannel.appendLine('Modal CLI credentials found in home directory');
      return { installed: true, authenticated: true, pythonInfo };
    }

    this.outputChannel.appendLine('No Modal credentials found');
    return {
      installed: true,
      authenticated: false,
      error: 'Modal is not authenticated. Add MODAL_TOKEN_ID and MODAL_TOKEN_SECRET to .env file, or run "modal token set"',
      pythonInfo
    };
  }

  /**
   * Setup Modal environment - detects uv vs pip
   */
  async setupModal(): Promise<boolean> {
    const venvInfo = this.detectVirtualEnvironment();
    const terminal = vscode.window.createTerminal('Modal Setup');
    terminal.show();

    const workspaceFolders = vscode.workspace.workspaceFolders;
    const workspacePath = workspaceFolders ? workspaceFolders[0].uri.fsPath : '.';

    if (venvInfo.found) {
      const installCmd = venvInfo.type === 'uv' ? 'uv pip install modal' : 'pip install modal';
      terminal.sendText(`# Detected ${venvInfo.type} virtual environment at: ${venvInfo.venvPath}`);
      terminal.sendText(`# Installing Modal...`);
      terminal.sendText(installCmd);
    } else {
      const choice = await vscode.window.showQuickPick(
        [
          { label: 'uv (recommended)', description: 'Fast Python package manager', value: 'uv' },
          { label: 'pip + venv', description: 'Standard Python tooling', value: 'pip' }
        ],
        { placeHolder: 'Select package manager to use' }
      );

      if (!choice) {
        terminal.dispose();
        return false;
      }

      if (choice.value === 'uv') {
        terminal.sendText(`cd "${workspacePath}"`);
        terminal.sendText('# Creating virtual environment with uv...');
        terminal.sendText('uv venv');
        terminal.sendText('source .venv/bin/activate');
        terminal.sendText('uv pip install modal');
      } else {
        terminal.sendText(`cd "${workspacePath}"`);
        terminal.sendText('# Creating virtual environment...');
        terminal.sendText('python3 -m venv .venv');
        terminal.sendText('source .venv/bin/activate');
        terminal.sendText('pip install modal');
      }
    }

    terminal.sendText('');
    terminal.sendText('# After installation, add your Modal credentials to .env file:');
    terminal.sendText('# MODAL_TOKEN_ID=your-token-id');
    terminal.sendText('# MODAL_TOKEN_SECRET=your-token-secret');
    terminal.sendText('');
    terminal.sendText('# Get your tokens from: https://modal.com/settings');
    terminal.sendText('');

    this.clearCache();

    const result = await vscode.window.showInformationMessage(
      'Modal installation started. Would you like to create a .env file for credentials?',
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

  /**
   * Detect kernel type from file
   */
  detectKernelType(filePath: string): 'cuda' | 'triton' | null {
    const ext = path.extname(filePath).toLowerCase();
    if (ext === '.cu' || ext === '.cuh') {
      return 'cuda';
    } else if (ext === '.py') {
      try {
        const content = fs.readFileSync(filePath, 'utf-8');
        if (content.includes('import triton') || content.includes('from triton')) {
          return 'triton';
        }
        return 'triton';
      } catch {
        return 'triton';
      }
    }
    return null;
  }

  /**
   * Run a kernel on Modal
   */
  async runKernel(
    filePath: string,
    gpuType: string,
    options: {
      warmupRuns?: number;
      benchmarkRuns?: number;
      enableProfiling?: boolean;
      gpuCount?: number;
    } = {}
  ): Promise<KernelResult> {
    const config = vscode.workspace.getConfiguration('modalKernel');
    const warmupRuns = options.warmupRuns ?? config.get<number>('warmupRuns', 3);
    const benchmarkRuns = options.benchmarkRuns ?? config.get<number>('benchmarkRuns', 10);
    const enableProfiling = options.enableProfiling ?? config.get<boolean>('enableProfiling', true);
    const gpuCount = options.gpuCount ?? config.get<number>('gpuCount', 1);

    const kernelType = this.detectKernelType(filePath);
    if (!kernelType) {
      throw new Error(`Unsupported file type: ${path.extname(filePath)}`);
    }

    const pythonPath = this.getPythonPath();
    const scriptsPath = this.getScriptsPath();
    const runnerScript = path.join(scriptsPath, 'kernel_runner.py');

    const outputFile = path.join(os.tmpdir(), `modal_result_${Date.now()}.json`);

    this.outputChannel.show(true);
    this.outputChannel.appendLine('');
    this.outputChannel.appendLine('='.repeat(60));
    this.outputChannel.appendLine(`🚀 Running ${kernelType.toUpperCase()} kernel on Modal...`);
    this.outputChannel.appendLine('='.repeat(60));
    this.outputChannel.appendLine(`File: ${filePath}`);
    this.outputChannel.appendLine(`GPU: ${gpuType}`);
    this.outputChannel.appendLine(`Warmup runs: ${warmupRuns}`);
    this.outputChannel.appendLine(`Benchmark runs: ${benchmarkRuns}`);
    this.outputChannel.appendLine('');

    return new Promise((resolve, reject) => {
      const args = [
        '-m', 'modal', 'run',
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

      this.outputChannel.appendLine(`Executing: ${pythonPath} ${args.join(' ')}`);
      this.outputChannel.appendLine('');

      const modalEnv = this.getModalEnv();
      const credentials = this.loadModalCredentials();
      if (credentials.found) {
        this.outputChannel.appendLine(`Using Modal credentials from: ${credentials.envFilePath}`);
      }

      const childProcess = spawn(pythonPath, args, {
        cwd: scriptsPath,
        env: modalEnv
      });

      let stdout = '';
      let stderr = '';

      childProcess.stdout.on('data', (data: Buffer) => {
        const text = data.toString();
        stdout += text;
        this.outputChannel.append(text);
      });

      childProcess.stderr.on('data', (data: Buffer) => {
        const text = data.toString();
        stderr += text;
        this.outputChannel.append(text);
      });

      childProcess.on('close', (code: number | null) => {
        this.outputChannel.appendLine('');

        if (code === 0) {
          try {
            const resultJson = fs.readFileSync(outputFile, 'utf-8');
            const rawResult = JSON.parse(resultJson);
            const result = convertKeysToCamelCase(rawResult) as KernelResult;

            try { fs.unlinkSync(outputFile); } catch { }

            resolve(result);
          } catch (error) {
            reject(new Error(`Failed to parse results: ${error}`));
          }
        } else {
          reject(new Error(`Modal run failed with code ${code}:\n${stderr}`));
        }
      });

      childProcess.on('error', (error: Error) => {
        reject(new Error(`Failed to start Modal: ${error.message}`));
      });
    });
  }

  /**
   * Run kernel with progress indicator
   */
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
        const runId = `run_${Date.now()}`;
        const historyItem: RunHistoryItem = {
          id: runId,
          fileName: path.basename(filePath),
          kernelType: this.detectKernelType(filePath) || 'unknown',
          gpuType: gpuType,
          timestamp: new Date(),
          result: null,
          status: 'running'
        };
        state.currentRun = historyItem;
        state.addToHistory(historyItem);

        try {
          progress.report({ increment: 20, message: 'Uploading to Modal...' });

          const result = await this.runKernel(filePath, gpuType, options);

          progress.report({ increment: 80, message: 'Complete!' });

          historyItem.result = result;
          historyItem.status = result.successful ? 'completed' : 'failed';
          state.currentRun = null;

          return result;
        } catch (error) {
          historyItem.status = 'failed';
          state.currentRun = null;
          throw error;
        }
      }
    );
  }

  /**
   * Get output channel for direct access
   */
  getOutputChannel(): vscode.OutputChannel {
    return this.outputChannel;
  }

  dispose() {
    this.outputChannel.dispose();
  }
}
