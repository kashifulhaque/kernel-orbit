import * as path from 'path';
import * as fs from 'fs';
import * as crypto from 'crypto';
import * as vscode from 'vscode';
import { ModalRunner } from './modalRunner';
import { ChildProcess, spawn } from 'child_process';
import { KernelSessionState, ModalKernelState, NotebookCellResult } from './types';

export interface SessionInfo {
  notebookUri: string;
  gpu: string;
  gpuName: string;
  ready: boolean;
  state: KernelSessionState;
}

class ModalNotebookSession {
  private _process: ChildProcess | null = null;
  private _ready = false;
  private _state: KernelSessionState = 'starting';
  private _gpu: string;
  private _gpuName = '';
  private _stdoutBuffer = '';

  private _readyResolve: ((value: void) => void) | null = null;
  private _readyReject: ((err: Error) => void) | null = null;
  private _pendingResolve: ((msg: any) => void) | null = null;
  private _pendingReject: ((err: Error) => void) | null = null;

  private _activeExecution: vscode.NotebookCellExecution | null = null;
  private _streamedOutputCount = 0;

  private _syncResolve: ((msg: any) => void) | null = null;
  private _syncReject: ((err: Error) => void) | null = null;
  private _syncedFileHashes: Map<string, string> = new Map();

  private readonly _modalRunner: ModalRunner;
  private readonly _outputChannel: vscode.OutputChannel;

  constructor(modalRunner: ModalRunner, gpu: string) {
    this._modalRunner = modalRunner;
    this._gpu = gpu;
    this._outputChannel = modalRunner.getOutputChannel();
  }

  isReady(): boolean { return this._ready; }
  getGpu(): string { return this._gpu; }
  getGpuName(): string { return this._gpuName; }
  getState(): KernelSessionState { return this._state; }
  setState(s: KernelSessionState): void { this._state = s; }
  getStreamedOutputCount(): number { return this._streamedOutputCount; }

  setActiveExecution(exec: vscode.NotebookCellExecution | null): void {
    this._activeExecution = exec;
    this._streamedOutputCount = 0;
  }

  interrupt(): void {
    if (!this._process) {
      this._resolvePendingAsInterrupted();
      return;
    }
    try {
      this._process.stdin!.write(JSON.stringify({ action: 'interrupt' }) + '\n');
    } catch {
      this._resolvePendingAsInterrupted();
    }
  }

  private _resolvePendingAsInterrupted(): void {
    if (this._pendingResolve) {
      this._pendingResolve({
        successful: false,
        interrupted: true,
        stdout: '', stderr: '',
        error: 'Execution interrupted', error_traceback: null,
        images: [], html: [], svg: [], latex: [], markdown: [],
        json_outputs: [], display_outputs: [], result_repr: null,
        execution_time_ms: 0,
      });
      this._pendingResolve = null;
      this._pendingReject = null;
    }
  }

  start(timeoutMs = 180_000): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      const root = this._modalRunner.getWorkspaceRoot();
      const scriptsPath = this._modalRunner.getScriptsPath();
      const runnerScript = path.join(scriptsPath, 'notebook_runner.py');

      // Get timeout from VS Code settings (in seconds)
      const config = vscode.workspace.getConfiguration('modalKernel');
      const timeoutSeconds = config.get<number>('timeout', 3600);

      const args = [
        'run', 'modal', 'run',
        runnerScript,
        '--gpu', this._gpu,
        '--interactive',
        '--timeout', timeoutSeconds.toString(),
      ];

      this._outputChannel.appendLine('');
      this._outputChannel.appendLine(`[Session] Starting on ${this._gpu}…`);
      this._outputChannel.appendLine(`[Session] uv ${args.join(' ')}`);

      const env = this._modalRunner.getModalEnv();

      this._process = spawn('uv', args, { cwd: root || scriptsPath, env });

      this._process.stdout!.on('data', (data: Buffer) => this._onStdoutData(data));
      this._process.stderr!.on('data', (data: Buffer) => this._outputChannel.append(data.toString()));

      this._process.on('close', (code: number | null) => {
        this._ready = false;
        this._state = 'disconnected';
        const msg = `Session process exited (code ${code})`;
        this._outputChannel.appendLine(`[Session] ${msg}`);
        if (this._readyReject) {
          this._readyReject(new Error(msg));
          this._readyResolve = null;
          this._readyReject = null;
        }
        if (this._pendingReject) {
          this._pendingReject(new Error(msg));
          this._pendingResolve = null;
          this._pendingReject = null;
        }
      });

      this._process.on('error', (err: Error) => {
        if (this._readyReject) {
          this._readyReject(err);
          this._readyResolve = null;
          this._readyReject = null;
        }
      });

      this._readyResolve = resolve;
      this._readyReject = reject;
      setTimeout(() => {
        if (!this._ready && this._readyReject) {
          this._readyReject(new Error(
            `Session startup timed out after ${timeoutMs / 1000}s. ` +
            'The GPU container may still be provisioning — try again.'
          ));
          this._readyResolve = null;
          this._readyReject = null;
          this.terminate();
        }
      }, timeoutMs);
    });
  }

  executeCell(code: string): Promise<NotebookCellResult> {
    if (!this._process || !this._ready) {
      return Promise.reject(new Error('Session is not ready'));
    }
    return new Promise<NotebookCellResult>((resolve, reject) => {
      this._pendingResolve = (msg) => resolve(msg as NotebookCellResult);
      this._pendingReject = reject;
      this._process!.stdin!.write(JSON.stringify({ action: 'execute', code }) + '\n');
    });
  }

  reset(): Promise<void> {
    if (!this._process || !this._ready) { return Promise.resolve(); }
    return new Promise<void>((resolve, reject) => {
      this._pendingResolve = () => resolve();
      this._pendingReject = reject;
      this._process!.stdin!.write(JSON.stringify({ action: 'reset' }) + '\n');
    });
  }

  async syncFiles(): Promise<void> {
    if (!this._process || !this._ready) { return; }

    const config = vscode.workspace.getConfiguration('modalKernel');
    const syncEnabled = config.get<boolean>('syncFiles', true);
    if (!syncEnabled) { return; }

    const maxSizeMB = config.get<number>('maxSyncFileSizeMB', 100);
    const maxSizeBytes = maxSizeMB * 1024 * 1024;
    const excludePatterns = config.get<string[]>('syncExcludePatterns', []);

    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders || workspaceFolders.length === 0) { return; }

    const workspaceRoot = workspaceFolders[0].uri;

    // Build exclude pattern: combines user excludes + common large/binary dirs
    const defaultExcludes = [
      '**/node_modules/**', '**/.git/**', '**/dist/**', '**/build/**',
      '**/__pycache__/**', '**/.venv/**', '**/venv/**', '**/*.pyc',
      '**/.vscode/**', '**/.idea/**',
    ];
    const allExcludes = [...defaultExcludes, ...excludePatterns].join(',');
    const excludeGlob = `{${allExcludes}}`;

    // Find all files in the workspace (respects .gitignore by default)
    const files = await vscode.workspace.findFiles(
      new vscode.RelativePattern(workspaceRoot, '**/*'),
      excludeGlob,
      5000 // Cap at 5000 files to avoid overwhelming the sync
    );

    const filesToSync: Array<{ path: string; content: string }> = [];
    const skippedLargeFiles: string[] = [];

    for (const fileUri of files) {
      try {
        const stat = await vscode.workspace.fs.stat(fileUri);
        if (stat.type !== vscode.FileType.File) { continue; }

        const relativePath = path.relative(workspaceRoot.fsPath, fileUri.fsPath);

        if (stat.size > maxSizeBytes) {
          skippedLargeFiles.push(`${relativePath} (${(stat.size / (1024 * 1024)).toFixed(1)} MB)`);
          continue;
        }

        // Read file and compute hash for incremental sync
        const contentBytes = await vscode.workspace.fs.readFile(fileUri);
        const hash = crypto.createHash('md5').update(contentBytes).digest('hex');

        // Skip if the file hasn't changed since last sync
        if (this._syncedFileHashes.get(relativePath) === hash) { continue; }

        const contentBase64 = Buffer.from(contentBytes).toString('base64');
        filesToSync.push({ path: relativePath, content: contentBase64 });
        this._syncedFileHashes.set(relativePath, hash);
      } catch {
        // Skip files we can't read (permissions, etc.)
      }
    }

    // Show warning for large files
    if (skippedLargeFiles.length > 0) {
      const fileList = skippedLargeFiles.slice(0, 5).join(', ');
      const extra = skippedLargeFiles.length > 5 ? ` and ${skippedLargeFiles.length - 5} more` : '';
      vscode.window.showWarningMessage(
        `Skipped ${skippedLargeFiles.length} large file(s): ${fileList}${extra}. ` +
        `To use large files on the GPU, either download them directly (e.g. !wget <url> or !curl -O <url>) ` +
        `or use a Modal Volume.`,
        'OK'
      );
    }

    if (filesToSync.length === 0) {
      this._outputChannel.appendLine('[Sync] All files up to date, nothing to sync');
      return;
    }

    this._outputChannel.appendLine(`[Sync] Sending ${filesToSync.length} file(s) to remote container…`);

    // Send the sync_files action and wait for sync_complete response
    return new Promise<void>((resolve, reject) => {
      this._syncResolve = (msg: any) => {
        this._outputChannel.appendLine(`[Sync] Complete — ${msg.files_written ?? 0} file(s) written`);
        if (msg.errors && msg.errors.length > 0) {
          this._outputChannel.appendLine(`[Sync] Errors: ${JSON.stringify(msg.errors)}`);
        }
        resolve();
      };
      this._syncReject = reject;
      this._process!.stdin!.write(JSON.stringify({
        action: 'sync_files',
        files: filesToSync,
        workspace_root: '/workspace',
      }) + '\n');
    });
  }

  terminate(): void {
    if (!this._process) { return; }
    this._ready = false;
    try { this._process.stdin!.write(JSON.stringify({ action: 'terminate' }) + '\n'); } catch { /* closed */ }
    const proc = this._process;
    this._process = null;
    setTimeout(() => {
      try {
        if (process.platform === 'win32') {
          // On Windows, kill the entire process tree to terminate uv and children.
          spawn('taskkill', ['/pid', String(proc.pid), '/T', '/F'], { detached: true, stdio: 'ignore' }).unref();
        } else {
          proc.kill('SIGTERM');
        }
      } catch { /* gone */ }
    }, 3000);
  }

  private _onStdoutData(data: Buffer): void {
    this._stdoutBuffer += data.toString();
    const lines = this._stdoutBuffer.split('\n');
    this._stdoutBuffer = lines.pop()!;
    for (const raw of lines) {
      const line = raw.trim();
      if (!line) { continue; }
      let msg: any;
      try { msg = JSON.parse(line); } catch { this._outputChannel.appendLine(line); continue; }
      this._dispatchMessage(msg);
    }
  }

  private _appendStreamedOutput(output: vscode.NotebookCellOutput): void {
    if (!this._activeExecution) { return; }
    if (this._streamedOutputCount === 0) {
      this._activeExecution.replaceOutput([output]);
    } else {
      this._activeExecution.appendOutput([output]);
    }
    this._streamedOutputCount++;
  }

  private _dispatchMessage(msg: any): void {
    const type: string = msg.type;
    if (type === 'ready') {
      this._ready = true;
      this._state = 'idle';
      this._gpuName = msg.gpu_name ?? '';
      this._outputChannel.appendLine(`[Session] Ready — GPU: ${this._gpuName} (${msg.gpu})`);
      if (this._readyResolve) { this._readyResolve(); this._readyResolve = null; this._readyReject = null; }
      return;
    }
    if (type === 'stream') {
      const item = msg.stream === 'stderr'
        ? vscode.NotebookCellOutputItem.stderr(msg.text)
        : vscode.NotebookCellOutputItem.stdout(msg.text);
      this._appendStreamedOutput(new vscode.NotebookCellOutput([item]));
      return;
    }
    if (type === 'display') {
      const item = msg.mime.startsWith('image/')
        ? new vscode.NotebookCellOutputItem(Buffer.from(msg.data, 'base64'), msg.mime)
        : vscode.NotebookCellOutputItem.text(msg.data, msg.mime);
      this._appendStreamedOutput(new vscode.NotebookCellOutput([item]));
      return;
    }
    if (type === 'interrupted') {
      this._resolvePendingAsInterrupted();
      return;
    }
    if (type === 'result' || type === 'reset' || type === 'terminated') {
      if (this._pendingResolve) { this._pendingResolve(msg); this._pendingResolve = null; this._pendingReject = null; }
      return;
    }
    if (type === 'sync_complete') {
      if (this._syncResolve) { this._syncResolve(msg); this._syncResolve = null; this._syncReject = null; }
      return;
    }
    if (type === 'error') {
      const error = new Error(msg.message ?? 'Unknown session error');
      if (this._pendingReject) { this._pendingReject(error); this._pendingResolve = null; this._pendingReject = null; }
      else if (this._syncReject) { this._syncReject(error); this._syncResolve = null; this._syncReject = null; }
      else if (this._readyReject) { this._readyReject(error); this._readyResolve = null; this._readyReject = null; }
      return;
    }
    this._outputChannel.appendLine(`[Session] Unknown message type: ${type}`);
  }
}

export class ModalNotebookController {
  private readonly _controller: vscode.NotebookController;
  private _executionOrder = 0;

  private _sessions: Map<string, ModalNotebookSession> = new Map();
  private _startPromises: Map<string, Promise<void>> = new Map();
  private _executingNotebooks: Set<string> = new Set();
  private _executionCancellation: Map<string, vscode.CancellationTokenSource> = new Map();

  private readonly _outputChannel: vscode.OutputChannel;
  private readonly _modalRunner: ModalRunner;

  private _onSessionsChanged = new vscode.EventEmitter<void>();
  readonly onSessionsChanged = this._onSessionsChanged.event;

  constructor(modalRunner: ModalRunner) {
    this._modalRunner = modalRunner;
    this._outputChannel = modalRunner.getOutputChannel();

    this._controller = vscode.notebooks.createNotebookController(
      'kernel-orbit-modal-gpu',
      'jupyter-notebook',
      'Modal GPU (Python)'
    );
    this._controller.supportedLanguages = ['python'];
    this._controller.supportsExecutionOrder = true;
    this._controller.description = 'Run on Modal.com with GPU';
    this._controller.detail = 'Execute notebook cells on cloud GPUs via Modal.com. Container stays warm for 15 min.';
    this._controller.executeHandler = this._executeAll.bind(this);
    this._controller.interruptHandler = this._interrupt.bind(this);

    this._controller.onDidChangeSelectedNotebooks((e) => {
      if (e.selected) { this._warmupSession(e.notebook); }
    });
  }

  getActiveSessions(): SessionInfo[] {
    const result: SessionInfo[] = [];
    for (const [uri, session] of this._sessions) {
      result.push({
        notebookUri: uri,
        gpu: session.getGpu(),
        gpuName: session.getGpuName(),
        ready: session.isReady(),
        state: session.getState(),
      });
    }
    return result;
  }

  getSessionState(notebookUri: string): KernelSessionState | null {
    const session = this._sessions.get(notebookUri);
    return session ? session.getState() : null;
  }

  private async _warmupSession(notebook: vscode.NotebookDocument): Promise<void> {
    this._outputChannel.show(true);
    try {
      const setupStatus = await this._modalRunner.checkModalSetup();
      if (!setupStatus.installed || !setupStatus.authenticated) { return; }
    } catch { return; }

    const state = ModalKernelState.getInstance();
    try {
      await this._ensureSession(notebook.uri.toString(), state.selectedGpu);
    } catch { /* error will surface when user runs a cell */ }
  }

  private async _ensureSession(notebookUri: string, gpuType: string): Promise<ModalNotebookSession> {
    const existing = this._sessions.get(notebookUri);

    if (existing?.isReady() && existing.getGpu() === gpuType) {
      return existing;
    }

    if (existing && !existing.isReady() && existing.getGpu() === gpuType) {
      const pending = this._startPromises.get(notebookUri);
      if (pending) {
        await pending;
        const s = this._sessions.get(notebookUri);
        if (s?.isReady()) { return s; }
        throw new Error('Session failed to start');
      }
    }

    if (existing) { existing.terminate(); }

    const session = new ModalNotebookSession(this._modalRunner, gpuType);
    this._sessions.set(notebookUri, session);
    this._onSessionsChanged.fire();

    const startPromise = session.start().finally(() => {
      this._startPromises.delete(notebookUri);
      this._onSessionsChanged.fire();
    });
    this._startPromises.set(notebookUri, startPromise);

    await startPromise;
    return session;
  }

  private async _executeAll(
    cells: vscode.NotebookCell[],
    notebook: vscode.NotebookDocument,
    _controller: vscode.NotebookController
  ): Promise<void> {
    const uri = notebook.uri.toString();
    if (this._executingNotebooks.has(uri)) {
      vscode.window.showWarningMessage('Cells are already executing on Modal. Please wait or interrupt first.');
      return;
    }

    const cts = new vscode.CancellationTokenSource();
    this._executionCancellation.set(uri, cts);
    this._executingNotebooks.add(uri);

    const session = this._sessions.get(uri);
    if (session) { session.setState('busy'); this._onSessionsChanged.fire(); }

    try {
      for (const cell of cells) {
        if (cts.token.isCancellationRequested) { break; }
        await this._executeCell(cell, notebook);
      }
    } finally {
      this._executingNotebooks.delete(uri);
      this._executionCancellation.delete(uri);
      cts.dispose();
      const s = this._sessions.get(uri);
      if (s && s.getState() === 'busy') { s.setState('idle'); this._onSessionsChanged.fire(); }
    }
  }

  private async _executeCell(
    cell: vscode.NotebookCell,
    notebook: vscode.NotebookDocument
  ): Promise<void> {
    const execution = this._controller.createNotebookCellExecution(cell);
    execution.executionOrder = ++this._executionOrder;
    execution.start(Date.now());
    const notebookUri = notebook.uri.toString();

    try {
      const setupStatus = await this._modalRunner.checkModalSetup();
      if (!setupStatus.installed || !setupStatus.authenticated) {
        const msg = !setupStatus.installed
          ? 'Modal is not installed. Run the "Setup Modal Environment" command first.'
          : 'Modal is not authenticated. Add credentials to .env or run "modal token set".';
        execution.replaceOutput([new vscode.NotebookCellOutput([vscode.NotebookCellOutputItem.error(new Error(msg))])]);
        execution.end(false, Date.now());
        return;
      }

      const gpuType = ModalKernelState.getInstance().selectedGpu;

      if (!this._sessions.get(notebookUri)?.isReady() || this._sessions.get(notebookUri)?.getGpu() !== gpuType) {
        execution.replaceOutput([new vscode.NotebookCellOutput([
          vscode.NotebookCellOutputItem.stdout(
            `Starting Modal GPU session on ${gpuType}… (first cell may take a minute while the container provisions)\n`
          )
        ])]);
      }

      const session = await this._ensureSession(notebookUri, gpuType);

      // Sync workspace files to the remote container before execution
      try {
        await session.syncFiles();
      } catch (syncError) {
        this._outputChannel.appendLine(`[Sync] Warning: file sync failed — ${syncError}`);
        // Non-fatal: continue with execution even if sync fails
      }

      // Enable progressive output streaming
      session.setActiveExecution(execution);
      const result = await session.executeCell(cell.document.getText());
      const streamed = session.getStreamedOutputCount() > 0;
      session.setActiveExecution(null);

      if (result.interrupted) {
        if (streamed) {
          execution.appendOutput([new vscode.NotebookCellOutput([
            vscode.NotebookCellOutputItem.stderr('Execution interrupted\n')
          ])]);
        } else {
          execution.replaceOutput([new vscode.NotebookCellOutput([
            vscode.NotebookCellOutputItem.stderr('Execution interrupted\n')
          ])]);
        }
        execution.end(false, Date.now());
      } else if (result.successful) {
        if (streamed) {
          this._appendRemainingOutputs(execution, result);
        } else {
          this._buildSuccessOutputs(execution, result);
        }
        execution.end(true, Date.now());
      } else {
        if (streamed) {
          this._appendErrorOutputs(execution, result);
        } else {
          this._buildErrorOutputs(execution, result);
        }
        execution.end(false, Date.now());
      }
    } catch (error) {
      const session = this._sessions.get(notebookUri);
      if (session) { session.setActiveExecution(null); }
      this._sessions.delete(notebookUri);
      this._onSessionsChanged.fire();
      execution.replaceOutput([new vscode.NotebookCellOutput([
        vscode.NotebookCellOutputItem.error(new Error(error instanceof Error ? error.message : String(error)))
      ])]);
      execution.end(false, Date.now());
    }
  }

  private _buildSuccessOutputs(execution: vscode.NotebookCellExecution, result: NotebookCellResult): void {
    const outputs: vscode.NotebookCellOutput[] = [];
    if (result.stdout) { outputs.push(new vscode.NotebookCellOutput([vscode.NotebookCellOutputItem.stdout(result.stdout)])); }
    // Mid-cell display() calls
    if (result.display_outputs) {
      for (const item of result.display_outputs) {
        const outItem = item.mime.startsWith('image/')
          ? new vscode.NotebookCellOutputItem(Buffer.from(item.data, 'base64'), item.mime)
          : vscode.NotebookCellOutputItem.text(item.data, item.mime);
        outputs.push(new vscode.NotebookCellOutput([outItem]));
      }
    }
    if (result.images) { for (const img of result.images) { outputs.push(new vscode.NotebookCellOutput([new vscode.NotebookCellOutputItem(Buffer.from(img, 'base64'), 'image/png')])); } }
    if (result.html) { for (const html of result.html) { outputs.push(new vscode.NotebookCellOutput([vscode.NotebookCellOutputItem.text(html, 'text/html')])); } }
    if (result.svg) { for (const svg of result.svg) { outputs.push(new vscode.NotebookCellOutput([vscode.NotebookCellOutputItem.text(svg, 'image/svg+xml')])); } }
    if (result.latex) { for (const tex of result.latex) { outputs.push(new vscode.NotebookCellOutput([vscode.NotebookCellOutputItem.text(tex, 'text/latex')])); } }
    if (result.markdown) { for (const md of result.markdown) { outputs.push(new vscode.NotebookCellOutput([vscode.NotebookCellOutputItem.text(md, 'text/markdown')])); } }
    if (result.json_outputs) { for (const j of result.json_outputs) { outputs.push(new vscode.NotebookCellOutput([vscode.NotebookCellOutputItem.text(j, 'application/json')])); } }
    if (result.result_repr) { outputs.push(new vscode.NotebookCellOutput([vscode.NotebookCellOutputItem.text(result.result_repr)])); }
    if (result.stderr) { outputs.push(new vscode.NotebookCellOutput([vscode.NotebookCellOutputItem.stderr(result.stderr)])); }
    execution.replaceOutput(outputs);
  }

  private _buildErrorOutputs(execution: vscode.NotebookCellExecution, result: NotebookCellResult): void {
    const outputs: vscode.NotebookCellOutput[] = [];
    if (result.stdout) { outputs.push(new vscode.NotebookCellOutput([vscode.NotebookCellOutputItem.stdout(result.stdout)])); }
    const err = new Error(result.error || 'Unknown error');
    if (result.error_traceback) { err.stack = result.error_traceback; }
    outputs.push(new vscode.NotebookCellOutput([vscode.NotebookCellOutputItem.error(err)]));
    execution.replaceOutput(outputs);
  }

  /** Append only remaining metadata outputs after progressive streaming */
  private _appendRemainingOutputs(execution: vscode.NotebookCellExecution, result: NotebookCellResult): void {
    if (result.result_repr) {
      execution.appendOutput([new vscode.NotebookCellOutput([
        vscode.NotebookCellOutputItem.text(result.result_repr)
      ])]);
    }
  }

  /** Append error output after progressive streaming */
  private _appendErrorOutputs(execution: vscode.NotebookCellExecution, result: NotebookCellResult): void {
    if (result.stderr) {
      execution.appendOutput([new vscode.NotebookCellOutput([
        vscode.NotebookCellOutputItem.stderr(result.stderr)
      ])]);
    }
    const err = new Error(result.error || 'Unknown error');
    if (result.error_traceback) { err.stack = result.error_traceback; }
    execution.appendOutput([new vscode.NotebookCellOutput([
      vscode.NotebookCellOutputItem.error(err)
    ])]);
  }

  private async _interrupt(notebook: vscode.NotebookDocument): Promise<void> {
    const uri = notebook.uri.toString();
    const session = this._sessions.get(uri);
    if (session) {
      session.interrupt();
    }
    // Cancel remaining queued cells
    const cts = this._executionCancellation.get(uri);
    if (cts) { cts.cancel(); }
    this._onSessionsChanged.fire();
    this._outputChannel.appendLine('[Notebook] Execution interrupted');
  }

  async resetState(notebookUri?: string): Promise<void> {
    if (notebookUri) {
      const session = this._sessions.get(notebookUri);
      if (session?.isReady()) { await session.reset(); }
    } else {
      for (const session of this._sessions.values()) {
        if (session.isReady()) { try { await session.reset(); } catch { /* ignore */ } }
      }
    }
    this._executionOrder = 0;
    this._outputChannel.appendLine('[Notebook] Kernel state reset');
  }

  async onGpuChanged(newGpu: string): Promise<void> {
    const uris = [...this._sessions.keys()];
    if (uris.length === 0) { return; }
    this._outputChannel.show(true);
    for (const uri of uris) {
      try { await this._ensureSession(uri, newGpu); } catch { /* surfaces on next cell run */ }
    }
  }

  terminateSession(notebookUri?: string): void {
    if (notebookUri) {
      const session = this._sessions.get(notebookUri);
      if (session) { session.terminate(); this._sessions.delete(notebookUri); }
    } else {
      for (const session of this._sessions.values()) { session.terminate(); }
      this._sessions.clear();
    }
    this._onSessionsChanged.fire();
    this._outputChannel.appendLine('[Notebook] Session(s) terminated');
  }

  dispose(): void {
    for (const session of this._sessions.values()) { session.terminate(); }
    this._sessions.clear();
    this._onSessionsChanged.fire();
    this._controller.dispose();
  }
}
