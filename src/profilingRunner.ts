import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import { spawn } from 'child_process';
import { ProfilingResult, ModalKernelState } from './types';
import { ModalRunner } from './modalRunner';

// Key conversion helpers (same as modalRunner.ts)
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

export class ProfilingRunner {
  private modalRunner: ModalRunner;

  constructor(modalRunner: ModalRunner) {
    this.modalRunner = modalRunner;
  }

  /**
   * Run profiling on a kernel file and return the result.
   */
  async profileKernel(filePath: string, gpuType: string): Promise<ProfilingResult> {
    const kernelType = this.modalRunner.detectKernelType(filePath);
    if (!kernelType) {
      throw new Error(`Unsupported file type: ${path.extname(filePath)}`);
    }

    const root = this.modalRunner.getWorkspaceRoot();
    const scriptsPath = this.modalRunner.getScriptsPath();
    const profilerScript = path.join(scriptsPath, 'kernel_profiler.py');
    const outputFile = path.join(os.tmpdir(), `modal_profile_${Date.now()}.json`);

    const outputChannel = this.modalRunner.getOutputChannel();
    outputChannel.show(true);
    outputChannel.appendLine('');
    outputChannel.appendLine('='.repeat(60));
    outputChannel.appendLine(`Profiling ${kernelType.toUpperCase()} kernel on Modal...`);
    outputChannel.appendLine('='.repeat(60));
    outputChannel.appendLine(`File: ${filePath}`);
    outputChannel.appendLine(`GPU: ${gpuType}`);
    outputChannel.appendLine('');

    return new Promise((resolve, reject) => {
      const args = [
        'run', 'modal', 'run',
        profilerScript,
        '--kernel-file', filePath,
        '--kernel-type', kernelType,
        '--gpu', gpuType,
        '--output-json', outputFile,
      ];

      outputChannel.appendLine(`Executing: uv ${args.join(' ')}`);

      const modalEnv = this.modalRunner.getModalEnv();
      const childProcess = spawn('uv', args, {
        cwd: root || scriptsPath,
        env: modalEnv,
        shell: true
      });

      let stderr = '';

      childProcess.stdout.on('data', (data: Buffer) => {
        outputChannel.append(data.toString());
      });

      childProcess.stderr.on('data', (data: Buffer) => {
        const text = data.toString();
        stderr += text;
        outputChannel.append(text);
      });

      childProcess.on('close', (code: number | null) => {
        if (code === 0) {
          try {
            const resultJson = fs.readFileSync(outputFile, 'utf-8');
            const result = convertKeysToCamelCase(JSON.parse(resultJson)) as ProfilingResult;
            try { fs.unlinkSync(outputFile); } catch { /* ignore */ }

            // Store result in global state
            const state = ModalKernelState.getInstance();
            state.lastProfilingResult = result;
            state.profilingResults.set(filePath, result);

            resolve(result);
          } catch (error) {
            reject(new Error(`Failed to parse profiling results: ${error}`));
          }
        } else {
          reject(new Error(`Modal profiling run failed with code ${code}:\n${stderr}`));
        }
      });

      childProcess.on('error', (error: Error) => {
        reject(new Error(`Failed to start Modal profiler: ${error.message}`));
      });
    });
  }

  /**
   * Wraps profileKernel with a VS Code progress notification.
   */
  async profileKernelWithProgress(filePath: string, gpuType: string): Promise<ProfilingResult> {
    return vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: `Profiling kernel on ${gpuType}...`,
        cancellable: false
      },
      async (progress) => {
        progress.report({ increment: 0, message: 'Starting Modal profiler...' });

        try {
          progress.report({ increment: 20, message: 'Uploading to Modal...' });
          const result = await this.profileKernel(filePath, gpuType);
          progress.report({ increment: 80, message: 'Complete!' });
          return result;
        } catch (error) {
          throw error;
        }
      }
    );
  }
}
