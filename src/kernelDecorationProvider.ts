import * as vscode from 'vscode';
import { ProfilingResult, KernelProfilingMetrics, ModalKernelState } from './types';

/**
 * Provides inline editor decorations on __global__ / @triton.jit lines
 * with key profiling metrics from the most recent profiling run.
 */
export class KernelDecorationProvider implements vscode.Disposable {
  private _decorationType: vscode.TextEditorDecorationType;
  private _disposables: vscode.Disposable[] = [];

  /** Maps absolute filePath → array of (lineNumber, decorationText) */
  private _decorationCache: Map<string, Array<{ line: number; text: string }>> = new Map();

  constructor() {
    this._decorationType = vscode.window.createTextEditorDecorationType({
      after: {
        color: new vscode.ThemeColor('editorCodeLens.foreground'),
        fontStyle: 'italic',
        margin: '0 0 0 2em',
      },
      isWholeLine: true,
      backgroundColor: 'rgba(100, 100, 255, 0.06)',
      rangeBehavior: vscode.DecorationRangeBehavior.ClosedClosed,
    });

    // Re-apply decorations when switching editors
    this._disposables.push(
      vscode.window.onDidChangeActiveTextEditor(editor => {
        if (editor) {
          this._applyDecorations(editor);
        }
      })
    );

    // Re-apply when document changes (clears stale decorations)
    this._disposables.push(
      vscode.workspace.onDidChangeTextDocument(e => {
        // Invalidate cache for this file — metrics may no longer match
        this._decorationCache.delete(e.document.uri.fsPath);
        const editor = vscode.window.activeTextEditor;
        if (editor && editor.document.uri.fsPath === e.document.uri.fsPath) {
          editor.setDecorations(this._decorationType, []);
        }
      })
    );
  }

  /**
   * Called after profiling completes. Builds decoration data and applies to the
   * visible editor if it matches the profiled file.
   */
  public updateDecorations(filePath: string, result: ProfilingResult): void {
    // Clear old decorations for this file
    this._decorationCache.delete(filePath);

    if (!result.successful || !result.kernelMetrics || result.kernelMetrics.length === 0) {
      return;
    }

    // Find kernel declaration lines in the file
    const editor = this._findEditorForFile(filePath);
    if (!editor) { return; }

    const document = editor.document;
    const text = document.getText();

    const decorations: Array<{ line: number; text: string }> = [];

    for (const metric of result.kernelMetrics) {
      const m = metric as unknown as KernelProfilingMetrics;
      const kernelName = m.kernelName || '';
      if (!kernelName) { continue; }

      const line = this._findKernelLine(text, kernelName, filePath);
      if (line < 0) { continue; }

      const parts: string[] = [];
      if (m.achievedOccupancyPercent !== undefined && m.achievedOccupancyPercent > 0) {
        parts.push(`Occupancy: ${m.achievedOccupancyPercent.toFixed(1)}%`);
      }
      if (m.registersPerThread !== undefined && m.registersPerThread > 0) {
        parts.push(`Registers: ${m.registersPerThread}`);
      }
      if (m.limitingFactor && m.limitingFactor !== 'unknown') {
        parts.push(`Limited by: ${m.limitingFactor}`);
      }
      if (m.executionTimeUs !== undefined && m.executionTimeUs > 0) {
        parts.push(`Time: ${m.executionTimeUs.toFixed(2)}μs`);
      }
      if (m.smEfficiencyPercent !== undefined && m.smEfficiencyPercent > 0) {
        parts.push(`SM Eff: ${m.smEfficiencyPercent.toFixed(1)}%`);
      }

      if (parts.length > 0) {
        decorations.push({ line, text: parts.join(' | ') });
      }
    }

    this._decorationCache.set(filePath, decorations);
    this._applyDecorations(editor);
  }

  /**
   * Clear all decorations for a given file (or all files if no path given).
   */
  public clearDecorations(filePath?: string): void {
    if (filePath) {
      this._decorationCache.delete(filePath);
    } else {
      this._decorationCache.clear();
    }
    // Clear visible editors
    for (const editor of vscode.window.visibleTextEditors) {
      if (!filePath || editor.document.uri.fsPath === filePath) {
        editor.setDecorations(this._decorationType, []);
      }
    }
  }

  // ---- internal ----

  private _applyDecorations(editor: vscode.TextEditor): void {
    const filePath = editor.document.uri.fsPath;
    const cached = this._decorationCache.get(filePath);
    if (!cached || cached.length === 0) {
      editor.setDecorations(this._decorationType, []);
      return;
    }

    const ranges: vscode.DecorationOptions[] = cached.map(d => ({
      range: new vscode.Range(d.line, 0, d.line, editor.document.lineAt(d.line).text.length),
      renderOptions: {
        after: {
          contentText: `  ⚡ ${d.text}`,
        }
      }
    }));

    editor.setDecorations(this._decorationType, ranges);
  }

  /**
   * Locate the source line that declares a kernel with the given name.
   *
   * CUDA: look for `__global__` + function name
   * Triton: look for `@triton.jit` / `@triton.autotune` followed by `def funcname`
   */
  private _findKernelLine(text: string, kernelName: string, filePath: string): number {
    const isCuda = filePath.endsWith('.cu') || filePath.endsWith('.cuh');
    const lines = text.split('\n');

    if (isCuda) {
      // Match `__global__ void kernelName` (possibly with template params, return type qualifiers, etc.)
      const globalRe = new RegExp(`__global__[\\s\\S]*?\\b${this._escapeRegex(kernelName)}\\s*\\(`, 'm');
      for (let i = 0; i < lines.length; i++) {
        // Check multi-line: combine current + next few lines for matching
        const chunk = lines.slice(i, Math.min(i + 5, lines.length)).join(' ');
        if (globalRe.test(chunk)) {
          return i;
        }
      }
      // Fallback: just find the function name
      for (let i = 0; i < lines.length; i++) {
        if (lines[i].includes(kernelName) && lines[i].includes('__global__')) {
          return i;
        }
      }
    } else {
      // Triton: look for @triton.jit / @triton.autotune then `def kernelName`
      const defRe = new RegExp(`^\\s*def\\s+${this._escapeRegex(kernelName)}\\s*\\(`, 'm');
      for (let i = 0; i < lines.length; i++) {
        if (defRe.test(lines[i])) {
          return i;
        }
      }
      // Fallback: partial name match (ncu/torch profiler may mangle names)
      const shortName = kernelName.split('_')[0]; // rough heuristic
      if (shortName.length > 2) {
        for (let i = 0; i < lines.length; i++) {
          if (/^\s*def\s+/.test(lines[i]) && lines[i].includes(shortName)) {
            return i;
          }
        }
      }
    }

    return -1;
  }

  private _findEditorForFile(filePath: string): vscode.TextEditor | undefined {
    return vscode.window.visibleTextEditors.find(e => e.document.uri.fsPath === filePath);
  }

  private _escapeRegex(str: string): string {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  dispose(): void {
    this._decorationType.dispose();
    for (const d of this._disposables) {
      d.dispose();
    }
  }
}
