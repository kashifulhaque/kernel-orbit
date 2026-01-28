import * as vscode from 'vscode';
import { KernelResult } from './types';

/**
 * Manages the Results Panel webview
 */
export class ResultsPanel {
  public static currentPanel: ResultsPanel | undefined;

  private readonly _panel: vscode.WebviewPanel;
  private readonly _extensionUri: vscode.Uri;
  private _disposables: vscode.Disposable[] = [];

  private constructor(panel: vscode.WebviewPanel, extensionUri: vscode.Uri) {
    this._panel = panel;
    this._extensionUri = extensionUri;

    this._panel.webview.html = this._getHtmlContent(null);

    this._panel.webview.onDidReceiveMessage(
      message => this._handleMessage(message),
      null,
      this._disposables
    );

    this._panel.onDidDispose(() => this.dispose(), null, this._disposables);
  }

  /**
   * Create or show the results panel
   */
  public static createOrShow(extensionUri: vscode.Uri): ResultsPanel {
    const column = vscode.window.activeTextEditor
      ? vscode.window.activeTextEditor.viewColumn
      : undefined;

    if (ResultsPanel.currentPanel) {
      ResultsPanel.currentPanel._panel.reveal(column);
      return ResultsPanel.currentPanel;
    }

    const panel = vscode.window.createWebviewPanel(
      'modalKernelResults',
      'Results',
      column || vscode.ViewColumn.Two,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [extensionUri]
      }
    );

    ResultsPanel.currentPanel = new ResultsPanel(panel, extensionUri);
    return ResultsPanel.currentPanel;
  }

  /**
   * Update results display
   */
  public updateResults(result: KernelResult, fileName?: string) {
    this._panel.webview.html = this._getHtmlContent(result, fileName);
  }

  /**
   * Show loading state
   */
  public showLoading(message: string = 'Running kernel...') {
    this._panel.webview.html = this._getLoadingHtml(message);
  }

  /**
   * Show error state
   */
  public showError(error: string) {
    this._panel.webview.html = this._getErrorHtml(error);
  }

  private _handleMessage(message: any) {
    switch (message.command) {
      case 'export':
        this._exportResults(message.format, message.data);
        break;
      case 'copyToClipboard':
        vscode.env.clipboard.writeText(message.text);
        vscode.window.showInformationMessage('Copied to clipboard!');
        break;
    }
  }

  private async _exportResults(format: string, data: any) {
    const options: vscode.SaveDialogOptions = {
      defaultUri: vscode.Uri.file(`kernel_results.${format}`),
      filters: {
        'JSON': ['json'],
        'CSV': ['csv'],
        'Markdown': ['md']
      }
    };

    const uri = await vscode.window.showSaveDialog(options);
    if (uri) {
      let content: string;
      switch (format) {
        case 'csv':
          content = this._toCSV(data);
          break;
        case 'md':
          content = this._toMarkdown(data);
          break;
        default:
          content = JSON.stringify(data, null, 2);
      }

      await vscode.workspace.fs.writeFile(uri, Buffer.from(content, 'utf-8'));
      vscode.window.showInformationMessage(`Results exported to ${uri.fsPath}`);
    }
  }

  /**
   * Safely format a number with toFixed, handling undefined/null values
   */
  private _safeFixed(value: number | undefined | null, decimals: number = 2): string {
    if (value === undefined || value === null || isNaN(value)) {
      return '0';
    }
    return value.toFixed(decimals);
  }

  private _toCSV(data: KernelResult): string {
    const headers = [
      'GPU', 'Kernel Type', 'Execution Time (ms)', 'Std Dev (ms)',
      'Min (ms)', 'Max (ms)', 'Compilation (ms)', 'Memory Used (MB)',
      'Peak Memory (MB)', 'Temperature (C)', 'Power (W)', 'Successful'
    ];

    const values = [
      data.gpuName || '',
      data.kernelType || '',
      this._safeFixed(data.executionTimeMs, 3),
      this._safeFixed(data.executionTimeStdMs, 3),
      this._safeFixed(data.minExecutionTimeMs, 3),
      this._safeFixed(data.maxExecutionTimeMs, 3),
      this._safeFixed(data.compilationTimeMs, 3),
      this._safeFixed(data.gpuMemoryUsedMb, 2),
      this._safeFixed(data.peakMemoryMb, 2),
      (data.gpuTemperatureC ?? 0).toString(),
      this._safeFixed(data.gpuPowerDrawW, 1),
      (data.successful ?? false).toString()
    ];

    return headers.join(',') + '\n' + values.join(',');
  }

  private _toMarkdown(data: KernelResult): string {
    return `# Kernel Execution Results

## GPU Information
- **GPU**: ${data.gpuName || 'Unknown'}
- **Compute Capability**: ${data.gpuComputeCapability || 'N/A'}
- **Total Memory**: ${this._safeFixed(data.gpuMemoryTotalMb, 0)} MB

## Timing Results
| Metric | Value |
|--------|-------|
| Compilation Time | ${this._safeFixed(data.compilationTimeMs, 2)} ms |
| Warmup Time (${data.warmupRuns ?? 0} runs) | ${this._safeFixed(data.warmupTimeMs, 2)} ms |
| **Execution Time** | **${this._safeFixed(data.executionTimeMs, 2)} ± ${this._safeFixed(data.executionTimeStdMs, 2)} ms** |
| Min Execution Time | ${this._safeFixed(data.minExecutionTimeMs, 2)} ms |
| Max Execution Time | ${this._safeFixed(data.maxExecutionTimeMs, 2)} ms |
| Total Time | ${this._safeFixed(data.totalTimeMs, 2)} ms |

## Memory Usage
| Metric | Value |
|--------|-------|
| Memory Used | ${this._safeFixed(data.gpuMemoryUsedMb, 2)} MB |
| Peak Memory | ${this._safeFixed(data.peakMemoryMb, 2)} MB |

## GPU Status
| Metric | Value |
|--------|-------|
| Utilization | ${data.gpuUtilizationPercent ?? 0}% |
| Temperature | ${data.gpuTemperatureC ?? 0}°C |
| Power Draw | ${this._safeFixed(data.gpuPowerDrawW, 1)} W |

${data.kernelOutput ? `## Kernel Output\n\`\`\`\n${data.kernelOutput}\n\`\`\`` : ''}

${data.profilerOutput ? `## Profiler Output\n\`\`\`\n${data.profilerOutput}\n\`\`\`` : ''}
`;
  }

  private _getHtmlContent(result: KernelResult | null, fileName?: string): string {
    if (!result) {
      return this._getEmptyHtml();
    }

    const statusIcon = result.successful ? '✅' : '❌';
    const statusClass = result.successful ? 'success' : 'error';

    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Results</title>
    <style>
        :root {
            --bg-color: var(--vscode-editor-background);
            --text-color: var(--vscode-editor-foreground);
            --border-color: var(--vscode-panel-border);
            --card-bg: var(--vscode-editorWidget-background);
            --success-color: #4caf50;
            --error-color: #f44336;
            --accent-color: var(--vscode-button-background);
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: var(--vscode-font-family);
            background-color: var(--bg-color);
            color: var(--text-color);
            padding: 20px;
            line-height: 1.6;
        }
        
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border-color);
        }
        
        .header h1 {
            font-size: 1.5em;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .status-badge {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 500;
        }
        
        .status-badge.success {
            background-color: rgba(76, 175, 80, 0.2);
            color: var(--success-color);
        }
        
        .status-badge.error {
            background-color: rgba(244, 67, 54, 0.2);
            color: var(--error-color);
        }
        
        .export-buttons {
            display: flex;
            gap: 8px;
        }
        
        .export-btn {
            padding: 6px 12px;
            border: 1px solid var(--border-color);
            background: var(--card-bg);
            color: var(--text-color);
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85em;
        }
        
        .export-btn:hover {
            background: var(--accent-color);
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        
        .card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
        }
        
        .card h2 {
            font-size: 1.1em;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .card h2 .icon {
            font-size: 1.2em;
        }
        
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }
        
        .metric {
            padding: 8px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 4px;
        }
        
        .metric-label {
            font-size: 0.75em;
            color: var(--vscode-descriptionForeground);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .metric-value {
            font-size: 1.2em;
            font-weight: 600;
            margin-top: 4px;
        }
        
        .metric-value.highlight {
            color: var(--accent-color);
            font-size: 1.5em;
        }
        
        .metric-unit {
            font-size: 0.8em;
            color: var(--vscode-descriptionForeground);
            margin-left: 2px;
        }
        
        .output-section {
            margin-top: 24px;
        }
        
        .output-section h2 {
            margin-bottom: 12px;
        }
        
        .output-content {
            background: var(--vscode-terminal-background);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            padding: 12px;
            font-family: var(--vscode-editor-font-family);
            font-size: 12px;
            overflow-x: auto;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .timing-chart {
            margin-top: 16px;
            height: 100px;
            display: flex;
            align-items: flex-end;
            gap: 4px;
            padding: 8px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 4px;
        }
        
        .timing-bar {
            flex: 1;
            background: var(--accent-color);
            border-radius: 2px 2px 0 0;
            min-width: 4px;
            transition: height 0.3s ease;
        }
        
        .file-info {
            font-size: 0.9em;
            color: var(--vscode-descriptionForeground);
        }
        
        .error-message {
            background: rgba(244, 67, 54, 0.1);
            border: 1px solid var(--error-color);
            border-radius: 4px;
            padding: 16px;
            margin-bottom: 16px;
        }
        
        .timestamp {
            font-size: 0.8em;
            color: var(--vscode-descriptionForeground);
        }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>
                ${statusIcon} Kernel Results
                <span class="status-badge ${statusClass}">${result.successful ? 'Success' : 'Failed'}</span>
            </h1>
            ${fileName ? `<div class="file-info">${fileName}</div>` : ''}
            <div class="timestamp">${new Date().toLocaleString()}</div>
        </div>
        <div class="export-buttons">
            <button class="export-btn" onclick="exportResults('json')">📄 JSON</button>
            <button class="export-btn" onclick="exportResults('csv')">📊 CSV</button>
            <button class="export-btn" onclick="exportResults('md')">📝 Markdown</button>
        </div>
    </div>
    
    ${!result.successful ? `
    <div class="error-message">
        <strong>Error:</strong> ${this._escapeHtml(result.errorMessage)}
    </div>
    ` : ''}
    
    <div class="grid">
        <div class="card">
            <h2><span class="icon">🖥️</span> GPU Information</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-label">GPU</div>
                    <div class="metric-value">${result.gpuName || result.gpuTypeRequested || 'Unknown'}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Compute Capability</div>
                    <div class="metric-value">${result.gpuComputeCapability || 'N/A'}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Total Memory</div>
                    <div class="metric-value">${this._safeFixed(result.gpuMemoryTotalMb, 0)}<span class="metric-unit">MB</span></div>
                </div>
                <div class="metric">
                    <div class="metric-label">Temperature</div>
                    <div class="metric-value">${result.gpuTemperatureC ?? 0}<span class="metric-unit">°C</span></div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2><span class="icon">⏱️</span> Execution Time</h2>
            <div class="metric-grid">
                <div class="metric" style="grid-column: span 2;">
                    <div class="metric-label">Average Execution Time</div>
                    <div class="metric-value highlight">${this._safeFixed(result.executionTimeMs, 2)}<span class="metric-unit">± ${this._safeFixed(result.executionTimeStdMs, 2)} ms</span></div>
                </div>
                <div class="metric">
                    <div class="metric-label">Min</div>
                    <div class="metric-value">${this._safeFixed(result.minExecutionTimeMs, 2)}<span class="metric-unit">ms</span></div>
                </div>
                <div class="metric">
                    <div class="metric-label">Max</div>
                    <div class="metric-value">${this._safeFixed(result.maxExecutionTimeMs, 2)}<span class="metric-unit">ms</span></div>
                </div>
            </div>
            ${result.timingSamplesMs && result.timingSamplesMs.length > 0 ? `
            <div class="timing-chart">
                ${this._renderTimingBars(result.timingSamplesMs)}
            </div>
            ` : ''}
        </div>
        
        <div class="card">
            <h2><span class="icon">📊</span> Timing Breakdown</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-label">Compilation</div>
                    <div class="metric-value">${this._safeFixed(result.compilationTimeMs, 2)}<span class="metric-unit">ms</span></div>
                </div>
                <div class="metric">
                    <div class="metric-label">Warmup (${result.warmupRuns ?? 0} runs)</div>
                    <div class="metric-value">${this._safeFixed(result.warmupTimeMs, 2)}<span class="metric-unit">ms</span></div>
                </div>
                <div class="metric">
                    <div class="metric-label">Benchmark Runs</div>
                    <div class="metric-value">${result.benchmarkRuns ?? 0}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Total Time</div>
                    <div class="metric-value">${this._safeFixed(result.totalTimeMs, 2)}<span class="metric-unit">ms</span></div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2><span class="icon">💾</span> Memory Usage</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-label">Memory Used</div>
                    <div class="metric-value">${this._safeFixed(result.gpuMemoryUsedMb, 2)}<span class="metric-unit">MB</span></div>
                </div>
                <div class="metric">
                    <div class="metric-label">Peak Memory</div>
                    <div class="metric-value">${this._safeFixed(result.peakMemoryMb, 2)}<span class="metric-unit">MB</span></div>
                </div>
                <div class="metric">
                    <div class="metric-label">Utilization</div>
                    <div class="metric-value">${result.gpuUtilizationPercent ?? 0}<span class="metric-unit">%</span></div>
                </div>
                <div class="metric">
                    <div class="metric-label">Power Draw</div>
                    <div class="metric-value">${this._safeFixed(result.gpuPowerDrawW, 1)}<span class="metric-unit">W</span></div>
                </div>
            </div>
        </div>
    </div>
    
    ${result.kernelOutput ? `
    <div class="output-section">
        <h2>📝 Kernel Output</h2>
        <div class="output-content">${this._escapeHtml(result.kernelOutput)}</div>
    </div>
    ` : ''}
    
    ${result.compilerOutput ? `
    <div class="output-section">
        <h2>🔧 Compiler Output</h2>
        <div class="output-content">${this._escapeHtml(result.compilerOutput)}</div>
    </div>
    ` : ''}
    
    ${result.profilerOutput ? `
    <div class="output-section">
        <h2>🔬 Profiler Output</h2>
        <div class="output-content">${this._escapeHtml(result.profilerOutput)}</div>
    </div>
    ` : ''}
    
    <script>
        const vscode = acquireVsCodeApi();
        const resultData = ${JSON.stringify(result)};
        
        function exportResults(format) {
            vscode.postMessage({
                command: 'export',
                format: format,
                data: resultData
            });
        }
        
        function copyToClipboard(text) {
            vscode.postMessage({
                command: 'copyToClipboard',
                text: text
            });
        }
    </script>
</body>
</html>`;
  }

  private _renderTimingBars(samples: number[]): string {
    if (!samples || samples.length === 0) {
      return '';
    }

    const validSamples = samples.filter(t => t !== undefined && t !== null && !isNaN(t));
    if (validSamples.length === 0) {
      return '';
    }

    const maxTime = Math.max(...validSamples);
    const bars = validSamples.map(time => {
      const height = maxTime > 0 ? (time / maxTime) * 80 : 0;
      const timeStr = this._safeFixed(time, 2);
      return `<div class="timing-bar" style="height: ${height}px;" title="${timeStr} ms"></div>`;
    });

    return bars.join('');
  }

  private _escapeHtml(text: string): string {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  private _getLoadingHtml(message: string): string {
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Results</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            background-color: var(--vscode-editor-background);
            color: var(--vscode-editor-foreground);
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .loading {
            text-align: center;
        }
        .spinner {
            width: 50px;
            height: 50px;
            border: 3px solid var(--vscode-panel-border);
            border-top-color: var(--vscode-button-background);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="loading">
        <div class="spinner"></div>
        <p>${message}</p>
    </div>
</body>
</html>`;
  }

  private _getErrorHtml(error: string): string {
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Results - Error</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            background-color: var(--vscode-editor-background);
            color: var(--vscode-editor-foreground);
            padding: 20px;
        }
        .error-container {
            background: rgba(244, 67, 54, 0.1);
            border: 1px solid #f44336;
            border-radius: 8px;
            padding: 20px;
            max-width: 600px;
            margin: 40px auto;
        }
        h1 {
            color: #f44336;
            margin-bottom: 16px;
        }
        pre {
            background: var(--vscode-terminal-background);
            padding: 12px;
            border-radius: 4px;
            overflow-x: auto;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <div class="error-container">
        <h1>❌ Error</h1>
        <pre>${this._escapeHtml(error)}</pre>
    </div>
</body>
</html>`;
  }

  private _getEmptyHtml(): string {
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Results</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            background-color: var(--vscode-editor-background);
            color: var(--vscode-editor-foreground);
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .empty {
            text-align: center;
            color: var(--vscode-descriptionForeground);
        }
        .icon {
            font-size: 48px;
            margin-bottom: 16px;
        }
    </style>
</head>
<body>
    <div class="empty">
        <div class="icon">🚀</div>
        <h2>No Results Yet</h2>
        <p>Run a kernel to see results here.</p>
        <p style="margin-top: 12px; font-size: 0.9em;">
            Open a .cu or .py file and press <kbd>Cmd+Shift+R</kbd> (Mac) or <kbd>Ctrl+Shift+R</kbd> (Windows/Linux)
        </p>
    </div>
</body>
</html>`;
  }

  public dispose() {
    ResultsPanel.currentPanel = undefined;
    this._panel.dispose();

    while (this._disposables.length) {
      const disposable = this._disposables.pop();
      if (disposable) {
        disposable.dispose();
      }
    }
  }
}
