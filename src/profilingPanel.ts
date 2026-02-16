import * as vscode from 'vscode';
import { ProfilingResult, KernelProfilingMetrics } from './types';

/**
 * Manages the Profiling Panel webview — a dedicated panel for ncu / torch.profiler results.
 */
export class ProfilingPanel {
  public static currentPanel: ProfilingPanel | undefined;

  private readonly _panel: vscode.WebviewPanel;
  private readonly _extensionUri: vscode.Uri;
  private _disposables: vscode.Disposable[] = [];
  private _lastResult: ProfilingResult | null = null;

  private constructor(panel: vscode.WebviewPanel, extensionUri: vscode.Uri) {
    this._panel = panel;
    this._extensionUri = extensionUri;

    this._panel.webview.html = this._getEmptyHtml();

    this._panel.webview.onDidReceiveMessage(
      message => this._handleMessage(message),
      null,
      this._disposables
    );

    this._panel.onDidDispose(() => this.dispose(), null, this._disposables);
  }

  public static createOrShow(extensionUri: vscode.Uri): ProfilingPanel {
    const column = vscode.window.activeTextEditor
      ? vscode.window.activeTextEditor.viewColumn
      : undefined;

    if (ProfilingPanel.currentPanel) {
      ProfilingPanel.currentPanel._panel.reveal(column);
      return ProfilingPanel.currentPanel;
    }

    const panel = vscode.window.createWebviewPanel(
      'modalKernelProfiling',
      'Profiling',
      column || vscode.ViewColumn.Two,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [extensionUri]
      }
    );

    ProfilingPanel.currentPanel = new ProfilingPanel(panel, extensionUri);
    return ProfilingPanel.currentPanel;
  }

  public updateResults(result: ProfilingResult, fileName?: string) {
    this._lastResult = result;
    this._panel.webview.html = this._getHtmlContent(result, fileName);
  }

  public showLoading(message: string = 'Profiling kernel...') {
    this._panel.webview.html = this._getLoadingHtml(message);
  }

  public showError(error: string) {
    this._panel.webview.html = this._getErrorHtml(error);
  }

  // ---- message handling ----

  private _handleMessage(message: any) {
    switch (message.command) {
      case 'export':
        this._exportResults(message.format);
        break;
      case 'copyToClipboard':
        vscode.env.clipboard.writeText(message.text);
        vscode.window.showInformationMessage('Copied to clipboard!');
        break;
    }
  }

  private async _exportResults(format: string) {
    if (!this._lastResult) { return; }

    const uri = await vscode.window.showSaveDialog({
      defaultUri: vscode.Uri.file(`profiling_results.${format}`),
      filters: {
        'JSON': ['json'],
        'CSV': ['csv']
      }
    });

    if (uri) {
      let content: string;
      if (format === 'csv') {
        content = this._toCSV(this._lastResult);
      } else {
        content = JSON.stringify(this._lastResult, null, 2);
      }

      await vscode.workspace.fs.writeFile(uri, Buffer.from(content, 'utf-8'));
      vscode.window.showInformationMessage(`Profiling results exported to ${uri.fsPath}`);
    }
  }

  private _toCSV(result: ProfilingResult): string {
    const headers = [
      'Kernel Name', 'Execution Time (μs)', 'SM Efficiency (%)',
      'Memory Throughput (GB/s)', 'Occupancy (%)', 'Registers/Thread',
      'Shared Memory (bytes)', 'Limiting Factor'
    ];

    const rows = (result.kernelMetrics || []).map((m: KernelProfilingMetrics) => [
      m.kernelName || '',
      (m.executionTimeUs ?? 0).toFixed(2),
      (m.smEfficiencyPercent ?? 0) < 0 ? 'N/A' : (m.smEfficiencyPercent ?? 0).toFixed(1),
      (m.memoryThroughputGbs ?? 0) < 0 ? 'N/A' : (m.memoryThroughputGbs ?? 0).toFixed(1),
      (m.achievedOccupancyPercent ?? 0).toFixed(1),
      (m.registersPerThread ?? 0).toString(),
      (m.sharedMemoryBytes ?? 0).toString(),
      m.limitingFactor || 'unknown'
    ].join(','));

    return headers.join(',') + '\n' + rows.join('\n');
  }

  // ---- helpers ----

  private _safeFixed(value: number | undefined | null, decimals: number = 2): string {
    if (value === undefined || value === null || isNaN(value)) { return '0'; }
    return value.toFixed(decimals);
  }

  private _escapeHtml(text: string): string {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  private _colorClass(value: number, lowThresh: number = 30, highThresh: number = 60): string {
    if (value < 0) { return 'metric-na'; }
    if (value < lowThresh) { return 'metric-red'; }
    if (value < highThresh) { return 'metric-yellow'; }
    return 'metric-green';
  }

  /** Format a block/grid dim tuple, skipping trailing 1s for readability. */
  private _formatDims(dims: number[]): string {
    if (!dims || dims.length === 0) { return '—'; }
    // All zeros means unknown
    if (dims.every(d => d === 0)) { return '—'; }
    // Trim trailing 1s for display: (64,64,1) → "64×64"
    let significant = dims.length;
    while (significant > 1 && dims[significant - 1] === 1) { significant--; }
    return dims.slice(0, significant).join('×');
  }

  /** Generate the Performance Timeline HTML section with 4 chart canvases. */
  private _getTimelineHtml(result: ProfilingResult): string {
    const samples = (result as any).timelineSamples || [];
    if (!samples || samples.length < 2) { return ''; }

    const peakBw = (result as any).gpuPeakMemoryBwGbs ?? 0;
    const powerLimit = (result as any).gpuPowerLimitW ?? 0;

    // Find the total duration to show in the header
    const lastTs = samples[samples.length - 1].timestampMs ?? 0;
    const durationStr = lastTs > 1000 ? `${(lastTs / 1000).toFixed(1)}s` : `${lastTs.toFixed(0)}ms`;

    return `
    <!-- Performance Timeline -->
    <div class="timeline-section">
        <h2>Performance Timeline <span class="sample-count">${samples.length} samples · ${durationStr}</span></h2>
        <div class="timeline-grid">
            <div class="chart-card">
                <div class="chart-title">GPU Utilization</div>
                <div class="chart-canvas-wrap"><canvas id="chartGpuUtil" class="timeline-chart"></canvas></div>
                <div class="chart-legend"><span><span class="swatch" style="background:#4fc3f7"></span>GPU Util %</span></div>
            </div>
            <div class="chart-card">
                <div class="chart-title">Memory Bandwidth</div>
                <div class="chart-canvas-wrap"><canvas id="chartMemBw" class="timeline-chart"></canvas></div>
                <div class="chart-legend"><span><span class="swatch" style="background:#81c784"></span>Mem Util %</span>${peakBw > 0 ? `<span style="opacity:.6">Peak: ${peakBw} GB/s</span>` : ''}</div>
            </div>
            <div class="chart-card">
                <div class="chart-title">Power Draw</div>
                <div class="chart-canvas-wrap"><canvas id="chartPower" class="timeline-chart"></canvas></div>
                <div class="chart-legend"><span><span class="swatch" style="background:#ffb74d"></span>Power (W)</span>${powerLimit > 0 ? `<span style="opacity:.6">TDP: ${powerLimit}W</span>` : ''}</div>
            </div>
            <div class="chart-card">
                <div class="chart-title">Temperature</div>
                <div class="chart-canvas-wrap"><canvas id="chartTemp" class="timeline-chart"></canvas></div>
                <div class="chart-legend"><span><span class="swatch" style="background:#e57373"></span>Temp (°C)</span></div>
            </div>
        </div>
    </div>

    <script>
    // ---- Lightweight canvas chart renderer ----
    (function() {
        const samples = ${JSON.stringify(samples)};
        if (!samples || samples.length < 2) return;

        const timestamps = samples.map(s => s.timestamp_ms ?? s.timestampMs ?? 0);
        const maxT = Math.max(...timestamps);

        function drawChart(canvasId, values, color, opts) {
            const canvas = document.getElementById(canvasId);
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            const dpr = window.devicePixelRatio || 1;
            const rect = canvas.parentElement.getBoundingClientRect();
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.scale(dpr, dpr);
            const W = rect.width, H = rect.height;

            const yMin = opts.yMin ?? 0;
            const yMax = opts.yMax ?? Math.max(...values, 1);
            const pad = { top: 8, right: 8, bottom: 20, left: 40 };
            const cW = W - pad.left - pad.right;
            const cH = H - pad.top - pad.bottom;

            // Grid lines
            ctx.strokeStyle = 'rgba(255,255,255,0.06)';
            ctx.lineWidth = 1;
            const gridSteps = 4;
            for (let i = 0; i <= gridSteps; i++) {
                const y = pad.top + (cH / gridSteps) * i;
                ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
            }

            // Y-axis labels
            ctx.fillStyle = 'rgba(255,255,255,0.4)';
            ctx.font = '10px var(--vscode-font-family, sans-serif)';
            ctx.textAlign = 'right';
            for (let i = 0; i <= gridSteps; i++) {
                const y = pad.top + (cH / gridSteps) * i;
                const val = yMax - (yMax - yMin) * (i / gridSteps);
                ctx.fillText(val.toFixed(opts.decimals ?? 0), pad.left - 4, y + 3);
            }

            // X-axis labels (time)
            ctx.textAlign = 'center';
            const xSteps = Math.min(5, timestamps.length);
            for (let i = 0; i <= xSteps; i++) {
                const t = (maxT / xSteps) * i;
                const x = pad.left + (t / maxT) * cW;
                const label = t > 1000 ? (t/1000).toFixed(1) + 's' : t.toFixed(0) + 'ms';
                ctx.fillText(label, x, H - 4);
            }

            // Reference line (e.g. TDP, peak BW)
            if (opts.refLine && opts.refLine > yMin && opts.refLine <= yMax) {
                const refY = pad.top + cH - ((opts.refLine - yMin) / (yMax - yMin)) * cH;
                ctx.strokeStyle = 'rgba(255,255,255,0.2)';
                ctx.setLineDash([4, 4]);
                ctx.beginPath(); ctx.moveTo(pad.left, refY); ctx.lineTo(W - pad.right, refY); ctx.stroke();
                ctx.setLineDash([]);
            }

            // Area fill
            ctx.beginPath();
            ctx.moveTo(pad.left, pad.top + cH);
            for (let i = 0; i < values.length; i++) {
                const x = pad.left + (timestamps[i] / maxT) * cW;
                const y = pad.top + cH - ((Math.max(values[i], yMin) - yMin) / (yMax - yMin)) * cH;
                ctx.lineTo(x, y);
            }
            ctx.lineTo(pad.left + (timestamps[timestamps.length-1] / maxT) * cW, pad.top + cH);
            ctx.closePath();
            ctx.fillStyle = color.replace(')', ',0.15)').replace('rgb', 'rgba');
            ctx.fill();

            // Line
            ctx.beginPath();
            for (let i = 0; i < values.length; i++) {
                const x = pad.left + (timestamps[i] / maxT) * cW;
                const y = pad.top + cH - ((Math.max(values[i], yMin) - yMin) / (yMax - yMin)) * cH;
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }

        function renderAll() {
            const gpuUtil = samples.map(s => s.gpu_utilization_percent ?? s.gpuUtilizationPercent ?? 0);
            const memUtil = samples.map(s => s.memory_utilization_percent ?? s.memoryUtilizationPercent ?? 0);
            const power = samples.map(s => s.power_draw_w ?? s.powerDrawW ?? 0);
            const temp = samples.map(s => s.temperature_c ?? s.temperatureC ?? 0);

            const maxPower = Math.max(...power, ${powerLimit > 0 ? powerLimit : 1});
            const maxTemp = Math.max(...temp, 50);

            drawChart('chartGpuUtil', gpuUtil, 'rgb(79,195,247)', { yMin: 0, yMax: 100, decimals: 0 });
            drawChart('chartMemBw', memUtil, 'rgb(129,199,132)', { yMin: 0, yMax: 100, decimals: 0 });
            drawChart('chartPower', power, 'rgb(255,183,77)', { yMin: 0, yMax: Math.ceil(maxPower * 1.1), decimals: 0, refLine: ${powerLimit} });
            drawChart('chartTemp', temp, 'rgb(229,115,115)', { yMin: Math.max(0, Math.min(...temp) - 5), yMax: Math.ceil(maxTemp * 1.1), decimals: 0 });
        }

        // Render on load and on resize
        renderAll();
        window.addEventListener('resize', renderAll);
    })();
    </script>`;
  }

  // ---- HTML generators ----

  private _getHtmlContent(result: ProfilingResult, fileName?: string): string {
    if (!result) { return this._getEmptyHtml(); }

    const statusIcon = result.successful ? '✅' : '❌';
    const statusClass = result.successful ? 'success' : 'error';

    // Headline metrics from first kernel (or aggregated)
    const metrics = result.kernelMetrics || [];
    const first = metrics.length > 0 ? metrics[0] as any : null;

    const execTimeDisplay = first ? this._safeFixed(first.executionTimeUs, 2) : '—';
    const smEffRaw = first?.smEfficiencyPercent ?? 0;
    const memTpRaw = first?.memoryThroughputGbs ?? 0;
    const smEffDisplay = (smEffRaw < 0) ? 'N/A' : (first ? this._safeFixed(smEffRaw, 1) : '—');
    const memTpDisplay = (memTpRaw < 0) ? 'N/A' : (first ? this._safeFixed(memTpRaw, 1) : '—');
    const occDisplay = first ? this._safeFixed(first.achievedOccupancyPercent, 1) : '—';

    const smClass = (smEffRaw < 0) ? 'metric-na' : (first ? this._colorClass(smEffRaw) : '');
    const occClass = first ? this._colorClass(first.achievedOccupancyPercent ?? 0) : '';

    const isFallback = result.profilingToolUsed === 'nvcc_fallback';

    const kernelTableRows = metrics.map((m: any, i: number) => {
      const smEff = m.smEfficiencyPercent ?? 0;
      const memTp = m.memoryThroughputGbs ?? 0;
      const smDisplay = smEff < 0 ? '<span class="metric-na">N/A</span>' : `${this._safeFixed(smEff, 1)}%`;
      const memDisplay = memTp < 0 ? '<span class="metric-na">N/A</span>' : `${this._safeFixed(memTp, 1)} GB/s`;
      const smCls = smEff < 0 ? '' : this._colorClass(smEff);
      const blockStr = Array.isArray(m.blockSize) ? this._formatDims(m.blockSize) : (m.blockSize ?? '—');
      const gridStr = Array.isArray(m.gridSize) ? this._formatDims(m.gridSize) : (m.gridSize ?? '—');
      return `
      <tr>
        <td>${i + 1}</td>
        <td class="kernel-name">${this._escapeHtml(m.kernelName || 'unknown')}</td>
        <td>${this._safeFixed(m.executionTimeUs, 2)} μs</td>
        <td class="${smCls}">${smDisplay}</td>
        <td>${memDisplay}</td>
        <td class="${this._colorClass(m.achievedOccupancyPercent ?? 0)}">${this._safeFixed(m.achievedOccupancyPercent, 1)}%</td>
        <td>${m.registersPerThread ?? 0}</td>
        <td>${blockStr}</td>
        <td>${gridStr}</td>
        <td>${m.limitingFactor || 'unknown'}</td>
      </tr>
    `;
    }).join('');

    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Profiling</title>
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
        * { box-sizing: border-box; margin: 0; padding: 0; }
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
        .header h1 { font-size: 1.5em; display: flex; align-items: center; gap: 10px; }
        .status-badge {
            padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 500;
        }
        .status-badge.success { background: rgba(76,175,80,.2); color: var(--success-color); }
        .status-badge.error { background: rgba(244,67,54,.2); color: var(--error-color); }
        .tool-badge {
            padding: 3px 10px; border-radius: 10px; font-size: 0.75em; font-weight: 600;
            background: rgba(100,100,255,.15); color: var(--accent-color);
        }
        .tool-badge.fallback {
            background: rgba(255,152,0,.15); color: #ff9800;
        }
        .export-buttons { display: flex; gap: 8px; }
        .export-btn {
            padding: 6px 12px; border: 1px solid var(--border-color); background: var(--card-bg);
            color: var(--text-color); border-radius: 4px; cursor: pointer; font-size: 0.85em;
        }
        .export-btn:hover { background: var(--accent-color); }

        /* Overview cards */
        .overview-grid {
            display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px;
        }
        @media (max-width: 800px) { .overview-grid { grid-template-columns: repeat(2, 1fr); } }
        .overview-card {
            background: var(--card-bg); border: 1px solid var(--border-color);
            border-radius: 8px; padding: 20px; text-align: center;
        }
        .overview-card .label {
            font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.5px;
            color: var(--vscode-descriptionForeground); margin-bottom: 8px;
        }
        .overview-card .value { font-size: 1.8em; font-weight: 700; }
        .overview-card .unit { font-size: 0.6em; color: var(--vscode-descriptionForeground); margin-left: 2px; }

        .metric-red { color: #f44336; }
        .metric-yellow { color: #ff9800; }
        .metric-green { color: #4caf50; }

        /* Kernel table */
        .table-container { margin-bottom: 24px; overflow-x: auto; }
        table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
        th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border-color); }
        th {
            background: var(--card-bg); font-size: 0.75em; text-transform: uppercase;
            letter-spacing: 0.5px; cursor: pointer; user-select: none; white-space: nowrap;
        }
        th:hover { background: var(--accent-color); }
        .kernel-name { font-family: var(--vscode-editor-font-family); font-size: 0.85em; word-break: break-all; }

        /* Raw output */
        details {
            margin-top: 24px; background: var(--card-bg); border: 1px solid var(--border-color);
            border-radius: 8px; overflow: hidden;
        }
        summary {
            padding: 12px 16px; cursor: pointer; font-weight: 600;
            background: rgba(255,255,255,.03);
        }
        .raw-output {
            padding: 12px 16px; font-family: var(--vscode-editor-font-family);
            font-size: 12px; overflow-x: auto; white-space: pre-wrap;
            max-height: 400px; overflow-y: auto;
            background: var(--vscode-terminal-background);
        }

        .error-message {
            background: rgba(244,67,54,.1); border: 1px solid var(--error-color);
            border-radius: 4px; padding: 16px; margin-bottom: 16px;
        }
        .warning-banner {
            background: rgba(255,152,0,.12); border: 1px solid #ff9800;
            border-radius: 6px; padding: 14px 18px; margin-bottom: 20px;
            line-height: 1.5;
        }
        .warning-banner strong { color: #ff9800; }
        .warning-banner code {
            background: rgba(255,255,255,.08); padding: 1px 5px; border-radius: 3px;
            font-family: var(--vscode-editor-font-family); font-size: 0.9em;
        }
        .metric-na { opacity: 0.35; }
        .file-info { font-size: 0.9em; color: var(--vscode-descriptionForeground); }
        .timestamp { font-size: 0.8em; color: var(--vscode-descriptionForeground); }

        .card-note {
            font-size: 0.65em; color: var(--vscode-descriptionForeground);
            margin-top: 4px; font-style: italic;
        }

        /* Performance Timeline */
        .timeline-section { margin-bottom: 24px; }
        .timeline-section h2 {
            font-size: 1.1em; margin-bottom: 12px;
            color: var(--vscode-descriptionForeground);
            display: flex; align-items: center; gap: 8px;
        }
        .timeline-section h2 .sample-count {
            font-size: 0.75em; font-weight: 400; opacity: 0.7;
        }
        .timeline-grid {
            display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px;
        }
        @media (max-width: 700px) { .timeline-grid { grid-template-columns: 1fr; } }
        .chart-card {
            background: var(--card-bg); border: 1px solid var(--border-color);
            border-radius: 8px; padding: 16px; position: relative;
        }
        .chart-card .chart-title {
            font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.5px;
            color: var(--vscode-descriptionForeground); margin-bottom: 8px;
        }
        .chart-canvas-wrap {
            position: relative; width: 100%; height: 140px;
        }
        canvas.timeline-chart {
            width: 100% !important; height: 100% !important;
            display: block;
        }
        .chart-legend {
            display: flex; gap: 14px; margin-top: 6px; font-size: 0.7em;
            color: var(--vscode-descriptionForeground);
        }
        .chart-legend .swatch {
            display: inline-block; width: 10px; height: 10px; border-radius: 2px;
            margin-right: 3px; vertical-align: middle;
        }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>
                ${statusIcon} Profiling Results
                <span class="status-badge ${statusClass}">${result.successful ? 'Success' : 'Failed'}</span>
                <span class="tool-badge${isFallback ? ' fallback' : ''}">${isFallback ? 'nvcc fallback' : (result.profilingToolUsed || '—')}</span>
            </h1>
            ${fileName ? `<div class="file-info">${this._escapeHtml(fileName)}</div>` : ''}
            <div class="file-info">${this._escapeHtml(result.gpuName || '—')} · CC ${this._escapeHtml(result.computeCapability || '—')}</div>
            <div class="timestamp">${new Date().toLocaleString()}</div>
        </div>
        <div class="export-buttons">
            <button class="export-btn" onclick="doExport('json')">JSON</button>
            <button class="export-btn" onclick="doExport('csv')">CSV</button>
            <button class="export-btn" onclick="copyRaw()">Copy Raw</button>
        </div>
    </div>

    ${!result.successful ? `
    <div class="error-message">
        <strong>Error:</strong> ${this._escapeHtml(result.errorMessage || 'Unknown error')}
    </div>` : ''}

    ${isFallback ? `
    <div class="warning-banner">
        <strong>⚠ Fallback profiler used</strong> — <code>ncu</code> could not access GPU hardware performance counters
        (container lacks <code>CAP_SYS_ADMIN</code>). Showing data from <code>nvcc --resource-usage</code> + NVML sampling + timing.
        SM Efficiency is approximated via NVML GPU utilization. Memory Throughput is estimated from NVML memory utilization × peak bandwidth.
    </div>` : ''}

    <!-- Overview cards -->
    <div class="overview-grid">
        <div class="overview-card">
            <div class="label">Execution Time</div>
            <div class="value">${execTimeDisplay}<span class="unit">μs</span></div>
        </div>
        <div class="overview-card">
            <div class="label">${isFallback ? 'GPU Utilization' : 'SM Efficiency'}</div>
            <div class="value ${smClass}">${smEffDisplay}${smEffDisplay !== 'N/A' ? '<span class="unit">%</span>' : ''}</div>
            ${isFallback && smEffDisplay !== 'N/A' ? '<div class="card-note">via NVML</div>' : ''}
        </div>
        <div class="overview-card">
            <div class="label">Memory Throughput</div>
            <div class="value">${memTpDisplay}${memTpDisplay !== 'N/A' ? '<span class="unit">GB/s</span>' : ''}</div>
            ${isFallback && memTpDisplay !== 'N/A' ? '<div class="card-note">estimated</div>' : ''}
        </div>
        <div class="overview-card">
            <div class="label">Occupancy</div>
            <div class="value ${occClass}">${occDisplay}<span class="unit">%</span></div>
        </div>
    </div>

    ${metrics.length > 0 ? `
    <!-- Per-kernel table -->
    <div class="table-container">
        <table id="kernelTable">
            <thead>
                <tr>
                    <th data-col="0">#</th>
                    <th data-col="1">Kernel Name</th>
                    <th data-col="2">Time</th>
                    <th data-col="3">${isFallback ? 'GPU Util.' : 'SM Eff.'}</th>
                    <th data-col="4">Mem Throughput</th>
                    <th data-col="5">Occupancy</th>
                    <th data-col="6">Registers</th>
                    <th data-col="7">Block Size</th>
                    <th data-col="8">Grid Size</th>
                    <th data-col="9">Limiting Factor</th>
                </tr>
            </thead>
            <tbody>
                ${kernelTableRows}
            </tbody>
        </table>
    </div>` : ''}

    ${this._getTimelineHtml(result)}

    ${result.rawNcuOutput ? `
    <details>
        <summary>Raw Profiler Output</summary>
        <div class="raw-output">${this._escapeHtml(result.rawNcuOutput)}</div>
    </details>` : ''}

    <script>
        const vscode = acquireVsCodeApi();
        const rawOutput = ${JSON.stringify(result.rawNcuOutput || '')};

        function doExport(format) {
            vscode.postMessage({ command: 'export', format });
        }
        function copyRaw() {
            vscode.postMessage({ command: 'copyToClipboard', text: rawOutput });
        }

        // Sortable table
        document.querySelectorAll('#kernelTable th').forEach(th => {
            th.addEventListener('click', () => {
                const table = document.getElementById('kernelTable');
                if (!table) return;
                const tbody = table.querySelector('tbody');
                if (!tbody) return;
                const col = parseInt(th.dataset.col || '0');
                const rows = Array.from(tbody.querySelectorAll('tr'));
                const asc = th.dataset.sort !== 'asc';
                th.dataset.sort = asc ? 'asc' : 'desc';
                rows.sort((a, b) => {
                    const aText = a.cells[col]?.textContent || '';
                    const bText = b.cells[col]?.textContent || '';
                    const aNum = parseFloat(aText);
                    const bNum = parseFloat(bText);
                    if (!isNaN(aNum) && !isNaN(bNum)) {
                        return asc ? aNum - bNum : bNum - aNum;
                    }
                    return asc ? aText.localeCompare(bText) : bText.localeCompare(aText);
                });
                rows.forEach(r => tbody.appendChild(r));
            });
        });
    </script>
</body>
</html>`;
  }

  private _getLoadingHtml(message: string): string {
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Profiling</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            background-color: var(--vscode-editor-background);
            color: var(--vscode-editor-foreground);
            display: flex; justify-content: center; align-items: center;
            height: 100vh; margin: 0;
        }
        .loading { text-align: center; }
        .spinner {
            width: 50px; height: 50px;
            border: 3px solid var(--vscode-panel-border);
            border-top-color: var(--vscode-button-background);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="loading">
        <div class="spinner"></div>
        <p>${message}</p>
        <p style="margin-top:8px;font-size:.85em;color:var(--vscode-descriptionForeground);">
            ncu replays kernels multiple times — this may take a while
        </p>
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
    <title>Profiling - Error</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            background-color: var(--vscode-editor-background);
            color: var(--vscode-editor-foreground);
            padding: 20px;
        }
        .error-container {
            background: rgba(244,67,54,.1); border: 1px solid #f44336;
            border-radius: 8px; padding: 20px; max-width: 600px; margin: 40px auto;
        }
        h1 { color: #f44336; margin-bottom: 16px; }
        pre {
            background: var(--vscode-terminal-background);
            padding: 12px; border-radius: 4px;
            overflow-x: auto; white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <div class="error-container">
        <h1>Profiling Error</h1>
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
    <title>Profiling</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            background-color: var(--vscode-editor-background);
            color: var(--vscode-editor-foreground);
            display: flex; justify-content: center; align-items: center;
            height: 100vh; margin: 0;
        }
        .empty { text-align: center; color: var(--vscode-descriptionForeground); }
    </style>
</head>
<body>
    <div class="empty">
        <h2>No Profiling Results Yet</h2>
        <p>Run "Profile Kernel on GPU" to see detailed profiling data here.</p>
        <p style="margin-top:12px;font-size:.9em;">
            CUDA kernels → Nsight Compute (ncu)<br>
            Triton kernels → torch.profiler
        </p>
    </div>
</body>
</html>`;
  }

  public dispose() {
    ProfilingPanel.currentPanel = undefined;
    this._panel.dispose();
    while (this._disposables.length) {
      const d = this._disposables.pop();
      if (d) { d.dispose(); }
    }
  }
}
