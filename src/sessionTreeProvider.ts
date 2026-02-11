import * as path from 'path';
import * as vscode from 'vscode';
import { SessionInfo, ModalNotebookController } from './notebookController';
import { KernelSessionState } from './types';

class SessionTreeItem extends vscode.TreeItem {
  public readonly notebookUri: string;

  constructor(session: SessionInfo) {
    const filename = path.basename(vscode.Uri.parse(session.notebookUri).fsPath);
    super(filename, vscode.TreeItemCollapsibleState.None);

    this.notebookUri = session.notebookUri;

    const stateLabels: Record<KernelSessionState, string> = {
      starting: 'starting…',
      idle: 'idle',
      busy: 'executing…',
      disconnected: 'disconnected',
    };
    const stateIcons: Record<KernelSessionState, string> = {
      starting: 'sync~spin',
      idle: 'pass-filled',
      busy: 'loading~spin',
      disconnected: 'error',
    };

    const label = stateLabels[session.state] || session.state;
    const icon = stateIcons[session.state] || 'question';

    this.description = `${session.gpu} — ${label}`;
    this.tooltip = `GPU: ${session.gpuName || session.gpu}\nStatus: ${label}`;
    this.iconPath = new vscode.ThemeIcon(icon);
    this.contextValue = 'activeSession';
  }
}

export class SessionTreeProvider implements vscode.TreeDataProvider<vscode.TreeItem> {
  private _onDidChangeTreeData = new vscode.EventEmitter<void>();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  private _controller: ModalNotebookController;

  constructor(controller: ModalNotebookController) {
    this._controller = controller;
    controller.onSessionsChanged(() => this.refresh());
  }

  refresh(): void {
    this._onDidChangeTreeData.fire();
  }

  getTreeItem(element: vscode.TreeItem): vscode.TreeItem {
    return element;
  }

  getChildren(): vscode.TreeItem[] {
    const sessions = this._controller.getActiveSessions();
    if (sessions.length === 0) {
      const empty = new vscode.TreeItem('No active sessions');
      empty.description = 'Select Modal GPU kernel in a notebook';
      return [empty];
    }
    return sessions.map(s => new SessionTreeItem(s));
  }
}
