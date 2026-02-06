import * as path from 'path';
import * as vscode from 'vscode';
import { SessionInfo, ModalNotebookController } from './notebookController';

class SessionTreeItem extends vscode.TreeItem {
  public readonly notebookUri: string;

  constructor(session: SessionInfo) {
    const filename = path.basename(vscode.Uri.parse(session.notebookUri).fsPath);
    super(filename, vscode.TreeItemCollapsibleState.None);

    this.notebookUri = session.notebookUri;
    this.description = `${session.gpu} — ${session.ready ? 'ready' : 'starting…'}`;
    this.tooltip = `GPU: ${session.gpuName || session.gpu}\nStatus: ${session.ready ? 'Ready' : 'Starting…'}`;
    this.iconPath = new vscode.ThemeIcon(session.ready ? 'pass-filled' : 'sync~spin');
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
