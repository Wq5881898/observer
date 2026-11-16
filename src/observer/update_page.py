"""Two-button public updater: check, download, verify and switch when idle."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, QTimer, QUrl, Signal, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from observer.updates import GitHubUpdates, RELEASES_URL, UpdateError, current_version
from observer.deployment import is_frozen
from observer.update_install import finish_upgrade


class UpdateTask(QThread):
    result = Signal(object)
    failure = Signal(str)
    progress = Signal(int)

    def __init__(self, current, parent, release=None, destination=None):
        super().__init__(parent)
        self.current = current
        self.release, self.destination = release, destination

    def run(self):
        try:
            # The public repository needs no token. Include published previews
            # automatically so users do not have to choose a release channel.
            client = GitHubUpdates()
            if self.release is None:
                value = client.check(self.current, True, self.isInterruptionRequested)
            else:
                value = client.stage(self.release, self.destination,
                    lambda done, total: self.progress.emit(int(done*100/total)),
                    self.isInterruptionRequested)
            if not self.isInterruptionRequested():
                self.result.emit(value)
        except UpdateError as exc:
            self.failure.emit(str(exc))
        except Exception:
            self.failure.emit('更新未完成，请重试或打开 GitHub 发布页。')


class UpdatePage(QWidget):
    def __init__(self, window, args):
        super().__init__(window)
        self.window = window
        self.args = args
        self.task = None
        self.release = None
        self.staged_app = None
        self.auto_switch = False
        self.switch_queued = False
        self.switching = False
        self.current, _build = current_version()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 24, 18, 18)
        layout.setSpacing(16)
        self.version_label = QLabel(f'当前版本  {self.current}')
        self.version_label.setStyleSheet('font-size:18px;font-weight:600;color:#0f172a;')
        layout.addWidget(self.version_label)
        buttons = QHBoxLayout()
        buttons.setSpacing(12)
        self.check = QPushButton('检查更新')
        self.web = QPushButton('GitHub 发布页')
        for button in (self.check, self.web):
            button.setMinimumHeight(36)
            buttons.addWidget(button, 1)
        self.check.setStyleSheet('background:#2563eb;color:white;border-radius:5px;')
        layout.addLayout(buttons)
        self.status = QLabel('从 GitHub 获取最新版本。')
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setStyleSheet('color:#64748b;')
        layout.addWidget(self.status)
        layout.addStretch(1)
        self.check.clicked.connect(self.primary_action)
        self.web.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(RELEASES_URL)))

    def busy(self):
        return self.task is not None and self.task.isRunning()

    def check_updates(self):
        if self.busy() or self.window.closing:
            return
        self.task = UpdateTask(self.current, self)
        self.task.result.connect(self.checked)
        self.connect_task()
        self.status.setText('正在检查更新…')
        self.task.start()
        self.refresh()

    def connect_task(self):
        self.task.failure.connect(self.failed)
        self.task.finished.connect(self.task_finished)

    def primary_action(self):
        if self.busy():
            if self.task.release is not None:
                self.cancel_task()
                self.status.setText('正在取消下载…')
            return
        if self.staged_app is not None:
            if not is_frozen():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.staged_app)))
            elif self.auto_switch:
                self.auto_switch = False
                self.status.setText('已取消自动切换，下载的新版已保留。')
                self.refresh()
            else:
                self.auto_switch = True
                self.refresh()
        elif self.release is not None:
            self.download_update()
        else:
            self.check_updates()

    def checked(self, release):
        self.release = release
        self.status.setText('暂无更新。' if release is None else
                            f'发现新版 {release.tag}，点击下载并更新。')

    def download_update(self):
        if self.busy() or self.window.closing or self.release is None:
            return
        self.task = UpdateTask(self.current, self, self.release,
                               Path(self.args.data_dir).resolve()/'updates')
        self.task.result.connect(self.downloaded)
        self.task.progress.connect(self.download_progress)
        self.connect_task()
        self.status.setText('正在下载更新…')
        self.task.start()
        self.refresh()

    def download_progress(self, percent):
        self.status.setText(f'正在下载更新… {percent}%' if percent < 100 else '正在校验并解压更新…')

    def downloaded(self, app):
        self.staged_app = Path(app)
        self.auto_switch = is_frozen()
        self.status.setText('下载和校验完成。' if self.auto_switch else '下载和校验完成，点击打开新版目录。')

    def failed(self, message):
        self.auto_switch = False
        self.status.setText(message)

    def task_finished(self):
        task, self.task = self.task, None
        if task is not None:
            task.deleteLater()
        self.refresh()
        if self.window.closing:
            self.window.close()

    def cancel_task(self):
        if self.busy():
            self.task.requestInterruption()
            self.auto_switch = False

    def idle(self):
        return (self.window.state == 'idle' and not self.window.blocked
                and self.window.pending_command is None and not self.window.closing)

    def refresh(self):
        busy = self.busy()
        self.check.setEnabled(not self.window.closing and not self.switching
                              and (not busy or self.task.release is not None))
        if busy:
            self.check.setText('取消下载' if self.task.release is not None else '检查中…')
        elif self.staged_app is not None:
            self.check.setText('取消自动更新' if self.auto_switch else
                               '重启更新' if is_frozen() else '打开新版目录')
            if self.auto_switch and not self.idle() and not self.window.closing:
                self.status.setText('新版已就绪，当前计时结束后自动更新。')
            if self.auto_switch and self.idle() and not self.switch_queued:
                self.switch_queued = True
                QTimer.singleShot(0, self.restart_update)
        else:
            self.check.setText('下载并更新' if self.release is not None else '检查更新')

    def restart_update(self):
        self.switch_queued = False
        if (not self.auto_switch or self.staged_app is None or not is_frozen()
                or not self.idle() or self.busy() or self.switching):
            return
        self.switching = True
        self.check.setEnabled(False)
        self.status.setText('正在保存并启动新版…')
        app = self.staged_app

        def after_exit():
            finish_upgrade(app, self.args)

        # exit_ready follows the worker's database/camera release, and the
        # existing wake endpoint's close slot. Never launch while a task runs.
        self.window.exit_ready.connect(after_exit)
        self.window.request_exit()
