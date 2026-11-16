"""Minimal public GitHub release check, without credentials or installation UI."""
from __future__ import annotations

from PySide6.QtCore import QThread, QUrl, Signal, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from observer.updates import GitHubUpdates, RELEASES_URL, UpdateError, current_version


class UpdateTask(QThread):
    result = Signal(object)
    failure = Signal(str)

    def __init__(self, current, parent):
        super().__init__(parent)
        self.current = current

    def run(self):
        try:
            # The public repository needs no token. Include published previews
            # automatically so users do not have to choose a release channel.
            value = GitHubUpdates().check(self.current, True, self.isInterruptionRequested)
            if not self.isInterruptionRequested():
                self.result.emit(value)
        except UpdateError as exc:
            self.failure.emit(str(exc))
        except Exception:
            self.failure.emit('检查失败，请稍后重试或打开 GitHub 发布页。')


class UpdatePage(QWidget):
    def __init__(self, window, args):
        super().__init__(window)
        self.window = window
        self.task = None
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
        self.check.clicked.connect(self.check_updates)
        self.web.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(RELEASES_URL)))

    def busy(self):
        return self.task is not None and self.task.isRunning()

    def check_updates(self):
        if self.busy() or self.window.closing:
            return
        self.task = UpdateTask(self.current, self)
        self.task.result.connect(self.checked)
        self.task.failure.connect(self.status.setText)
        self.task.finished.connect(self.task_finished)
        self.check.setEnabled(False)
        self.status.setText('正在检查更新…')
        self.task.start()

    def checked(self, release):
        self.status.setText('暂无更新。' if release is None else
                            f'发现新版 {release.tag}，请打开 GitHub 发布页下载。')

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

    def refresh(self):
        self.check.setEnabled(not self.window.closing and not self.busy())
