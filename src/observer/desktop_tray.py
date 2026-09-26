"""Qt-only tray presentation; no product, camera or database ownership."""
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon


def observer_icon():
    icon = QIcon()
    for size in (16, 32, 64):
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor('#2563eb'))
        painter.drawEllipse(0, 0, size, size)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor('white'), size / 10))
        inset = size // 4
        painter.drawEllipse(inset, inset, size - 2 * inset, size - 2 * inset)
        painter.end()
        icon.addPixmap(pixmap)
    return icon


class DesktopTray(QObject):
    show_requested = Signal()
    pause_requested = Signal()
    end_requested = Signal()
    quit_requested = Signal()

    def __init__(self, window):
        super().__init__(window)
        icon = observer_icon()
        window.setWindowIcon(icon)
        self.icon = QSystemTrayIcon(icon, self)
        self.menu = QMenu(window)
        self.show_action = self.menu.addAction('显示窗口')
        self.pause_action = self.menu.addAction('暂停当前会话')
        self.end_action = self.menu.addAction('结束当前会话')
        self.menu.addSeparator()
        self.quit_action = self.menu.addAction('退出')
        self.show_action.triggered.connect(self.show_requested.emit)
        self.pause_action.triggered.connect(self.pause_requested.emit)
        self.end_action.triggered.connect(self.end_requested.emit)
        self.quit_action.triggered.connect(self.quit_requested.emit)
        self.icon.setContextMenu(self.menu)
        self.icon.activated.connect(self._activated)
        self.icon.messageClicked.connect(self.show_requested.emit)
        self.update('等待开始', paused=False, can_pause=False, can_end=False)
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.icon.show()

    @property
    def available(self):
        return self.icon.isVisible() and QSystemTrayIcon.isSystemTrayAvailable()

    def _activated(self, reason):
        if reason in (QSystemTrayIcon.ActivationReason.Trigger,
                      QSystemTrayIcon.ActivationReason.DoubleClick):
            self.show_requested.emit()

    def update(self, status, *, paused, can_pause, can_end, closing=False):
        self.icon.setToolTip('Observer · ' + status)
        self.pause_action.setText('继续当前会话' if paused else '暂停当前会话')
        self.pause_action.setEnabled(can_pause and not closing)
        self.end_action.setEnabled(can_end and not closing)
        self.quit_action.setEnabled(not closing)

    def close(self):
        self.menu.close()
        self.icon.hide()

    def notify(self, message):
        if not self.available or not QSystemTrayIcon.supportsMessages():
            return '系统通知不可用；请查看窗口提醒。'
        try:
            # Do not add an application sound here; OS sound policy is external.
            self.icon.showMessage('Observer · 任务通知', message,
                                  QSystemTrayIcon.MessageIcon.NoIcon, 5000)
        except Exception:
            return '系统通知提交失败；请查看窗口提醒。'
        return '系统通知已提交；是否显示取决于 Windows 通知设置，请勿依赖通知作为唯一提醒。'
