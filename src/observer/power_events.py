"""Forward Windows suspend/resume broadcasts without doing worker I/O in Qt's callback."""
from __future__ import annotations

import ctypes
from ctypes import wintypes

from PySide6.QtCore import QAbstractNativeEventFilter


WM_POWERBROADCAST = 0x0218
PBT_APMSUSPEND = 0x0004
PBT_APMRESUMEAUTOMATIC = 0x0012
POWER_TRANSITION_REASON = '检测到 Windows 系统挂起或唤醒'


class WindowsPowerEventFilter(QAbstractNativeEventFilter):
    """Only notify the worker; Qt must be allowed to process the message too."""

    def __init__(self, on_transition):
        super().__init__()
        self.on_transition = on_transition

    def nativeEventFilter(self, event_type, message):
        if bytes(event_type) not in (b'windows_generic_MSG', b'windows_dispatcher_MSG'):
            return False
        native = ctypes.cast(int(message), ctypes.POINTER(wintypes.MSG)).contents
        if (native.message == WM_POWERBROADCAST and
                native.wParam in (PBT_APMSUSPEND, PBT_APMRESUMEAUTOMATIC)):
            self.on_transition(POWER_TRANSITION_REASON)
        return False
