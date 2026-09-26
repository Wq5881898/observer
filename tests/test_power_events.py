"""Synthetic native messages; no machine sleep or camera is required."""
import ctypes
from ctypes import wintypes
import importlib.util
import unittest


HAS_QT = importlib.util.find_spec('PySide6') is not None
if HAS_QT:
    from PySide6.QtCore import QByteArray
    from observer.power_events import (
        PBT_APMRESUMEAUTOMATIC, PBT_APMSUSPEND, POWER_TRANSITION_REASON,
        WM_POWERBROADCAST, WindowsPowerEventFilter,
    )


@unittest.skipUnless(HAS_QT, 'desktop optional dependency not installed')
class WindowsPowerEventFilterTests(unittest.TestCase):
    def test_suspend_and_automatic_resume_signal_once_each(self):
        seen = []
        event_filter = WindowsPowerEventFilter(seen.append)
        message = wintypes.MSG()
        message.message = WM_POWERBROADCAST
        for power_code in (PBT_APMSUSPEND, PBT_APMRESUMEAUTOMATIC):
            message.wParam = power_code
            self.assertFalse(event_filter.nativeEventFilter(
                QByteArray(b'windows_generic_MSG'), ctypes.addressof(message)))
        self.assertEqual(seen, [POWER_TRANSITION_REASON] * 2)

    def test_other_messages_and_duplicate_user_resume_are_ignored(self):
        seen = []
        event_filter = WindowsPowerEventFilter(seen.append)
        message = wintypes.MSG()
        message.message = WM_POWERBROADCAST
        message.wParam = 7  # PBT_APMRESUMESUSPEND follows automatic resume.
        self.assertFalse(event_filter.nativeEventFilter(
            QByteArray(b'windows_generic_MSG'), ctypes.addressof(message)))
        message.wParam = PBT_APMSUSPEND
        self.assertFalse(event_filter.nativeEventFilter(
            QByteArray(b'other_platform'), ctypes.addressof(message)))
        message.message = 0x0010  # WM_CLOSE
        self.assertFalse(event_filter.nativeEventFilter(
            QByteArray(b'windows_dispatcher_MSG'), ctypes.addressof(message)))
        self.assertEqual(seen, [])
