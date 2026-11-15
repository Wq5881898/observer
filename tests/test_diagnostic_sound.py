import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
HAS_QT = importlib.util.find_spec('PySide6') is not None
if HAS_QT:
    from observer import desktop

spec=importlib.util.spec_from_file_location('sound_diagnostics',Path(__file__).resolve().parents[1]/'packaging/frozen_diagnostics.py')
diag=importlib.util.module_from_spec(spec);spec.loader.exec_module(diag)

@unittest.skipUnless(HAS_QT, 'desktop optional dependency not installed')
class DiagnosticSoundTests(unittest.TestCase):
    def test_synthetic_events_cannot_play_real_audio_or_show_notification(self):
        def exercise(_root):
            desktop.play_reminder_sound('phone_held')
            desktop.play_completion_sound()
            desktop.DesktopTray.notify(None,'模拟计时结束')
            return {}
        with patch.object(diag,'_desktop_smoke',side_effect=exercise), \
             patch.object(desktop,'play_reminder_sound') as alarm, \
             patch.object(desktop,'play_completion_sound') as complete, \
             patch.object(desktop.DesktopTray,'notify') as notification:
            report=diag.desktop_smoke(Path('unused'))
            alarm.assert_not_called();complete.assert_not_called();notification.assert_not_called()
            self.assertIs(desktop.play_completion_sound,complete)
        self.assertTrue(report['diagnostic_audio_suppressed'])
        self.assertEqual(report['completion_sound_requests'],1)
        self.assertEqual(report['alarm_sound_requests'],1)

    def test_failure_restores_normal_application_sound_functions(self):
        alarm,complete,notify=desktop.play_reminder_sound,desktop.play_completion_sound,desktop.DesktopTray.notify
        with patch.object(diag,'_desktop_smoke',side_effect=RuntimeError('synthetic failure')):
            with self.assertRaisesRegex(RuntimeError,'synthetic failure'):
                diag.desktop_smoke(Path('unused'))
        self.assertIs(desktop.play_reminder_sound,alarm)
        self.assertIs(desktop.play_completion_sound,complete)
        self.assertIs(desktop.DesktopTray.notify,notify)
