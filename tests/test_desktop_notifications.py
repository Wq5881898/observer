import importlib.util
import os
import tempfile
import time
import unittest
from pathlib import Path
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch, PropertyMock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HAS_QT = importlib.util.find_spec('PySide6') is not None
if HAS_QT:
    from PySide6.QtWidgets import QApplication
    from observer.desktop import ObserverWindow
    from observer.desktop_tray import DesktopTray


@unittest.skipUnless(HAS_QT, 'desktop optional dependency not installed')
class NotificationTests(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        self.temp = tempfile.TemporaryDirectory()
        self.window = ObserverWindow(SimpleNamespace(
            data_dir=Path(self.temp.name), demo=True, models_dir=Path('models'), profile=None))
        self.wait_for(lambda: self.window.start.isEnabled())

    def wait_for(self, predicate):
        deadline = time.monotonic() + 5
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.app.processEvents()
        self.assertTrue(predicate())

    def tearDown(self):
        self.window.request_exit()
        self.wait_for(lambda: not self.window.worker.isRunning())
        self.window.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def test_duplicate_action_does_not_repeat_any_channel_or_acknowledgment(self):
        with patch.object(self.window.worker, 'submit') as submit, \
             patch.object(self.window.tray, 'notify', return_value='已提交') as notify, \
             patch('observer.desktop.play_reminder_sound') as sound:
            for _ in range(3):
                self.window.show_reminder('same-action', 'focus_violation', '提醒：手机')
            sound.assert_called_once()
            notify.assert_not_called()
            submit.assert_called_once_with('notification', 'same-action', True, None)

    def test_posture_never_uses_system_channel(self):
        with patch.object(self.window.worker, 'submit'), \
             patch.object(self.window.tray, 'notify') as notify, \
             patch('observer.desktop.play_reminder_sound') as sound:
            self.window.show_reminder('posture', 'posture_deviation', '提醒：姿态')
            self.window.show_reminder('posture', 'posture_deviation', '提醒：姿态')
            notify.assert_not_called()
            sound.assert_called_once_with('posture_deviation')

    def test_task_system_failure_does_not_change_focus_delivery(self):
        with patch.object(DesktopTray, 'available', new_callable=PropertyMock, return_value=True), \
             patch('observer.desktop_tray.QSystemTrayIcon.supportsMessages', return_value=True), \
             patch.object(self.window.tray.icon, 'showMessage', side_effect=RuntimeError('OS failure')) as show, \
             patch.object(self.window.worker, 'submit') as submit, \
             patch('observer.desktop.play_reminder_sound'):
            self.window.show_reminder('failure', 'focus_violation', '提醒：离席')
            self.window.show_reminder('failure', 'focus_violation', '提醒：离席')
            self.window.show_task_notification('task-failure', '任务启动失败')
            self.window.show_task_notification('task-failure', '任务启动失败')
            submit.assert_called_once_with('notification', 'failure', True, None)
            show.assert_called_once()
            self.assertIn('提交失败', self.window.notification_status.text())
            self.assertIn('提醒：离席', self.window.alert_banner.text())

    def test_submission_is_not_claimed_as_delivery_and_click_restores(self):
        with patch.object(DesktopTray, 'available', new_callable=PropertyMock, return_value=True), \
             patch('observer.desktop_tray.QSystemTrayIcon.supportsMessages', return_value=True), \
             patch.object(self.window.tray.icon, 'showMessage') as show:
            status = self.window.tray.notify('提醒：手机')
            self.assertIn('已提交', status)
            self.assertIn('Windows', status)
            self.assertNotIn('已送达', status)
            show.assert_called_once()
            self.window.hide()
            self.window.tray.icon.messageClicked.emit()
            self.assertTrue(self.window.isVisible())

    def test_unavailable_channel_never_calls_os(self):
        with patch.object(DesktopTray, 'available', new_callable=PropertyMock, return_value=False), \
             patch.object(self.window.tray.icon, 'showMessage') as show:
            self.assertIn('不可用', self.window.tray.notify('提醒：手机'))
            show.assert_not_called()

    def test_focus_application_failure_still_never_sends_system_notification(self):
        with patch.object(self.window.worker, 'submit') as submit, \
             patch.object(self.window.tray, 'notify', return_value='已提交') as notify, \
             patch('observer.desktop.play_reminder_sound', side_effect=RuntimeError('sound error')):
            self.window.show_reminder('sound-failure', 'focus_violation', '提醒：手机')
            submit.assert_called_once_with('notification', 'sound-failure', False, 'sound error')
            notify.assert_not_called()

    def test_task_toggle_persists_and_disabled_notices_are_not_replayed(self):
        from PySide6.QtCore import QSettings
        self.window.task_notifications.setChecked(False)
        settings = QSettings(str(Path(self.temp.name) / 'desktop.ini'), QSettings.Format.IniFormat)
        self.assertFalse(settings.value('task_notifications', True, type=bool))
        with patch.object(self.window.tray, 'notify', return_value='已提交') as notify:
            self.window.show_task_notification('disabled', '到点')
            notify.assert_not_called()
            self.window.task_notifications.setChecked(True)
            self.window.show_task_notification('disabled', '到点')
            notify.assert_not_called()
            self.window.show_task_notification('next', '新的任务到点')
            self.window.show_task_notification('next', '新的任务到点')
            notify.assert_called_once_with('新的任务到点')

    def test_scheduled_due_and_failure_both_reach_system_channel(self):
        from observer.product import ProductStore, utc_now
        with patch.object(self.window.tray, 'notify', return_value='已提交') as notify, \
             patch.object(self.window.worker, 'factory', side_effect=RuntimeError('摄像头不可用')):
            store = ProductStore(Path(self.temp.name))
            try:
                store.add_task('每天学习', utc_now() + timedelta(seconds=.2), 60)
            finally:
                store.close()
            self.wait_for(lambda: notify.call_count == 2)
            self.assertIn('学习时间到了：每天学习', notify.call_args_list[0].args[0])
            self.assertIn('60 分钟', notify.call_args_list[0].args[0])
            self.assertIn('启动失败', notify.call_args_list[1].args[0])

    def test_scheduled_due_reaches_system_channel_while_hidden_and_running(self):
        from observer.product import ProductStore, utc_now
        self.window.hide()
        with patch.object(self.window.tray, 'notify', return_value='本次通知已提交') as notify:
            store = ProductStore(Path(self.temp.name))
            try:
                store.add_task('隐藏窗口正常启动', utc_now() + timedelta(seconds=.2), 1)
            finally:
                store.close()
            self.wait_for(lambda: notify.call_count == 1)
            self.wait_for(lambda: self.window.state == 'monitoring')
            notify.assert_called_once()
            self.assertIn('学习时间到了：隐藏窗口正常启动', notify.call_args.args[0])
            self.assertEqual(self.window.notification_status.text(), '本次通知已提交')
            self.assertFalse(self.window.isVisible())

    def test_manual_start_does_not_send_task_notification(self):
        with patch.object(self.window.tray, 'notify') as notify:
            self.window.start.click()
            self.wait_for(lambda: self.window.state == 'monitoring')
            notify.assert_not_called()
