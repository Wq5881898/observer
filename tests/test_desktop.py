import importlib.util
import os
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HAS_QT = importlib.util.find_spec('PySide6') is not None
if HAS_QT:
    from PySide6.QtWidgets import QApplication
    from observer.desktop import ObserverWindow
from observer.product import ProductStore, utc_now
from observer.product_cli import DemoSource


@unittest.skipUnless(HAS_QT, 'desktop optional dependency not installed')
class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.args = SimpleNamespace(data_dir=Path(self.temp.name), demo=True,
                                    models_dir=Path('models'), profile=None)
        self.window = None

    def open(self, factory=None):
        self.window = ObserverWindow(self.args, factory)
        self.window.show()
        self.wait_for(lambda: self.window.start.isEnabled())

    def wait_for(self, predicate, timeout=5):
        deadline = time.monotonic()+timeout
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.app.processEvents()
        self.assertTrue(predicate(), 'desktop event timeout')

    def tearDown(self):
        if self.window:
            self.window.close()
            self.wait_for(lambda: not self.window.worker.isRunning(), 10)
            self.app.processEvents()
            self.window.deleteLater()
            self.app.processEvents()
        self.temp.cleanup()

    def test_buttons_pause_resume_end_and_history(self):
        self.open()
        self.window.title.setText('窗口测试')
        self.window.start.click()
        self.wait_for(lambda: self.window.state == 'monitoring')
        self.window.pause.click()
        self.wait_for(lambda: self.window.state == 'paused')
        self.assertEqual(self.window.pause.text(), '继续')
        self.window.pause.click()
        self.wait_for(lambda: self.window.state == 'monitoring')
        self.window.end.click()
        self.wait_for(lambda: self.window.state == 'idle')
        self.assertIn('窗口测试', self.window.details.toPlainText())
        self.assertEqual(self.window.history.count(), 1)
        store = ProductStore(self.args.data_dir)
        try:
            self.assertEqual(store.history()[0]['state'], 'ended')
        finally:
            store.close()

    def test_alert_is_visible_and_keeps_plain_notices_plain(self):
        self.open()
        with patch('observer.desktop.play_reminder_sound') as sound:
            self.window.show_notice('提醒：离开屏幕（软提示）')
            sound.assert_called_once_with()
        self.assertIn('提醒：', self.window.message.text())
        self.assertIn('font-weight', self.window.message.styleSheet())
        self.assertIn('已记录', self.window.alert_banner.text())
        self.assertFalse(self.window.alert_banner.isHidden())
        self.window.show_notice('已开始')
        self.assertEqual(self.window.message.styleSheet(), '')
        self.assertIn('已记录', self.window.alert_banner.text())

    def test_live_attention_candidate_is_visible_before_alert(self):
        self.open()
        self.window.start.click()
        self.wait_for(lambda: self.window.state == 'monitoring')
        self.window.worker.submit('signal', 'attention_away')
        self.wait_for(lambda: '头眼：屏外候选' in self.window.status.text())

    def test_summary_exposes_soft_signal_totals_without_calling_them_continuous(self):
        from observer.desktop import summary_text
        result = {'score': 100, 'identity': '{"source":"camera"}', 'title': '测试',
                  'state': 'ended', 'coverage': 1, 'actions': [],
                  'totals': {'attention_away': {'violation': 16.189},
                             'posture_deviation': {'violation': 0},
                             'ergonomic_risk': {'violation': 4.2}}}
        text = summary_text(result)
        self.assertIn('16.2秒', text)
        self.assertIn('累计时间不等于连续触发时间', text)

    def test_scheduled_task_starts_and_ends_once(self):
        self.open()
        self.window.worker.submit('task', '定时测试', utc_now()+timedelta(seconds=.4), .01)
        self.wait_for(lambda: self.window.state == 'monitoring')
        self.wait_for(lambda: self.window.state == 'idle')
        store = ProductStore(self.args.data_dir)
        try:
            self.assertEqual(len(store.history()), 1)
            self.assertEqual(store.tasks()[0]['state'], 'ended')
        finally:
            store.close()

    def test_restart_starts_task_within_late_grace(self):
        store = ProductStore(self.args.data_dir)
        try:
            store.add_task('重启补启动', utc_now()-timedelta(seconds=30), .01)
        finally:
            store.close()
        self.window = ObserverWindow(self.args)
        self.window.show()
        self.wait_for(lambda: self.window.state == 'monitoring')
        self.wait_for(lambda: self.window.state == 'idle')
        store = ProductStore(self.args.data_dir)
        try:
            self.assertEqual(store.tasks()[0]['state'], 'ended')
            self.assertEqual(store.history()[0]['title'], '重启补启动')
        finally:
            store.close()

    def test_failed_source_leaves_buttons_usable_and_no_session(self):
        def broken():
            raise RuntimeError('模型不可用')
        self.open(broken)
        self.window.start.click()
        self.wait_for(lambda: '模型不可用' in self.window.message.text())
        self.wait_for(lambda: self.window.start.isEnabled())
        self.assertEqual(self.window.history.count(), 0)

    def test_scheduled_source_failure_is_persisted_in_task_list(self):
        store = ProductStore(self.args.data_dir)
        try:
            store.add_task('失败任务', utc_now()-timedelta(seconds=1), 1)
        finally:
            store.close()
        calls = 0
        def broken():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError('摄像头被占用')
            return DemoSource()
        self.window = ObserverWindow(self.args, broken)
        self.window.show()
        self.wait_for(lambda: '任务启动失败' in self.window.message.text())
        self.wait_for(lambda: self.window.tasks.count() == 1)
        self.assertIn('摄像头被占用', self.window.tasks.item(0).text())
        store = ProductStore(self.args.data_dir)
        try:
            task = store.tasks()[0]
            self.assertEqual(task['state'], 'failed')
            self.assertEqual(task['error'], '摄像头被占用')
        finally:
            store.close()
        self.window.tasks.setCurrentRow(0)
        self.window.retry_task.click()
        self.wait_for(lambda: self.window.state == 'monitoring')
        self.window.end.click()
        self.wait_for(lambda: self.window.state == 'idle')

    def test_close_while_source_loading_keeps_ui_responsive(self):
        def slow():
            time.sleep(.3)
            return DemoSource()
        self.open(slow)
        self.window.start.click()
        self.wait_for(lambda: '准备观察' in self.window.message.text())
        self.window.close()
        self.assertTrue(self.window.closing)
        self.wait_for(lambda: not self.window.worker.isRunning())
        store = ProductStore(self.args.data_dir)
        try:
            self.assertEqual(store.history(), [])
        finally:
            store.close()

    def test_close_active_session_persists_interruption(self):
        self.open()
        self.window.start.click()
        self.wait_for(lambda: self.window.state == 'monitoring')
        self.window.close()
        self.wait_for(lambda: not self.window.worker.isRunning())
        store = ProductStore(self.args.data_dir)
        try:
            self.assertEqual(store.history()[0]['state'], 'interrupted')
        finally:
            store.close()

    def test_scheduled_form_uses_future_time_and_can_cancel(self):
        self.open()
        self.window.scheduled.setChecked(True)
        self.window.start.click()
        self.wait_for(lambda: self.window.tasks.count() == 1)
        self.assertEqual(self.window.state, 'idle')
        self.window.tasks.setCurrentRow(0)
        self.window.cancel_selected()
        self.wait_for(lambda: self.window.tasks.count() == 0)

    def test_resume_failure_keeps_session_paused(self):
        calls = 0
        def factory():
            nonlocal calls
            calls += 1
            if calls > 1:
                raise RuntimeError('摄像头被其他程序占用')
            return DemoSource()
        self.open(factory)
        self.window.start.click()
        self.wait_for(lambda: self.window.state == 'monitoring')
        self.window.pause.click()
        self.wait_for(lambda: self.window.state == 'paused')
        self.window.pause.click()
        self.wait_for(lambda: '占用' in self.window.message.text())
        self.wait_for(lambda: self.window.end.isEnabled())
        self.assertEqual(self.window.state, 'paused')
        self.window.end.click()
        self.wait_for(lambda: self.window.state == 'idle')

    def test_read_failure_saves_interruption_and_releases_source(self):
        closed = []
        calls = 0
        class BrokenSource(DemoSource):
            def read(self, at):
                raise RuntimeError('模拟推理异常')
            def close(self):
                closed.append(True)
        def factory():
            nonlocal calls
            calls += 1
            return BrokenSource() if calls == 1 else DemoSource()
        self.open(factory)
        self.window.start.click()
        self.wait_for(lambda: '会话已中断' in self.window.message.text())
        self.wait_for(lambda: self.window.start.isEnabled())
        self.assertTrue(self.window.worker.isRunning())
        self.assertEqual(closed, [True])
        store = ProductStore(self.args.data_dir)
        try:
            row = store.history()[0]
            self.assertEqual(row['state'], 'interrupted')
            summary = store.summary(row['id'])
            self.assertEqual(summary['deducted'], 0)
            self.assertEqual(summary['error'], '模拟推理异常')
            self.assertIn('中断原因：模拟推理异常', self.window.details.toPlainText())
        finally:
            store.close()
        self.window.start.click()
        self.wait_for(lambda: self.window.state == 'monitoring')
        self.window.end.click()
        self.wait_for(lambda: self.window.state == 'idle')

    def test_stalled_camera_reconnects_once_and_keeps_session(self):
        from observer.product_cli import SourceStalled
        class RecoveringSource(DemoSource):
            def __init__(self):
                super().__init__()
                self.reconnected = False
                self.reads = 0
            def read(self, at):
                self.reads += 1
                if not self.reconnected and self.reads >= 2:
                    raise SourceStalled('连续没有新鲜摄像头画面')
                return super().read(at)
            def reconnect(self):
                self.reconnected = True
        source = RecoveringSource()
        self.open(lambda: source)
        self.window.start.click()
        self.wait_for(lambda: '摄像头已恢复' in self.window.message.text())
        self.assertEqual(self.window.state, 'monitoring')
        self.assertTrue(self.window.worker.isRunning())
        store = ProductStore(self.args.data_dir)
        try:
            self.assertEqual(len(store.history()), 1)
            self.assertEqual(store.history()[0]['state'], 'monitoring')
        finally:
            store.close()
        self.window.end.click()
        self.wait_for(lambda: self.window.state == 'idle')

    def test_failed_automatic_reconnect_interrupts_only_the_session(self):
        from observer.product_cli import SourceStalled
        class UnrecoverableSource(DemoSource):
            def read(self, at):
                raise SourceStalled('摄像头线程停止')
            def reconnect(self):
                raise SourceStalled('摄像头仍被占用')
        self.open(UnrecoverableSource)
        self.window.start.click()
        self.wait_for(lambda: '自动重连失败' in self.window.message.text())
        self.wait_for(lambda: self.window.start.isEnabled())
        self.assertTrue(self.window.worker.isRunning())
        store = ProductStore(self.args.data_dir)
        try:
            result = store.summary(store.history()[0]['id'])
            self.assertEqual(result['state'], 'interrupted')
            self.assertIn('摄像头仍被占用', result['error'])
        finally:
            store.close()

    def test_second_window_cannot_interrupt_first_session(self):
        self.open()
        self.window.start.click()
        self.wait_for(lambda: self.window.state == 'monitoring')
        second = ObserverWindow(self.args)
        try:
            second.show()
            self.wait_for(lambda: not second.worker.isRunning())
            self.assertIn('已有', second.message.text())
            self.assertEqual(self.window.state, 'monitoring')
            store = ProductStore(self.args.data_dir)
            try:
                self.assertEqual(store.history()[0]['state'], 'monitoring')
            finally:
                store.close()
        finally:
            second.close()
            second.deleteLater()
            self.app.processEvents()
