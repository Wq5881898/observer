"""E1 lifecycle tests with real Qt/worker/SQLite and a simulated tray shell."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HAS_QT = importlib.util.find_spec('PySide6') is not None
if HAS_QT:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from observer.desktop import ObserverWindow
    from observer.desktop_tray import DesktopTray
from observer.product import ProductStore, RuntimeLock, utc_now
from observer.product_cli import BackgroundCameraPreloader, DemoSource


class TrackedSource(DemoSource):
    def __init__(self, records, close_gate=None):
        super().__init__()
        self.records = records
        self.close_gate = close_gate
        self.reads = 0
        records.append(('open', threading.get_ident()))

    def read(self, at):
        self.reads += 1
        return super().read(at)

    def close(self):
        if self.close_gate is not None:
            self.close_gate.wait(5)
        self.records.append(('close', threading.get_ident()))


@unittest.skipUnless(HAS_QT, 'desktop optional dependency not installed')
class DesktopLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.args = SimpleNamespace(data_dir=Path(self.temp.name), demo=True,
                                    models_dir=Path('models'), profile=None)
        # Only substitute the OS tray. All actions, events and owners remain real.
        self.platform = patch('observer.desktop_tray.QSystemTrayIcon.isSystemTrayAvailable',
                              return_value=False)
        self.platform.start()
        self.available_patch = patch.object(DesktopTray, 'available',
                                            new_callable=PropertyMock, return_value=True)
        self.available = self.available_patch.start()
        self.windows = []
        self.gates = []

    def wait_for(self, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.app.processEvents()
        self.assertTrue(predicate(), 'lifecycle event timeout')

    def open(self, factory=None):
        window = ObserverWindow(self.args, factory)
        self.windows.append(window)
        window.show()
        self.wait_for(lambda: window.start.isEnabled())
        return window

    def gate(self):
        gate = threading.Event()
        self.gates.append(gate)
        return gate

    def tearDown(self):
        for gate in self.gates:
            gate.set()
        for window in reversed(self.windows):
            window.request_exit()
            self.wait_for(lambda: not window.worker.isRunning(), 10)
            window.deleteLater()
        self.app.processEvents()
        self.available_patch.stop()
        self.platform.stop()
        self.temp.cleanup()

    def test_hide_restore_and_minimize_keep_one_owner_and_live_observation(self):
        records, sources = [], []

        def factory():
            source = TrackedSource(records)
            sources.append(source)
            return source

        window = self.open(factory)
        window.start.click()
        self.wait_for(lambda: window.state == 'monitoring')
        owner, store = window.worker, window.worker.store
        for _ in range(3):
            window.close()
            self.assertFalse(window.isVisible())
            self.assertTrue(window.hidden_to_tray)
            self.wait_for(lambda: not window.worker.ui_visible)
            before = sources[0].reads
            self.wait_for(lambda: sources[0].reads > before)
            self.assertFalse(window.worker.isInterruptionRequested())
            with self.assertRaises(ValueError):
                RuntimeLock(self.args.data_dir)
            window.tray.show_action.trigger()
            self.assertTrue(window.isVisible())
            self.wait_for(lambda: window.worker.ui_visible)
            self.assertIs(window.worker, owner)
            self.assertIs(window.worker.store, store)
        window.showMinimized()
        self.app.processEvents()
        self.assertTrue(window.isMinimized())
        window.tray.show_action.trigger()
        self.assertFalse(window.isMinimized())
        self.assertEqual(len(sources), 1)
        window.exit_button.click()
        self.wait_for(lambda: window.exit_completed)
        self.assertEqual([item[0] for item in records], ['open', 'close'])
        self.assertEqual(records[0][1], records[1][1])
        self.assertNotEqual(records[0][1], threading.get_ident())

    def test_tray_commands_share_window_busy_guard_and_resume_same_session(self):
        records = []
        window = self.open(lambda: TrackedSource(records))
        self.assertFalse(window.tray.pause_action.isEnabled())
        self.assertFalse(window.tray.end_action.isEnabled())
        window.start.click()
        self.wait_for(lambda: window.state == 'monitoring')
        identifier = window.worker.service.identifier
        window.close()
        window.tray.pause_action.trigger()
        self.assertFalse(window.tray.pause_action.isEnabled())
        self.assertFalse(window.tray.end_action.isEnabled())
        window.tray.pause_action.trigger()
        self.wait_for(lambda: window.state == 'paused' and not window.blocked)
        self.assertEqual(window.tray.pause_action.text(), '继续当前会话')
        self.assertEqual([item[0] for item in records], ['open', 'close'])
        window.tray.pause_action.trigger()
        window.tray.pause_action.trigger()
        self.wait_for(lambda: window.state == 'monitoring' and not window.blocked)
        self.assertEqual(window.worker.service.identifier, identifier)
        window.tray.end_action.trigger()
        self.wait_for(lambda: window.state == 'idle')
        self.assertFalse(window.isVisible())
        self.assertEqual([item[0] for item in records], ['open', 'close', 'open', 'close'])
        with ProductStoreContext(self.args.data_dir) as store:
            self.assertEqual(len(store.history()), 1)
            self.assertEqual(store.history()[0]['state'], 'ended')

    def test_scheduled_session_starts_and_finishes_while_hidden(self):
        with ProductStoreContext(self.args.data_dir) as store:
            store.add_task('后台短任务', utc_now() + timedelta(seconds=.5), .02)
        records = []
        window = self.open(lambda: TrackedSource(records))
        window.close()
        self.wait_for(lambda: window.state == 'monitoring')
        self.assertFalse(window.isVisible())
        self.wait_for(lambda: window.state == 'idle')
        self.assertFalse(window.isVisible())
        self.assertTrue(window.worker.isRunning())
        self.assertEqual([item[0] for item in records], ['open', 'close'])
        with ProductStoreContext(self.args.data_dir) as store:
            self.assertEqual(store.history()[0]['state'], 'ended')

    def test_explicit_exit_waits_for_release_is_responsive_and_releases_lock(self):
        records, exit_signals = [], []
        release = self.gate()
        window = self.open(lambda: TrackedSource(records, release))
        window.start.click()
        self.wait_for(lambda: window.state == 'monitoring')
        window.exit_ready.connect(lambda: exit_signals.append('ready'))
        window.close()
        window.tray.quit_action.trigger()
        self.assertTrue(window.isVisible())
        self.assertTrue(window.closing)
        self.assertFalse(window.tray.quit_action.isEnabled())
        self.assertFalse(window.send('resume'))
        window.request_exit()
        timer_fired = []
        QTimer.singleShot(30, lambda: timer_fired.append(True))
        self.wait_for(lambda: timer_fired)
        self.assertTrue(window.worker.isRunning())
        self.assertEqual(exit_signals, [])
        with self.assertRaises(ValueError):
            RuntimeLock(self.args.data_dir)
        release.set()
        self.wait_for(lambda: window.exit_completed)
        self.assertFalse(window.isVisible())
        self.assertFalse(window.tray.icon.isVisible())
        self.assertEqual(exit_signals, ['ready'])
        self.assertEqual([item[0] for item in records], ['open', 'close'])
        RuntimeLock(self.args.data_dir).close()
        with ProductStoreContext(self.args.data_dir) as store:
            self.assertEqual(store.history()[0]['state'], 'interrupted')

    def test_no_tray_close_exits_and_tray_loss_restores_hidden_window(self):
        window = self.open()
        self.assertIn('仍在托盘运行', window.lifecycle_hint.text())
        window.close()
        self.available.return_value = False
        window.check_tray()
        self.assertTrue(window.isVisible())
        self.assertIn('系统托盘不可用', window.lifecycle_hint.text())
        window.close()
        self.wait_for(lambda: window.exit_completed)

    def test_hidden_worker_failure_returns_visible_stopped_window(self):
        window = self.open()
        window.close()
        window.worker.requestInterruption()
        self.wait_for(lambda: not window.worker.isRunning() and window.isVisible())
        self.assertIn('运行已停止', window.status.text())
        self.assertFalse(window.tray.pause_action.isEnabled())
        self.assertFalse(window.tray.end_action.isEnabled())
        self.assertFalse(window.tray.icon.isVisible())
        self.assertTrue(window.exit_button.isEnabled())

    def test_exit_during_loading_cancels_start_without_creating_session(self):
        entered, release, records = self.gate(), self.gate(), []

        def prepare():
            entered.set()
            release.wait(5)
            return TrackedSource(records)

        window = self.open(prepare)
        window.start.click()
        self.wait_for(entered.is_set)
        window.close()
        self.assertFalse(window.closing)
        window.tray.quit_action.trigger()
        release.set()
        self.wait_for(lambda: window.exit_completed)
        self.assertEqual([item[0] for item in records], ['open', 'close'])
        with ProductStoreContext(self.args.data_dir) as store:
            self.assertEqual(store.history(), [])

    def test_exit_waits_for_prewarm_before_releasing_owner_lock(self):
        entered, release, closed = self.gate(), self.gate(), self.gate()

        def prepare(*_args, **_kwargs):
            entered.set()
            release.wait(5)
            return SimpleNamespace(close=closed.set)

        self.args.demo = False
        with patch('observer.desktop.prepare_camera_resources', side_effect=prepare):
            window = self.open()
            self.wait_for(entered.is_set)
            window.request_exit()
            timer_fired = []
            QTimer.singleShot(30, lambda: timer_fired.append(True))
            self.wait_for(lambda: timer_fired)
            self.assertTrue(window.worker.isRunning())
            with self.assertRaises(ValueError):
                RuntimeLock(self.args.data_dir)
            release.set()
            self.wait_for(lambda: window.exit_completed)
        self.assertTrue(closed.is_set())
        self.assertFalse(window.worker.preloader._thread.is_alive())
        RuntimeLock(self.args.data_dir).close()

    def test_model_cleanup_error_still_releases_database_and_lock(self):
        def fail_close():
            raise RuntimeError('模拟释放失败')

        self.args.demo = False
        with patch('observer.desktop.prepare_camera_resources',
                   return_value=SimpleNamespace(close=fail_close)):
            window = self.open()
            self.wait_for(lambda: window.worker.preloader.ready)
            window.request_exit()
            self.wait_for(lambda: window.exit_completed)
        self.assertIn('释放视觉模型时发生错误', window.message.text())
        RuntimeLock(self.args.data_dir).close()

    def test_runtime_gap_interrupts_session_and_releases_source(self):
        records = []
        window = self.open(lambda: TrackedSource(records))
        window.start.click()
        self.wait_for(lambda: window.state == 'monitoring')
        with patch.object(window.worker.runtime_watch, 'check', return_value='模拟睡眠恢复'):
            self.wait_for(lambda: window.state == 'idle')
        self.assertIn('模拟睡眠恢复', window.message.text())
        self.assertEqual([item[0] for item in records], ['open', 'close'])
        self.assertTrue(window.worker.isRunning())
        with ProductStoreContext(self.args.data_dir) as store:
            summary = store.summary(store.history()[0]['id'])
            self.assertEqual(summary['state'], 'interrupted')
            self.assertEqual(summary['error'], '模拟睡眠恢复')

    def test_slow_read_keeps_session_and_excludes_missing_time(self):
        entered, release = self.gate(), self.gate()

        class SlowReadSource(DemoSource):
            def read(self, at):
                if not entered.is_set():
                    entered.set()
                    if not release.wait(5):
                        raise RuntimeError('test did not release the simulated read')
                return super().read(at)

        window = self.open(SlowReadSource)
        window.worker.runtime_watch.max_gap = .2
        window.start.click()
        self.wait_for(entered.is_set)
        identifier = window.worker.service.identifier
        time.sleep(.45)
        release.set()
        self.wait_for(lambda: '已跳过这段时间并继续计时' in window.message.text())
        self.assertEqual(window.worker.service.identifier, identifier)
        self.assertEqual(window.state, 'monitoring')
        window.end.click()
        self.wait_for(lambda: window.state == 'idle')
        with ProductStoreContext(self.args.data_dir) as store:
            self.assertEqual(store.history()[0]['state'], 'ended')
            intervals = store.db.execute(
                'SELECT started,ended FROM session_work WHERE session_id=? ORDER BY sequence',
                (identifier,),
            ).fetchall()
        self.assertEqual(len(intervals), 2)
        self.assertGreater(intervals[1]['started'], intervals[0]['ended'])
        log = self.args.data_dir / 'logs' / 'runtime-gaps.jsonl'
        entry = json.loads(log.read_text(encoding='utf-8').splitlines()[0])
        self.assertEqual(entry['handling'], 'resume')
        self.assertEqual(entry['operation'], '摄像头读取与视觉推理')
        self.assertGreaterEqual(entry['unobserved_seconds'], .4)

    def test_runtime_gap_keeps_manually_paused_session_paused(self):
        window = self.open()
        window.start.click()
        self.wait_for(lambda: window.state == 'monitoring')
        window.pause.click()
        self.wait_for(lambda: window.state == 'paused')
        identifier = window.worker.service.identifier
        with patch.object(window.worker.runtime_watch, 'check', return_value='模拟睡眠恢复') as check:
            self.wait_for(lambda: check.call_count >= 2)
        self.assertEqual(window.state, 'paused')
        self.assertEqual(window.worker.service.identifier, identifier)

    def test_second_window_does_not_open_database_or_camera_while_first_hidden(self):
        first = self.open()
        first.start.click()
        self.wait_for(lambda: first.state == 'monitoring')
        first.close()
        with patch('observer.desktop.ProductStore') as open_store:
            second = ObserverWindow(self.args)
            self.windows.append(second)
            second.show()
            self.wait_for(lambda: not second.worker.isRunning())
            open_store.assert_not_called()
        self.assertIn('已有 Observer', second.message.text())
        self.assertEqual(first.state, 'monitoring')
        self.assertTrue(first.worker.isRunning())

    def test_launch_quits_event_loop_after_hidden_explicit_exit(self):
        # Real event loop in a fresh process catches hidden-window shutdown leaks.
        code = '''
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch
from PySide6.QtCore import QTimer
from observer.desktop import ObserverWindow, launch
from observer.desktop_tray import DesktopTray
from observer.product import RuntimeLock
original = ObserverWindow.__init__
def initialize(self, *args, **kwargs):
    original(self, *args, **kwargs)
    def hide_then_exit():
        self.close()
        assert not self.isVisible()
        assert self.worker.isRunning()
        self.tray.quit_action.trigger()
    QTimer.singleShot(300, hide_then_exit)
args = SimpleNamespace(data_dir=Path(sys.argv[1]), demo=True, models_dir=Path('models'), profile=None)
with patch.object(ObserverWindow, '__init__', initialize), patch.object(DesktopTray, 'available', new_callable=PropertyMock, return_value=True):
    assert launch(args) == 0
RuntimeLock(args.data_dir).close()
print('clean-exit')
'''
        result = subprocess.run([sys.executable, '-c', code, str(self.args.data_dir)],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('clean-exit', result.stdout)


class ProductStoreContext:
    def __init__(self, root):
        self.root = root

    def __enter__(self):
        self.store = ProductStore(self.root)
        return self.store

    def __exit__(self, *_):
        self.store.close()


class PreloaderShutdownTests(unittest.TestCase):
    def test_close_wait_joins_inflight_prepare_without_delivering_callback(self):
        entered, release, closed, exited = (threading.Event() for _ in range(4))
        callbacks = []

        def prepare():
            entered.set()
            release.wait(5)
            return SimpleNamespace(close=closed.set)

        preloader = BackgroundCameraPreloader(prepare)
        preloader.start(callbacks.append)
        self.assertTrue(entered.wait(2))

        def shutdown():
            preloader.close(wait=True)
            exited.set()

        thread = threading.Thread(target=shutdown)
        thread.start()
        try:
            self.assertFalse(exited.wait(.05))
            self.assertFalse(preloader.start())
        finally:
            release.set()
            thread.join(5)
        self.assertTrue(exited.is_set())
        self.assertTrue(closed.is_set())
        self.assertFalse(preloader._thread.is_alive())
        self.assertEqual(callbacks, [])
