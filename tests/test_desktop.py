import importlib.util
import json
import os
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HAS_QT = importlib.util.find_spec('PySide6') is not None
if HAS_QT:
    from PySide6.QtWidgets import QApplication, QScrollArea
    from observer.desktop import ObserverWindow, live_other_event_text, user_error_message
from observer.product import ProductStore, SessionService, utc_now
from observer.product_cli import DemoSource


@unittest.skipUnless(HAS_QT, 'desktop optional dependency not installed')
class DesktopTests(unittest.TestCase):
    def test_natural_deadline_plays_completion_once_and_persists_session(self):
        self.open()
        with patch('observer.desktop.play_completion_sound') as sound:
            self.window.worker.submit('start', '完成提示测试', .01)
            self.wait_for(lambda: bool(self.window.presented_completions), 8)
            self.wait_for(lambda: self.window.state == 'idle')
            sound.assert_called_once()
            identifier = next(iter(self.window.presented_completions))
            self.window.show_session_completed(identifier, '完成提示测试')
            sound.assert_called_once()
            self.assertIn('计时结束', self.window.alert_banner.text())
        store = ProductStore(self.args.data_dir)
        try:
            self.assertEqual(store.history()[0]['state'], 'ended')
        finally:
            store.close()

    def test_manual_end_does_not_play_completion(self):
        self.open()
        with patch('observer.desktop.play_completion_sound') as sound:
            self.window.start.click()
            self.wait_for(lambda: self.window.state == 'monitoring')
            self.window.end.click()
            self.wait_for(lambda: self.window.state == 'idle')
            sound.assert_not_called()

    def test_completion_sound_is_valid_three_note_wave(self):
        import wave
        from io import BytesIO
        from observer.desktop import completion_wave
        with wave.open(BytesIO(completion_wave())) as stream:
            self.assertEqual(stream.getnchannels(), 1)
            self.assertEqual(stream.getsampwidth(), 2)
            self.assertAlmostEqual(stream.getnframes() / stream.getframerate(), .525, places=3)

    def test_alarm_and_completion_waves_have_distinct_pitches(self):
        import wave
        from io import BytesIO
        import numpy as np
        from observer.desktop import completion_wave, _reminder_wave

        def pitch(raw, start):
            with wave.open(BytesIO(raw)) as stream:
                rate = stream.getframerate()
                samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2')
            chunk = samples[round(start*rate):round((start+.05)*rate)]
            spectrum = np.abs(np.fft.rfft(chunk*np.hanning(len(chunk))))
            return float(np.fft.rfftfreq(len(chunk), 1/rate)[np.argmax(spectrum)])

        for start in (.035, .235):
            self.assertAlmostEqual(pitch(_reminder_wave(False), start), 880, delta=12)
        for start, expected in ((.035, 523), (.210, 659), (.385, 784)):
            self.assertAlmostEqual(pitch(completion_wave(), start), expected, delta=12)

    def test_version_update_page_is_minimal_and_checks_without_credentials(self):
        from observer import __version__
        from observer.updates import Release
        from PySide6.QtWidgets import QPushButton, QLineEdit, QCheckBox, QTextBrowser
        self.open()
        page = self.window.update_page
        self.assertIn(__version__, self.window.windowTitle())
        self.assertEqual([button.text() for button in page.findChildren(QPushButton)],
                         ['检查更新', 'GitHub 发布页'])
        self.assertEqual(page.findChildren(QLineEdit), [])
        self.assertEqual(page.findChildren(QCheckBox), [])
        self.assertEqual(page.findChildren(QTextBrowser), [])
        release = Release('v0.2.0', '<script>untrusted notes</script>', '', 1, 'a.zip', 100, '', None)
        with patch('observer.update_page.GitHubUpdates') as client:
            client.return_value.check.return_value = release
            page.check.click()
            self.wait_for(lambda: not page.busy() and page.check.isEnabled())
            client.assert_called_once_with()
            self.assertEqual(client.return_value.check.call_args.args[:2], (page.current, True))
        self.assertIn('v0.2.0', page.status.text())
        self.assertNotIn('script', page.status.text())
        self.assertTrue(self.window.start.isEnabled())

    def test_update_access_error_remains_retryable(self):
        from observer.updates import UpdateError
        self.open()
        page = self.window.update_page
        with patch('observer.update_page.GitHubUpdates.check', side_effect=UpdateError('无法访问 GitHub')):
            page.check.click()
            self.wait_for(lambda: not page.busy() and page.check.isEnabled())
        self.assertIn('无法访问', page.status.text())

    def test_update_check_has_no_upgrade_to_offer(self):
        self.open()
        page = self.window.update_page
        with patch('observer.update_page.GitHubUpdates.check', return_value=None):
            page.check.click()
            self.wait_for(lambda: not page.busy() and page.check.isEnabled())
        self.assertEqual(page.status.text(), '暂无更新。')

    def test_update_release_button_opens_public_release_page(self):
        from observer.updates import RELEASES_URL
        self.open()
        with patch('observer.update_page.QDesktopServices.openUrl') as open_url:
            self.window.update_page.web.click()
        self.assertEqual(open_url.call_args.args[0].toString(), RELEASES_URL)

    def test_update_download_click_switches_only_after_worker_stops(self):
        from observer.updates import Release
        self.open(); page = self.window.update_page
        release = Release('v0.2.0', '', '', 1, 'a.zip', 100, '', None)
        downloaded = Path(self.temp.name)/'new-app'
        def completed(app, args):
            self.assertFalse(self.window.worker.isRunning())
            self.assertTrue(self.window.exit_completed)
        with patch('observer.update_page.GitHubUpdates.check', return_value=release), \
                patch('observer.update_page.GitHubUpdates.stage', return_value=downloaded) as stage, \
                patch('observer.update_page.is_frozen', return_value=True), \
                patch('observer.update_page.finish_upgrade', side_effect=completed) as finish:
            page.check.click(); self.wait_for(lambda: not page.busy())
            self.assertEqual(page.check.text(), '下载并更新')
            page.check.click(); self.wait_for(lambda: self.window.exit_completed, 10)
            stage.assert_called_once(); finish.assert_called_once_with(downloaded, self.args)

    def test_update_waits_for_current_session_and_switches_after_end(self):
        from observer.updates import Release
        self.open(); page = self.window.update_page
        self.window.start.click(); self.wait_for(lambda:self.window.state == 'monitoring')
        page.release = Release('v0.2.0', '', '', 1, 'a.zip', 100, '', None); page.refresh()
        with patch('observer.update_page.GitHubUpdates.stage', return_value=Path(self.temp.name)/'new'), \
                patch('observer.update_page.is_frozen', return_value=True), \
                patch('observer.update_page.finish_upgrade') as finish:
            page.check.click(); self.wait_for(lambda: not page.busy())
            finish.assert_not_called(); self.assertFalse(self.window.closing)
            self.assertIn('计时结束', page.status.text())
            self.window.end.click(); self.wait_for(lambda: self.window.exit_completed, 10)
            finish.assert_called_once()

    def test_update_pending_switch_can_be_cancelled_without_losing_download(self):
        from observer.updates import Release
        self.open(); page = self.window.update_page
        self.window.start.click(); self.wait_for(lambda:self.window.state == 'monitoring')
        page.release = Release('v0.2.0', '', '', 1, 'a.zip', 100, '', None); page.refresh()
        with patch('observer.update_page.GitHubUpdates.stage', return_value=Path(self.temp.name)/'new'), \
                patch('observer.update_page.is_frozen', return_value=True), \
                patch('observer.update_page.finish_upgrade') as finish:
            page.check.click(); self.wait_for(lambda: not page.busy())
            page.check.click(); self.assertFalse(page.auto_switch)
            self.assertIsNotNone(page.staged_app)
            self.window.end.click(); self.wait_for(lambda:self.window.state == 'idle')
            finish.assert_not_called(); self.assertFalse(self.window.closing)
            self.assertEqual(page.check.text(), '重启更新')

    def test_download_failure_keeps_session_open_and_download_retryable(self):
        from observer.updates import Release, UpdateError
        self.open(); page = self.window.update_page
        page.release = Release('v0.2.0', '', '', 1, 'a.zip', 100, '', None); page.refresh()
        with patch('observer.update_page.GitHubUpdates.stage', side_effect=UpdateError('校验失败')), \
                patch('observer.update_page.finish_upgrade') as finish:
            page.check.click(); self.wait_for(lambda: not page.busy())
            self.assertIn('校验失败',page.status.text())
            self.assertEqual(page.check.text(), '下载并更新')
            self.assertFalse(self.window.closing); finish.assert_not_called()

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
            self.window.request_exit()
            self.wait_for(lambda: not self.window.worker.isRunning(), 10)
            self.app.processEvents()
            self.window.deleteLater()
            self.app.processEvents()
        self.temp.cleanup()

    def test_buttons_pause_resume_end_and_history(self):
        self.open()
        self.assertIn('#2563eb', self.window.start.styleSheet())
        self.assertIn('#e2e8f0', self.window.pause.styleSheet())
        self.assertIn('#e2e8f0', self.window.end.styleSheet())
        self.window.title.setText('窗口测试')
        self.window.start.click()
        self.wait_for(lambda: self.window.state == 'monitoring')
        self.assertIn('#e2e8f0', self.window.start.styleSheet())
        self.assertIn('#2563eb', self.window.pause.styleSheet())
        self.assertIn('#2563eb', self.window.end.styleSheet())
        self.window.pause.click()
        self.wait_for(lambda: self.window.state == 'paused')
        self.assertEqual(self.window.pause.text(), '继续')
        paused_countdown = self.window.status.text().splitlines()[1]
        frozen_until = time.monotonic() + 1.1
        while time.monotonic() < frozen_until:
            self.app.processEvents()
            time.sleep(.01)
        self.assertEqual(self.window.status.text().splitlines()[1], paused_countdown)
        self.assertEqual(self.window.state, 'paused')
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

    def test_home_duration_unit_is_outside_editable_spinbox(self):
        self.open()
        self.assertEqual(self.window.minutes.suffix(), '')
        self.assertEqual(self.window.minutes.lineEdit().text(), '25')
        self.assertEqual(self.window.duration_unit.text(), '分钟')

    def test_focus_form_fields_have_readable_minimum_height(self):
        self.open()
        for widget in (self.window.title, self.window.minutes,
                       self.window.scheduled, self.window.when):
            self.assertGreaterEqual(widget.minimumHeight(), 40)
            self.assertGreaterEqual(widget.height(), 40)
        for widget in (self.window.title, self.window.minutes, self.window.when):
            self.assertGreaterEqual(widget.font().pixelSize(), 15)
        self.assertIn(
            'QLineEdit#focusTaskTitle,QSpinBox#focusDuration,'
            'QDateTimeEdit#focusStartTime {font-size:15px; padding:2px 8px;}',
            self.window.styleSheet())

    def test_focus_page_keeps_active_text_readable_when_window_is_short(self):
        self.open()
        default_height = self.window.height()
        available_height = self.app.primaryScreen().availableGeometry().height()
        self.window.status.setText(
            '已暂停 · 专注\n剩余 24:35 · 100 分\n头眼：不可判断\n其他事件：已暂停')
        self.window.resize(620, 650)
        self.app.processEvents()

        def vertical_bounds(widget):
            return (widget.mapTo(self.window, widget.rect().topLeft()).y(),
                    widget.mapTo(self.window, widget.rect().bottomLeft()).y() + 1)

        for upper, lower in ((self.window.title, self.window.minutes),
                             (self.window.minutes, self.window.when),
                             (self.window.when, self.window.status),
                             (self.window.status, self.window.start)):
            self.assertLessEqual(vertical_bounds(upper)[1], vertical_bounds(lower)[0])
        self.assertGreaterEqual(
            self.window.status.height(),
            self.window.status.fontMetrics().lineSpacing() * 4 + 20,
        )
        if available_height >= 952:
            self.assertGreaterEqual(self.window.minimumHeight(), 920)
            self.assertNotIsInstance(self.window.tabs.widget(0), QScrollArea)
        else:
            focus_scroll = self.window.tabs.widget(0)
            self.assertIsInstance(focus_scroll, QScrollArea)
            self.assertGreater(focus_scroll.verticalScrollBar().maximum(), 0)
        self.window.focus_events.setPlainText('事件\n' * 80)
        self.app.processEvents()
        self.assertGreater(self.window.focus_events.verticalScrollBar().maximum(), 0)
        self.assertGreaterEqual(default_height, min(940, max(650, available_height - 32)))

    def test_window_starts_near_primary_screen_center(self):
        self.open()
        screen_center = self.app.primaryScreen().availableGeometry().center()
        window_center = self.window.frameGeometry().center()
        self.assertLessEqual(abs(window_center.x() - screen_center.x()), 32)
        self.assertLessEqual(abs(window_center.y() - screen_center.y()), 32)

    def test_start_immediately_shows_busy_and_ignores_stale_snapshots(self):
        calls = 0

        def slow_source():
            nonlocal calls
            calls += 1
            time.sleep(.3)
            return DemoSource()

        self.open(slow_source)
        self.window.start.click()
        self.assertEqual(self.window.start.text(), '启动中…')
        self.assertFalse(self.window.start.isEnabled())
        self.assertFalse(self.window.title.isEnabled())
        deadline = time.monotonic() + .15
        while time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.assertEqual(self.window.start.text(), '启动中…')
        self.assertFalse(self.window.start.isEnabled())
        self.window.start.click()
        self.wait_for(lambda: self.window.state == 'monitoring')
        self.assertEqual(calls, 1)
        self.assertEqual(self.window.start.text(), '开始')
        self.assertIn('准备耗时', self.window.message.text())
        self.window.end.click()
        self.wait_for(lambda: self.window.state == 'idle')

    def test_session_identity_records_product_startup_duration(self):
        def measured_source():
            time.sleep(.05)
            return DemoSource()

        self.open(measured_source)
        self.window.start.click()
        self.wait_for(lambda: self.window.state == 'monitoring')
        self.window.end.click()
        self.wait_for(lambda: self.window.state == 'idle')
        store = ProductStore(self.args.data_dir)
        try:
            row = store.summary(store.history()[0]['id'])
            identity = json.loads(row['identity'])
        finally:
            store.close()
        self.assertGreaterEqual(identity['product_startup_seconds'], .04)

    def test_real_start_writes_jsonl_diagnostic_without_task_title(self):
        self.args.demo = False
        self.open(DemoSource)
        self.window.title.setText('不写入诊断的任务标题')
        self.window.start.click()
        self.wait_for(lambda: self.window.state == 'monitoring')
        path = Path(self.args.data_dir) / 'logs' / 'startup.jsonl'
        content = path.read_text(encoding='utf-8')
        record = json.loads(content.splitlines()[-1])
        self.assertEqual(record['status'], 'success')
        self.assertGreaterEqual(record['elapsed_seconds'], 0)
        self.assertTrue(record['session_id'])
        self.assertNotIn('不写入诊断的任务标题', content)

    def test_startup_log_rotates_at_bounded_size(self):
        from observer.desktop import append_startup_log

        with patch('observer.desktop.STARTUP_LOG_MAX_BYTES', 80):
            append_startup_log(self.args.data_dir, {'attempt': 1, 'padding': 'x'*40})
            path = append_startup_log(
                self.args.data_dir, {'attempt': 2, 'padding': 'x'*40})
        backup = path.with_name('startup.previous.jsonl')
        self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['attempt'], 2)
        self.assertEqual(json.loads(backup.read_text(encoding='utf-8'))['attempt'], 1)

    def test_history_page_shows_local_status_and_deletes_selected_session(self):
        self.open()
        self.assertIn('0 条记录', self.window.data_status.text())
        self.window.images.click()
        self.wait_for(lambda: '截图开启' in self.window.data_status.text())
        self.window.start.click()
        self.wait_for(lambda: self.window.state == 'monitoring')
        self.window.end.click()
        self.wait_for(lambda: self.window.state == 'idle')
        self.wait_for(lambda: '1 条记录' in self.window.data_status.text())
        self.window.history.setCurrentRow(0)
        self.window.delete_history.click()
        self.wait_for(lambda: self.window.history.count() == 0)
        self.wait_for(lambda: '0 条记录' in self.window.data_status.text())
        self.assertEqual(self.window.details.toPlainText(), '')

    def test_corrupt_history_metadata_is_visible_without_breaking_window(self):
        store = ProductStore(self.args.data_dir)
        try:
            service = SessionService(store)
            identifier = service.start('损坏记录', 1, 0, utc_now(), {'source': 'test'})
            service.end(1, utc_now())
            with store.db:
                store.db.execute(
                    "UPDATE sessions SET identity=?,resource_summary=? WHERE id=?",
                    ('not-json', '[]', identifier),
                )
        finally:
            store.close()

        self.open()
        self.wait_for(lambda: self.window.history.count() == 1)
        self.assertIn('身份异常', self.window.history.item(0).text())
        self.window.worker.submit('show', identifier)
        self.wait_for(lambda: '数据警告：会话身份记录损坏；资源汇总损坏'
                      in self.window.details.toPlainText())

    def test_alert_is_visible_and_keeps_plain_notices_plain(self):
        self.open()
        with patch('observer.desktop.play_reminder_sound') as sound:
            self.window.show_notice('提醒：离开屏幕')
            sound.assert_called_once_with('phone_held')
        self.assertIn('提醒：', self.window.message.text())
        self.assertIn('font-weight', self.window.message.styleSheet())
        self.assertIn('已记录', self.window.alert_banner.text())
        self.assertFalse(self.window.alert_banner.isHidden())
        self.window.show_notice('已开始')
        self.assertEqual(self.window.message.styleSheet(), '')
        self.assertIn('已记录', self.window.alert_banner.text())

    def test_current_event_image_link_opens_and_closes_right_preview(self):
        from PySide6.QtGui import QColor, QPixmap

        self.open()
        path = Path(self.temp.name) / 'event-preview.png'
        pixmap = QPixmap(80, 60)
        pixmap.fill(QColor('#2563eb'))
        self.assertTrue(pixmap.save(str(path)))
        result = {'id': 'session-1', 'title': '图片检查',
                  'score': 95, 'deducted': 5, 'coverage': .9}
        events = [{
            'event_id': 'event-1', 'rule_id': 'posture_deviation',
            'offset': 25.5, 'resolved_offset': 31.5, 'resolved_reason': 'recovered',
            'points': 0., 'delivery_state': 'delivered',
            'image_path': str(path), 'image_status': '查看',
            'image_sha256': 'abc123',
        }]
        self.window.update_focus_events(result, events, active=False)
        text = self.window.focus_events.toPlainText()
        self.assertIn('总分：95 分', text)
        self.assertIn('扣分：5', text)
        self.assertIn('覆盖：90%', text)
        self.assertIn('查看', text)
        self.assertNotIn('查看图片', text)
        initial_width = self.window.width()

        self.window.open_event_image('evidence:event-1')

        self.assertFalse(self.window.preview_panel.isHidden())
        self.assertIsNotNone(self.window.preview_image.pixmap())
        self.assertFalse(self.window.preview_image.pixmap().isNull())
        self.assertGreaterEqual(self.window.width(), initial_width)
        self.assertIn('姿态', self.window.preview_title.text())
        self.assertIn('abc123', self.window.preview_meta.text())
        self.window.close_event_preview()
        self.assertTrue(self.window.preview_panel.isHidden())

    def test_event_refresh_preserves_scroll_during_evidence_and_duration_updates(self):
        self.open()
        self.window.worker.snapshot.disconnect(self.window.update_view)
        result = {'id': 'session-scroll', 'title': '事件滚动检查'}
        events = [{
            'event_id': f'event-{i}', 'rule_id': 'focus_violation',
            'offset': i * 20., 'resolved_offset': None,
            'points': 0., 'delivery_state': 'delivered',
            'image_path': None, 'image_status': '未保存图片',
            'signals': ['phone_held'], 'violation_seconds': 12.,
        } for i in range(60)]
        self.window.update_focus_events(result, events, active=True)
        self.app.processEvents()
        scroll = self.window.focus_events.verticalScrollBar()
        self.assertGreater(scroll.maximum(), 0)
        for at_bottom in (False, True):
            with self.subTest(at_bottom=at_bottom):
                position = scroll.maximum() if at_bottom else scroll.maximum() // 2
                scroll.setValue(position)
                events[-1]['violation_seconds'] += 2.
                events[-1]['images'] = [{'id': f'sample-{at_bottom}', 'sha256': 'abc'}]
                self.window.update_focus_events(result, events, active=True)
                self.app.processEvents()
                self.assertEqual(scroll.value(), position)
                events.append({**events[-1], 'event_id': f'new-{at_bottom}',
                               'offset': 2000. + len(events)})
                self.window.update_focus_events(result, events, active=True)
                self.app.processEvents()
                self.assertEqual(scroll.value(), position)
        self.window.update_focus_events({**result, 'id': 'another-session'}, events, active=True)
        self.app.processEvents()
        self.assertEqual(scroll.value(), 0)

    def test_reminder_reports_delivery_success_or_failure_to_worker(self):
        self.open()
        with patch.object(self.window.worker, 'submit') as submit:
            with patch('observer.desktop.play_reminder_sound'):
                self.window.show_reminder(
                    'alert-ok', 'posture_deviation', '提醒：姿态（软提示）')
            submit.assert_called_once_with('notification', 'alert-ok', True, None)
        with patch.object(self.window.worker, 'submit') as submit:
            with patch('observer.desktop.play_reminder_sound', side_effect=RuntimeError('无声音设备')):
                self.window.show_reminder(
                    'alert-failed', 'posture_deviation', '提醒：姿态（软提示）')
            submit.assert_called_once_with('notification', 'alert-failed', False, '无声音设备')
            self.assertIn('呈现失败', self.window.message.text())

    def test_windows_reminder_plays_memory_wave_before_returning(self):
        import sys
        from observer.desktop import play_reminder_sound

        play_sound = Mock()
        winsound_stub = SimpleNamespace(
            SND_MEMORY=0x4, SND_NODEFAULT=0x2, PlaySound=play_sound)
        with patch('observer.desktop.os.name', 'nt'), \
             patch.dict(sys.modules, {'winsound': winsound_stub}):
            play_reminder_sound('focus_violation')

        play_sound.assert_called_once()
        sound, flags = play_sound.call_args.args
        self.assertTrue(sound.startswith(b'RIFF'))
        self.assertEqual(flags, winsound_stub.SND_MEMORY | winsound_stub.SND_NODEFAULT)

    def test_windows_reminder_falls_back_when_memory_playback_fails(self):
        import sys
        from observer.desktop import play_reminder_sound

        winsound_stub = SimpleNamespace(
            SND_MEMORY=0x4, SND_NODEFAULT=0x2,
            PlaySound=Mock(side_effect=RuntimeError('audio device unavailable')),
            Beep=Mock())
        for rule_id, double in (('focus_violation', True), ('posture_deviation', False)):
            winsound_stub.Beep.reset_mock()
            with self.subTest(rule_id=rule_id), \
                 patch('observer.desktop.os.name', 'nt'), \
                 patch.dict(sys.modules, {'winsound': winsound_stub}), \
                 patch('observer.desktop.QApplication.beep') as beep, \
                 patch('observer.desktop.QTimer.singleShot') as delayed, \
                 patch('observer.desktop.time.sleep'):
                play_reminder_sound(rule_id)
                self.assertEqual(winsound_stub.Beep.call_args_list,
                                 [unittest.mock.call(880, 120)]*2 if double
                                 else [unittest.mock.call(660, 120)])
                beep.assert_not_called()
                delayed.assert_not_called()

    def test_fallback_alarm_then_completion_has_distinct_pitches_and_no_deferred_beeps(self):
        import sys
        from observer.desktop import play_reminder_sound, play_completion_sound
        native = SimpleNamespace(SND_MEMORY=4, SND_NODEFAULT=2,
            PlaySound=Mock(side_effect=RuntimeError('WAV unavailable')), Beep=Mock())
        with patch('observer.desktop.os.name', 'nt'), patch.dict(sys.modules, {'winsound': native}), \
             patch('observer.desktop.time.sleep'), patch('observer.desktop.QTimer.singleShot') as delayed, \
             patch('observer.desktop.QApplication.beep') as generic:
            play_reminder_sound('phone_held')
            play_completion_sound()
        self.assertEqual(native.Beep.call_args_list, [unittest.mock.call(f, ms) for f, ms in
            [(880, 120), (880, 120), (523, 140), (659, 140), (784, 140)]])
        delayed.assert_not_called()
        generic.assert_not_called()

    def test_audio_failure_is_reported_instead_of_ambiguous_generic_beep(self):
        import sys
        from observer.desktop import play_completion_sound
        native = SimpleNamespace(SND_MEMORY=4, SND_NODEFAULT=2,
            PlaySound=Mock(side_effect=RuntimeError('WAV unavailable')),
            Beep=Mock(side_effect=RuntimeError('no sound device')))
        with patch('observer.desktop.os.name', 'nt'), patch.dict(sys.modules, {'winsound': native}), \
             patch('observer.desktop.QApplication.beep') as generic:
            with self.assertRaisesRegex(RuntimeError, 'no sound device'):
                play_completion_sound()
        generic.assert_not_called()

    def test_completion_and_reminder_use_separate_sound_routes(self):
        self.open()
        with patch('observer.desktop.play_reminder_sound') as alarm, \
             patch('observer.desktop.play_completion_sound') as complete:
            self.window.show_reminder('routing-alert', 'phone_held', '提醒：可见手机')
            alarm.assert_called_once_with('phone_held')
            complete.assert_not_called()
            self.window.show_session_completed('routing-session', '完成')
            alarm.assert_called_once_with('phone_held')
            complete.assert_called_once_with()

    def test_windows_violation_has_two_tones_and_posture_has_one(self):
        import sys
        import wave
        from io import BytesIO
        from observer.desktop import play_reminder_sound

        for rule_id, expected in (('phone_held', 2), ('away', 2), ('posture_deviation', 1)):
            with self.subTest(rule_id=rule_id):
                play_sound = Mock()
                stub = SimpleNamespace(SND_MEMORY=4, SND_NODEFAULT=2, PlaySound=play_sound)
                with patch('observer.desktop.os.name', 'nt'), patch.dict(sys.modules, {'winsound': stub}):
                    play_reminder_sound(rule_id)
                play_sound.assert_called_once()
                with wave.open(BytesIO(play_sound.call_args.args[0]), 'rb') as stream:
                    self.assertEqual((stream.getnchannels(), stream.getsampwidth()), (1, 2))
                    chunk_frames = round(stream.getframerate() * .02)
                    audible = []
                    while frames := stream.readframes(chunk_frames):
                        audible.append(any(frames))
                    starts = sum(active and (index == 0 or not audible[index-1])
                                 for index, active in enumerate(audible))
                    self.assertEqual(starts, expected)
                    self.assertLessEqual(stream.getnframes()/stream.getframerate(), .32)

    def test_health_reminder_is_quiet_short_and_does_not_flash_window(self):
        import wave
        from io import BytesIO
        from observer.desktop import _reminder_wave

        def properties(raw):
            with wave.open(BytesIO(raw), 'rb') as stream:
                frames = stream.readframes(stream.getnframes())
                samples = [abs(int.from_bytes(frames[index:index+2], 'little', signed=True))
                           for index in range(0, len(frames), 2)]
                return stream.getnframes()/stream.getframerate(), max(samples)

        soft_duration, soft_peak = properties(_reminder_wave(True))
        hard_duration, hard_peak = properties(_reminder_wave(False))
        self.assertLessEqual(soft_duration, .15)
        self.assertLess(soft_duration, hard_duration)
        self.assertGreater(soft_peak, 2500)
        self.assertLess(soft_peak, hard_peak/2)

        self.open()
        with patch('observer.desktop.play_reminder_sound') as sound, \
             patch('observer.desktop.QApplication.alert') as alert, \
             patch('observer.desktop.HEALTH_REMINDER_BANNER_MS', 20):
            self.window.show_notice('提醒：姿态（软提示）', 'posture_deviation')
        sound.assert_called_once_with('posture_deviation')
        alert.assert_not_called()
        self.assertIn('#f0fdfa', self.window.alert_banner.styleSheet())
        self.assertTrue(self.window.health_banner_timer.isActive())
        self.wait_for(self.window.alert_banner.isHidden)

    def test_hard_reminder_cancels_health_banner_auto_hide(self):
        self.open()
        with patch('observer.desktop.play_reminder_sound'), \
             patch('observer.desktop.QApplication.alert') as alert:
            self.window.show_notice('提醒：健康提示', 'posture_deviation')
            self.assertTrue(self.window.health_banner_timer.isActive())
            self.window.show_notice('提醒：可见手机', 'phone_held')
        self.assertFalse(self.window.health_banner_timer.isActive())
        self.assertFalse(self.window.alert_banner.isHidden())
        self.assertIn('#fee2e2', self.window.alert_banner.styleSheet())
        alert.assert_called_once_with(self.window, 3000)

    def test_live_attention_candidate_is_visible_before_alert(self):
        self.open()
        self.window.start.click()
        self.wait_for(lambda: self.window.state == 'monitoring')
        self.window.worker.submit('signal', 'attention_away')
        self.wait_for(lambda: '头眼：屏外候选' in self.window.status.text())

    def test_live_other_focus_disturbances_are_visible_before_alert(self):
        self.open()
        self.window.start.click()
        self.wait_for(lambda: self.window.state == 'monitoring')
        self.wait_for(lambda: '其他事件：未发现' in self.window.status.text())
        self.window.worker.submit('signal', 'phone_held')
        self.wait_for(lambda: '其他事件：发现手机' in self.window.status.text())
        self.window.worker.submit('signal', 'away')
        self.wait_for(lambda: '其他事件：离席候选' in self.window.status.text())
        self.window.worker.submit('signal', 'unknown')
        self.wait_for(lambda: '其他事件：不可判断' in self.window.status.text())
        self.assertEqual(
            live_other_event_text(
                {'phone_held': 'violation', 'away': 'violation'}, 'monitoring'),
            '发现手机 · 离席候选',
        )

    def test_summary_exposes_soft_signal_totals_without_calling_them_continuous(self):
        from observer.desktop import summary_text
        result = {'score': 100, 'mode': 'real', 'title': '测试',
                  'state': 'ended', 'coverage': 1, 'actions': [],
                  'resource_summary': {
                      'sample_count': 4,
                      'process_cpu_average_percent': 6.25,
                      'process_cpu_p95_percent': 9.75,
                      'process_memory_peak_mb': 321.4,
                      'sampling_error': None},
                  'totals': {'attention_away': {'violation': 16.189},
                             'posture_deviation': {'violation': 0},
                             'ergonomic_risk': {'violation': 4.2}}}
        text = summary_text(result)
        self.assertIn('16.2秒', text)
        self.assertIn('累计时间不等于连续触发时间', text)
        self.assertIn('旧版健康提示累计异常：4.2秒', text)
        self.assertIn('CPU平均 6.2%', text)
        self.assertIn('内存峰值 321 MB', text)

    def test_summary_exposes_notification_delivery_state(self):
        from observer.desktop import summary_text
        result = {'score': 100, 'mode': 'real', 'title': '测试',
                  'state': 'ended', 'coverage': 1, 'error': None,
                  'totals': {}, 'actions': [
                      {'kind': 'alert', 'offset': 8, 'rule_id': 'attention_away',
                       'points': 0, 'delivery_state': 'delivered'},
                      {'kind': 'alert', 'offset': 30, 'rule_id': 'ergonomic_risk',
                       'points': 0, 'delivery_state': 'failed'},
                      {'kind': 'alert', 'offset': 40, 'rule_id': 'posture_deviation',
                       'points': 0, 'delivery_state': 'unknown'}]}
        text = summary_text(result)
        self.assertIn('1次已确认', text)
        self.assertIn('1次未确认', text)
        self.assertIn('1次历史状态未知', text)

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
        self.args.demo = False
        def broken():
            raise RuntimeError('模型不可用')
        self.open(broken)
        self.window.start.click()
        self.wait_for(lambda: '模型不可用' in self.window.message.text())
        self.wait_for(lambda: self.window.start.isEnabled())
        self.assertEqual(self.window.history.count(), 0)
        path = Path(self.args.data_dir) / 'logs' / 'startup.jsonl'
        record = json.loads(path.read_text(encoding='utf-8').splitlines()[-1])
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(record['error_type'], 'RuntimeError')
        self.assertEqual(record['error'], '模型不可用')

    def test_attention_setup_diagnostics_are_localized_but_preserved_in_log(self):
        raw = ('session attention setup timed out; keep the full face visible and '
               'look at the screen centre; face_runs=385, valid_frames=0')
        self.args.demo = False

        def broken():
            raise RuntimeError(raw)

        self.open(broken)
        self.window.start.click()
        self.wait_for(lambda: '未能完成头眼基线' in self.window.message.text())
        self.assertNotIn('face_runs', self.window.message.text())
        self.assertIn('完整脸部和双眼可见', user_error_message(raw))
        path = Path(self.args.data_dir) / 'logs' / 'startup.jsonl'
        record = json.loads(path.read_text(encoding='utf-8').splitlines()[-1])
        self.assertEqual(record['error'], raw)

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
        self.args.demo = False
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
        path = Path(self.args.data_dir) / 'logs' / 'startup.jsonl'
        record = json.loads(path.read_text(encoding='utf-8').splitlines()[-1])
        self.assertEqual(record['status'], 'cancelled')

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

    def test_advanced_plan_tab_creates_updates_toggles_and_deletes(self):
        self.open()
        self.assertEqual(
            [self.window.tabs.tabText(index) for index in range(self.window.tabs.count())],
            ['专注', '历史', '高级计划', '版本与更新'])
        self.window.tabs.setCurrentIndex(2)
        self.assertTrue(self.window.advanced_tabs.isVisible())
        self.window.plan_title.setText('晨间专注')
        self.window.plan_time.setTime(self.window.plan_time.time().addSecs(3600))
        self.window.plan_recurrence.setCurrentIndex(
            self.window.plan_recurrence.findData('weekly'))
        self.window.plan_weekdays[0].setChecked(True)
        self.window.plan_weekdays[4].setChecked(True)
        self.window.plan_break_enabled.setChecked(True)
        self.window.plan_offscreen.setChecked(True)
        self.window.save_plan.click()
        self.wait_for(lambda: self.window.plans.count() == 1)
        self.wait_for(lambda: self.window.save_plan.isEnabled())

        item = self.window.plans.item(0)
        self.window.load_plan(item)
        self.assertEqual(self.window.plan_title.text(), '晨间专注')
        self.assertTrue(self.window.plan_offscreen.isChecked())
        self.assertTrue(self.window.plan_break_enabled.isChecked())
        self.window.plan_title.setText('更新后的专注')
        self.window.save_plan.click()
        self.wait_for(lambda: '更新后的专注' in self.window.plans.item(0).text())
        self.wait_for(lambda: self.window.toggle_plan.isEnabled())

        self.window.toggle_plan.click()
        self.wait_for(lambda: '停用' in self.window.plans.item(0).text())
        self.window.delete_plan.click()
        self.wait_for(lambda: self.window.plans.count() == 0)
        store = ProductStore(self.args.data_dir)
        try:
            self.assertEqual(store.plans(), [])
            self.assertEqual([task for task in store.tasks() if task['state'] == 'pending'], [])
        finally:
            store.close()

    def test_planned_break_closes_source_and_resumes_automatically(self):
        store = ProductStore(self.args.data_dir)
        try:
            task_id = store.add_task('短休息测试', utc_now() + timedelta(seconds=.2), .08)
            with store.db:
                store.db.execute(
                    "UPDATE tasks SET break_every_minutes=?,break_minutes=? WHERE id=?",
                    (.03, .01, task_id),
                )
        finally:
            store.close()
        self.window = ObserverWindow(self.args)
        self.window.show()
        self.wait_for(lambda: self.window.state == 'monitoring')
        self.wait_for(lambda: self.window.state == 'resting', 5)
        self.assertIsNone(self.window.worker.source)
        self.assertIn('休息剩余', self.window.status.text())
        self.wait_for(lambda: self.window.state == 'monitoring', 5)
        self.window.end.click()
        self.wait_for(lambda: self.window.state == 'idle')
        store = ProductStore(self.args.data_dir)
        try:
            result = store.summary(store.history()[0]['id'])
            self.assertGreaterEqual(result['break_seconds'], .5)
        finally:
            store.close()

    def test_advanced_tab_extends_only_active_session(self):
        self.open()
        self.window.start.click()
        self.wait_for(lambda: self.window.state == 'monitoring')
        self.window.extend_minutes.setValue(1)
        self.window.extend_session.click()
        self.assertEqual(self.window.extend_session.text(), '延长中…')
        self.assertFalse(self.window.extend_session.isEnabled())
        self.wait_for(lambda: '已延长 1 分钟' in self.window.extend_feedback.text())
        self.assertIn('累计延长 1 分钟', self.window.extend_feedback.text())
        self.assertFalse(self.window.extend_session.isEnabled())
        self.window.extend_session.click()
        self.window.end.click()
        self.wait_for(lambda: self.window.state == 'idle')
        store = ProductStore(self.args.data_dir)
        try:
            result = store.summary(store.history()[0]['id'])
            self.assertEqual(result['minutes'], 26)
            self.assertEqual(result['extended_minutes'], 1)
        finally:
            store.close()

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

    def test_power_transition_interrupts_active_session_and_releases_source(self):
        closed = []

        class TrackedSource(DemoSource):
            def close(self):
                closed.append(True)

        self.open(TrackedSource)
        self.window.start.click()
        self.wait_for(lambda: self.window.state == 'monitoring')
        interrupted_id = self.window.worker.service.identifier
        self.window.worker.note_power_transition('模拟系统挂起或唤醒')
        self.wait_for(lambda: self.window.state == 'idle')
        self.assertEqual(closed, [True])
        store = ProductStore(self.args.data_dir)
        try:
            summary = store.summary(interrupted_id)
            self.assertEqual(summary['state'], 'interrupted')
            self.assertEqual(summary['error'], '模拟系统挂起或唤醒')
            self.assertEqual(summary['violation_seconds'], 0)
            self.assertEqual(summary['actions'], [])
        finally:
            store.close()
        self.window.start.click()
        self.wait_for(lambda: self.window.state == 'monitoring')
        self.assertNotEqual(self.window.worker.service.identifier, interrupted_id)
        self.window.end.click()
        self.wait_for(lambda: self.window.state == 'idle')
        self.assertEqual(closed, [True, True])
        store = ProductStore(self.args.data_dir)
        try:
            self.assertEqual([row['state'] for row in store.history()],
                             ['ended', 'interrupted'])
        finally:
            store.close()

    def test_power_transition_keeps_manually_paused_session(self):
        self.open()
        self.window.start.click()
        self.wait_for(lambda: self.window.state == 'monitoring')
        self.window.pause.click()
        self.wait_for(lambda: self.window.state == 'paused')
        self.window.worker.note_power_transition('模拟系统挂起或唤醒')
        self.wait_for(lambda: self.window.worker.power_event_reason is None)
        self.assertEqual(self.window.state, 'paused')
        self.window.end.click()
        self.wait_for(lambda: self.window.state == 'idle')

    def test_unexpected_worker_failure_persists_reason_and_releases_source(self):
        closed = []

        class TrackedSource(DemoSource):
            def close(self):
                closed.append(True)

        self.open(TrackedSource)
        self.window.start.click()
        self.wait_for(lambda: self.window.state == 'monitoring')

        def fail_publish():
            raise RuntimeError('模拟工作线程异常')

        self.window.worker.publish = fail_publish
        self.wait_for(lambda: not self.window.worker.isRunning(), 10)
        self.assertEqual(closed, [True])
        store = ProductStore(self.args.data_dir)
        try:
            result = store.summary(store.history()[0]['id'])
            self.assertEqual(result['state'], 'interrupted')
            self.assertEqual(result['error'], '模拟工作线程异常')
            self.assertIsNotNone(result['resource_summary'])
        finally:
            store.close()

    def test_database_write_failure_interrupts_scoring_and_releases_source(self):
        closed = []

        class TrackedSource(DemoSource):
            def close(self):
                closed.append(True)

        self.open(TrackedSource)
        self.window.start.click()
        self.wait_for(lambda: self.window.state == 'monitoring')
        store = ProductStore(self.args.data_dir)
        try:
            with store.db:
                store.db.execute(
                    "CREATE TRIGGER reject_product_totals BEFORE INSERT ON totals "
                    "BEGIN SELECT RAISE(ABORT,'product database write failure'); END"
                )
        finally:
            store.close()
        self.wait_for(lambda: not self.window.worker.isRunning(), 10)
        self.assertEqual(closed, [True])
        store = ProductStore(self.args.data_dir)
        try:
            result = store.summary(store.history()[0]['id'])
            self.assertEqual(result['state'], 'interrupted')
            self.assertIn('product database write failure', result['error'])
            self.assertEqual(result['deducted'], 0)
            self.assertEqual(result['actions'], [])
        finally:
            store.close()

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

    def test_slow_successful_reconnect_keeps_session(self):
        from observer.product_cli import SourceStalled
        window = None

        class SlowReconnectSource(DemoSource):
            reconnected = False

            def read(self, at):
                if not self.reconnected:
                    raise SourceStalled('摄像头帧停止')
                return super().read(at)

            def reconnect(self):
                window.worker.runtime_watch.max_gap = .2
                time.sleep(.45)
                self.reconnected = True

        self.open(SlowReconnectSource)
        window = self.window
        self.window.start.click()
        self.wait_for(lambda: '摄像头已恢复' in self.window.message.text())
        store = ProductStore(self.args.data_dir)
        try:
            history = store.history()
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]['state'], 'monitoring')
            self.assertEqual(history[0]['id'], window.worker.service.identifier)
        finally:
            store.close()

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
