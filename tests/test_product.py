import contextlib
import io
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from observer.observations import Observation, SignalValue
from observer.product import NAMES, ProductStore, RuntimeLock, SessionService


class ProductTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'product'
        self.store = ProductStore(self.root)
        self.service = SessionService(self.store)
        self.now = datetime(2026, 9, 10, tzinfo=timezone.utc)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def start(self, minutes=10):
        return self.service.start('测试', minutes, 0, self.now, {'source': 'test'})

    def feed(self, at, *violations, unknown=False):
        signals = {name: SignalValue.unknown('test') if unknown else
                   SignalValue.violation() if name in violations else SignalValue.compliant()
                   for name in NAMES}
        return self.service.process(Observation(at, signals), self.now+timedelta(seconds=at))

    def test_persistent_phone_event_and_duplicate_observation(self):
        identifier = self.start()
        for at in range(16):
            self.feed(at, 'phone_held')
        self.feed(15, 'phone_held')
        for at in range(16, 20):
            self.feed(at)
        result = self.service.end(20, self.now+timedelta(seconds=20))
        self.assertEqual(result['score'], 95)
        self.assertEqual(sum(a['kind']=='penalty' for a in result['actions']), 1)
        self.assertEqual(sum(a['kind']=='resolved' for a in result['actions']), 1)
        self.store.close()
        self.store = ProductStore(self.root)
        self.assertEqual(self.store.summary(identifier)['deducted'], 5)

    def test_pause_does_not_bridge_trigger_or_inflate_coverage(self):
        identifier = self.start()
        for at in range(8):
            self.feed(at, 'phone_held')
        self.service.pause(8, self.now)
        self.feed(30, 'phone_held')
        self.service.resume(50, self.now)
        for at in range(50, 58):
            self.feed(at, 'phone_held')
        result = self.service.end(58, self.now)
        self.assertEqual(result['deducted'], 0)
        self.assertEqual(result['totals']['phone_held']['eligible'], 16)
        self.assertEqual(self.store.summary(identifier)['state'], 'ended')

    def test_source_gap_does_not_bridge_a_violation(self):
        self.start()
        for at in range(8):
            self.feed(at, 'phone_held')
        self.service.source_gap(8, self.now+timedelta(seconds=8), 'camera stalled')
        for at in range(20, 28):
            self.feed(at, 'phone_held')
        result = self.service.end(28, self.now+timedelta(seconds=28))
        self.assertEqual(result['deducted'], 0)
        self.assertEqual(result['totals']['phone_held']['eligible'], 28)
        self.assertEqual(result['totals']['phone_held']['valid'], 14)

    def test_pause_preserves_accumulated_penalty_cap(self):
        self.start()
        for cycle in range(6):
            start = cycle * 30
            if cycle:
                self.service.resume(start, self.now)
            for at in range(start, start+12):
                self.feed(at, 'phone_held')
            self.service.pause(start+12, self.now)
        self.assertEqual(self.service.end(200, self.now)['deducted'], 25)

    def test_unknown_and_missing_time_cannot_produce_valid_score(self):
        self.start()
        self.feed(0, 'phone_held')
        self.feed(40, 'phone_held')
        self.feed(41, unknown=True)
        result = self.service.end(42, self.now)
        self.assertIsNone(result['score'])
        self.assertEqual(result['deducted'], 0)
        self.assertEqual(result['totals']['away']['eligible'], 42)

    def test_soft_signals_never_deduct(self):
        self.start()
        for at in range(181):
            self.feed(at, 'attention_away', 'posture_deviation', 'ergonomic_risk')
        result = self.service.end(181, self.now)
        self.assertEqual(result['score'], 100)
        self.assertEqual(sum(a['kind']=='alert' for a in result['actions']), 3)
        self.assertFalse(any(a['kind']=='penalty' for a in result['actions']))

    def test_away_suppresses_unattended_phone(self):
        self.start()
        for at in range(25):
            self.feed(at, 'away', 'phone_held')
        result = self.service.end(25, self.now)
        self.assertEqual(result['deducted'], 3)
        self.assertTrue(all(a['rule_id']=='away' for a in result['actions']))

    def test_abandoned_session_keeps_ledger_and_stops_at_checkpoint(self):
        identifier = self.start()
        for at in range(12):
            self.feed(at, 'phone_held')
        self.store.interrupt_abandoned()
        summary = self.store.summary(identifier)
        self.assertEqual(summary['state'], 'interrupted')
        self.assertEqual(summary['ended'], summary['updated'])
        self.assertEqual(summary['deducted'], 5)

    def test_task_is_claimed_once_and_conflicts_are_explicit(self):
        task_id = self.store.add_task('定时', self.now, 1)
        task = self.service.due(self.now)
        self.service.start(task['title'], task['minutes'], 0, self.now, {}, task_id)
        self.assertIsNone(self.service.due(self.now))
        other = self.store.add_task('冲突', self.now, 1)
        self.assertIsNone(self.service.due(self.now))
        conflicted = next(t for t in self.store.tasks() if t['id']==other)
        self.assertEqual(conflicted['state'], 'conflict')
        self.assertIn('已有会话', conflicted['error'])
        self.service.end(1, self.now)
        self.assertEqual(self.store.tasks()[0]['state'], 'ended')

    def test_task_missed_cancel_and_invalid_inputs(self):
        self.store.add_task('迟到', self.now-timedelta(seconds=61), 1)
        self.assertIsNone(self.service.due(self.now))
        self.assertEqual(self.store.tasks()[0]['state'], 'missed')
        self.assertIn('超过60秒', self.store.tasks()[0]['error'])
        identifier = self.store.add_task('取消', self.now, 1)
        self.store.cancel_task(identifier)
        self.assertIsNone(self.service.due(self.now))
        for minutes in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.store.add_task('错误', self.now, minutes)
        with self.assertRaises(ValueError):
            self.store.add_task('错误', self.now.replace(tzinfo=None), 1)

    def test_restart_expiry_keeps_tasks_within_sixty_second_grace(self):
        expired = self.store.add_task('超过宽限', self.now-timedelta(seconds=60.1), 1)
        runnable = self.store.add_task('宽限内', self.now-timedelta(seconds=59.9), 1)
        self.assertEqual(self.store.expire_pending_tasks(self.now), 1)
        states = {task['id']: task['state'] for task in self.store.tasks()}
        self.assertEqual(states[expired], 'missed')
        self.assertEqual(states[runnable], 'pending')
        self.assertIn('补启动窗口', next(
            task for task in self.store.tasks() if task['id'] == expired)['error'])
        self.assertEqual(self.service.due(self.now)['id'], runnable)
        with self.assertRaises(ValueError):
            self.store.expire_pending_tasks(self.now, -1)

    def test_failed_or_conflicted_task_can_be_retried_or_cancelled(self):
        failed = self.store.add_task('失败', self.now, 1)
        self.store.fail_task(failed, '摄像头被占用')
        self.store.retry_task(failed, self.now+timedelta(seconds=1))
        task = next(task for task in self.store.tasks() if task['id'] == failed)
        self.assertEqual(task['state'], 'pending')
        self.assertIsNone(task['error'])
        self.store.cancel_task(failed)
        self.assertEqual(self.store.tasks()[0]['state'], 'cancelled')

        self.start()
        conflict = self.store.add_task('冲突后重试', self.now, 1)
        self.assertIsNone(self.service.due(self.now))
        with self.assertRaisesRegex(ValueError, '结束当前会话'):
            self.store.retry_task(conflict, self.now)
        self.service.end(1, self.now)
        self.store.retry_task(conflict, self.now+timedelta(seconds=2))
        self.assertEqual(next(
            task for task in self.store.tasks() if task['id'] == conflict)['state'], 'pending')

    def test_database_enforces_single_session(self):
        self.start()
        with self.assertRaises(sqlite3.IntegrityError):
            SessionService(self.store).start('第二个', 1, 0, self.now, {})
        self.assertEqual(len(self.store.history()), 1)

    def test_v1_database_migrates_without_losing_history(self):
        identifier = self.start()
        self.service.end(1, self.now)
        self.store.close()
        legacy = sqlite3.connect(self.root / 'observer.sqlite3')
        legacy.execute('ALTER TABLE tasks DROP COLUMN error')
        legacy.execute('ALTER TABLE sessions DROP COLUMN error')
        legacy.execute('DROP TABLE schema_migrations')
        legacy.execute('PRAGMA user_version=1')
        legacy.commit()
        legacy.close()
        self.store = ProductStore(self.root)
        self.service = SessionService(self.store)
        self.assertEqual(self.store.summary(identifier)['state'], 'ended')
        self.assertEqual(self.store.db.execute('PRAGMA user_version').fetchone()[0], 4)
        migrations = self.store.db.execute(
            'SELECT version FROM schema_migrations ORDER BY version').fetchall()
        self.assertEqual([row[0] for row in migrations], [1, 2, 3, 4])
        self.assertIn('error', {row[1] for row in self.store.db.execute('PRAGMA table_info(tasks)')})
        self.assertIn('error', {row[1] for row in self.store.db.execute('PRAGMA table_info(sessions)')})
        backup = sqlite3.connect(self.root / 'observer.sqlite3.v1.backup')
        try:
            self.assertEqual(backup.execute('PRAGMA user_version').fetchone()[0], 1)
            self.assertEqual(backup.execute('SELECT count(*) FROM sessions').fetchone()[0], 1)
        finally:
            backup.close()

    def test_migration_failure_rolls_back_and_keeps_backup(self):
        identifier = self.start()
        self.service.end(1, self.now)
        self.store.close()
        legacy = sqlite3.connect(self.root / 'observer.sqlite3')
        legacy.execute('ALTER TABLE tasks DROP COLUMN error')
        legacy.execute('ALTER TABLE sessions DROP COLUMN error')
        legacy.execute('DROP TABLE schema_migrations')
        legacy.execute('CREATE TABLE schema_migrations(broken TEXT)')
        legacy.execute('PRAGMA user_version=1')
        legacy.commit()
        legacy.close()
        with self.assertRaises(sqlite3.Error):
            ProductStore(self.root)
        inspect = sqlite3.connect(self.root / 'observer.sqlite3')
        try:
            self.assertEqual(inspect.execute('PRAGMA user_version').fetchone()[0], 1)
            self.assertEqual(inspect.execute('SELECT id FROM sessions').fetchone()[0], identifier)
            inspect.execute('DROP TABLE schema_migrations')
            inspect.commit()
        finally:
            inspect.close()
        self.store = ProductStore(self.root)
        self.service = SessionService(self.store)
        self.assertTrue((self.root / 'observer.sqlite3.v1.backup').exists())

    def test_v2_database_adds_task_error_and_keeps_tasks(self):
        identifier = self.store.add_task('保留任务', self.now+timedelta(minutes=1), 1)
        self.store.close()
        legacy = sqlite3.connect(self.root / 'observer.sqlite3')
        legacy.execute('ALTER TABLE tasks DROP COLUMN error')
        legacy.execute('ALTER TABLE sessions DROP COLUMN error')
        legacy.execute('DELETE FROM schema_migrations WHERE version=3')
        legacy.execute('DELETE FROM schema_migrations WHERE version=4')
        legacy.execute('PRAGMA user_version=2')
        legacy.commit()
        legacy.close()
        self.store = ProductStore(self.root)
        self.service = SessionService(self.store)
        task = next(task for task in self.store.tasks() if task['id'] == identifier)
        self.assertIsNone(task['error'])
        self.assertEqual(self.store.db.execute('PRAGMA user_version').fetchone()[0], 4)
        backup = sqlite3.connect(self.root / 'observer.sqlite3.v2.backup')
        try:
            self.assertEqual(backup.execute('PRAGMA user_version').fetchone()[0], 2)
            self.assertNotIn('error', {row[1] for row in backup.execute('PRAGMA table_info(tasks)')})
        finally:
            backup.close()

    def test_v3_database_adds_session_error_and_preserves_history(self):
        identifier = self.start()
        self.service.end(1, self.now)
        self.store.close()
        legacy = sqlite3.connect(self.root / 'observer.sqlite3')
        legacy.execute('ALTER TABLE sessions DROP COLUMN error')
        legacy.execute('DELETE FROM schema_migrations WHERE version=4')
        legacy.execute('PRAGMA user_version=3')
        legacy.commit()
        legacy.close()
        self.store = ProductStore(self.root)
        self.service = SessionService(self.store)
        self.assertIsNone(self.store.summary(identifier)['error'])
        self.assertEqual(self.store.db.execute('PRAGMA user_version').fetchone()[0], 4)
        backup = sqlite3.connect(self.root / 'observer.sqlite3.v3.backup')
        try:
            self.assertEqual(backup.execute('PRAGMA user_version').fetchone()[0], 3)
            self.assertNotIn('error', {row[1] for row in backup.execute('PRAGMA table_info(sessions)')})
        finally:
            backup.close()

    def test_current_version_with_missing_structure_is_rejected(self):
        self.store.close()
        damaged = sqlite3.connect(self.root / 'observer.sqlite3')
        damaged.execute('DROP TABLE totals')
        damaged.commit()
        damaged.close()
        with self.assertRaisesRegex(ValueError, '缺失表'):
            ProductStore(self.root)
        self.store = sqlite3.connect(self.root / 'observer.sqlite3')

    def test_write_failure_stops_scoring_and_rolls_back(self):
        identifier = self.start()
        self.store.db.execute("CREATE TRIGGER reject_action BEFORE INSERT ON actions "
                              "BEGIN SELECT RAISE(ABORT,'disk failure simulation'); END")
        self.store.db.commit()
        for at in range(10):
            self.feed(at, 'phone_held')
        with self.assertRaises(sqlite3.IntegrityError):
            self.feed(10, 'phone_held')
        self.assertEqual(self.service.state, 'faulted')
        self.feed(11, 'phone_held')
        self.assertEqual(self.store.summary(identifier)['actions'], [])

    def test_runtime_lock_can_be_reacquired(self):
        lock = RuntimeLock(self.root)
        try:
            with self.assertRaises(ValueError):
                RuntimeLock(self.root)
        finally:
            lock.close()
        RuntimeLock(self.root).close()

    def test_image_failure_keeps_event_and_delete_is_bounded(self):
        identifier = self.start()
        self.store.set_images(True)
        self.store.capture(identifier, 'event-test', None)
        self.assertIn('新鲜', self.store.summary(identifier)['evidence'][0]['error'])
        with self.assertRaises(ValueError):
            self.store.delete_session(identifier)
        self.service.end(0, self.now)
        outside = self.root.parent / 'research.jpg'
        outside.write_bytes(b'keep')
        with self.store.db:
            self.store.db.execute("UPDATE evidence SET path='../research.jpg'")
        with self.assertRaises(ValueError):
            self.store.delete_session(identifier)
        self.assertEqual(outside.read_bytes(), b'keep')
        self.assertEqual(len(self.store.history()), 1)
        with self.store.db:
            self.store.db.execute("UPDATE evidence SET path=''")
        self.store.delete_session(identifier)
        self.assertEqual(self.store.history(), [])

    def test_retention_only_deletes_ended_expired_product_sessions(self):
        self.start()
        self.service.end(0, self.now)
        self.start()
        self.assertEqual(self.store.purge(self.now+timedelta(days=31)), 1)
        self.assertEqual(len(self.store.history()), 1)

    def test_cli_history_and_parser_do_not_load_vision(self):
        from observer.cli import main, build_parser
        parsed = build_parser().parse_args(['product', 'run', '--demo'])
        self.assertTrue(parsed.demo)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['product', '--data-dir', str(self.root), 'history']), 0)
        self.assertIn('暂无', output.getvalue())

    def test_default_profile_is_automatic_and_can_be_disabled(self):
        from observer.product_cli import apply_default_profile
        from types import SimpleNamespace
        candidate = self.root / 'default.json'
        candidate.write_text('{}', encoding='utf-8')
        args = SimpleNamespace(demo=False, profile=None, no_profile=False)
        self.assertEqual(apply_default_profile(args, candidate), candidate)
        disabled = SimpleNamespace(demo=False, profile=None, no_profile=True)
        self.assertIsNone(apply_default_profile(disabled, candidate))
        explicit = SimpleNamespace(demo=False, profile=Path('other.json'), no_profile=False)
        self.assertEqual(apply_default_profile(explicit, candidate), Path('other.json'))

    def test_console_demo_uses_real_cli_and_persists_lifecycle(self):
        result = subprocess.run(
            [sys.executable, '-m', 'observer', 'product', '--data-dir', str(self.root), 'run', '--demo'],
            input='start 1 demo\npause\nresume\nend\nquit\n', text=True,
            capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr+result.stdout)
        rows = self.store.history()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['state'], 'ended')
        self.assertEqual(json.loads(self.store.summary(rows[0]['id'])['identity'])['source'], 'synthetic-demo')

    def test_image_written_once_and_removed_with_session(self):
        import numpy as np
        identifier = self.start()
        self.store.set_images(True)
        self.store.capture(identifier, 'test-image', np.zeros((32, 32, 3), dtype=np.uint8))
        self.store.capture(identifier, 'test-image', None)
        item = self.store.summary(identifier)['evidence'][0]
        self.assertIsNone(item['error'])
        path = self.store.safe_image_path(item['path'])
        self.assertTrue(path.exists())
        self.service.end(0, self.now)
        self.store.delete_session(identifier)
        self.assertFalse(path.exists())

    def test_camera_preflight_rejects_missing_model_before_camera_import(self):
        from observer.product_cli import CameraSource
        from unittest.mock import patch
        with patch('observer.camera.CameraWorker.start') as start:
            with self.assertRaises(FileNotFoundError):
                CameraSource(self.root / 'missing-models', None)
            start.assert_not_called()

    def test_camera_source_escalates_a_continuous_stale_frame(self):
        from observer.product_cli import CameraSource, SourceStalled
        from types import SimpleNamespace
        source = CameraSource.__new__(CameraSource)
        source.camera = SimpleNamespace(
            buffer=SimpleNamespace(latest=lambda: None),
            stats=SimpleNamespace(last_error='camera frame read failed'))
        source.frame = object()
        source.stale_since = None
        self.assertEqual(source.read(10).signal('away').state.value, 'unknown')
        with self.assertRaisesRegex(SourceStalled, 'camera frame read failed'):
            source.read(15.1)

    def test_camera_source_reconnect_replaces_only_camera_once(self):
        from observer.product_cli import CameraSource, SourceStalled
        from unittest.mock import MagicMock, patch
        source = CameraSource.__new__(CameraSource)
        original = MagicMock()
        replacement = MagicMock()
        replacement.start.return_value = True
        source.camera = original
        source.camera_config = object()
        source.probe = MagicMock()
        source.frame = object()
        source.stale_since = 1
        source.sequence = 9
        source.automatic_reconnects = 0
        with patch('observer.camera.CameraWorker', return_value=replacement) as worker:
            source.reconnect()
        original.stop.assert_called_once_with()
        worker.assert_called_once_with(source.camera_config)
        replacement.start.assert_called_once_with()
        self.assertIs(source.camera, replacement)
        self.assertIsNotNone(source.probe)
        self.assertEqual(source.automatic_reconnects, 1)
        with self.assertRaisesRegex(SourceStalled, '次数已用完'):
            source.reconnect()


if __name__ == '__main__':
    unittest.main()
