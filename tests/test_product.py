import contextlib
import hashlib
import io
import json
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from observer.observations import Observation, SignalValue
from observer.product import (
    NAMES, ProductStore, RuntimeLock, SessionService, product_rule_configs,
    unified_posture_signal,
)


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

    @staticmethod
    def drop_delivery_columns(db):
        db.execute('ALTER TABLE actions DROP COLUMN delivery_error')
        db.execute('ALTER TABLE actions DROP COLUMN delivery_attempted')
        db.execute('ALTER TABLE actions DROP COLUMN delivery_state')

    @staticmethod
    def drop_resource_column(db):
        db.execute('ALTER TABLE sessions DROP COLUMN resource_summary')

    @staticmethod
    def drop_plan_schema(db):
        db.execute('DROP TABLE focus_incidents')
        db.execute('ALTER TABLE sessions DROP COLUMN score_version')
        db.execute('DELETE FROM schema_migrations WHERE version=8')
        db.execute('DROP INDEX unique_plan_occurrence')
        db.execute('ALTER TABLE tasks DROP COLUMN allow_offscreen')
        db.execute('ALTER TABLE tasks DROP COLUMN break_minutes')
        db.execute('ALTER TABLE tasks DROP COLUMN break_every_minutes')
        db.execute('ALTER TABLE tasks DROP COLUMN plan_id')
        db.execute('DROP TABLE plans')
        db.execute('ALTER TABLE sessions DROP COLUMN break_seconds')
        db.execute('ALTER TABLE sessions DROP COLUMN extended_minutes')
        db.execute('ALTER TABLE sessions DROP COLUMN allow_offscreen')
        db.execute('ALTER TABLE sessions DROP COLUMN break_minutes')
        db.execute('ALTER TABLE sessions DROP COLUMN break_every_minutes')
        db.execute('DROP INDEX one_active_session')
        db.execute("CREATE UNIQUE INDEX one_active_session ON sessions((1)) "
                   "WHERE state IN ('monitoring','paused')")

    def test_persistent_phone_event_and_duplicate_observation(self):
        identifier = self.start()
        for at in range(16):
            self.feed(at, 'phone_held')
        self.feed(15, 'phone_held')
        for at in range(16, 20):
            self.feed(at)
        result = self.service.end(20, self.now+timedelta(seconds=20))
        self.assertEqual(result['score'], 25.0)
        self.assertEqual(result['violation_seconds'], 15)
        self.assertEqual(sum(a['kind']=='penalty' for a in result['actions']), 0)
        self.assertEqual(sum(a['kind']=='alert' for a in result['actions']), 1)
        self.assertEqual(sum(a['kind']=='resolved' for a in result['actions']), 1)
        self.store.close()
        self.store = ProductStore(self.root)
        reopened = self.store.summary(identifier)
        self.assertEqual(reopened['deducted'], 0)
        self.assertEqual(reopened['violation_seconds'], 15)

    def test_d7_acceptance_task_event_pause_reopen_and_delete(self):
        """One reproducible product loop crosses the storage and service boundaries."""
        scheduled = self.now + timedelta(seconds=10)
        task_id = self.store.add_task('D7验收', scheduled, 2)
        task = self.service.due(scheduled)
        self.assertEqual(task['id'], task_id)
        identifier = self.service.start(
            task['title'], task['minutes'], 0, scheduled,
            {'source': 'synthetic-acceptance'}, task_id,
        )

        for at in range(13):
            self.feed(at, 'phone_held')
        alert = next(action for action in self.store.summary(identifier)['actions']
                     if action['kind'] == 'alert')
        self.assertTrue(self.store.mark_notification(
            alert['id'], True, now=scheduled + timedelta(seconds=12)))

        self.service.pause(13, scheduled + timedelta(seconds=13))
        self.service.resume(30, scheduled + timedelta(seconds=30))
        for at in range(30, 35):
            self.feed(at)
        ended = self.service.end(35, scheduled + timedelta(seconds=35))
        self.assertAlmostEqual(ended['score'], 33.3)
        self.assertEqual(ended['violation_seconds'], 12)
        self.assertEqual(ended['state'], 'ended')

        self.store.close()
        self.store = ProductStore(self.root)
        reopened = self.store.summary(identifier)
        self.assertEqual(reopened['deducted'], 0)
        self.assertEqual(reopened['violation_seconds'], 12)
        self.assertEqual(next(action for action in reopened['actions']
                              if action['id'] == alert['id'])['delivery_state'], 'delivered')
        self.assertEqual(next(task for task in self.store.tasks()
                              if task['id'] == task_id)['state'], 'ended')

        self.store.delete_session(identifier)
        self.assertEqual(self.store.history(), [])
        self.assertEqual(self.store.data_status()['sessions'], 0)

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

    def test_pause_freezes_remaining_time_and_resume_extends_deadline(self):
        self.service.start('暂停倒计时', 1, 0, self.now, {'source': 'test'})
        self.service.pause(10, self.now + timedelta(seconds=10))
        self.assertEqual(self.service.remaining_seconds(10), 50)
        self.assertEqual(self.service.remaining_seconds(40), 50)

        self.service.extend(.5, self.now + timedelta(seconds=40))
        self.assertEqual(self.service.remaining_seconds(50), 80)
        self.service.resume(50, self.now + timedelta(seconds=50))
        self.assertEqual(self.service.deadline, 130)
        self.assertEqual(self.service.remaining_seconds(51), 79)
        self.service.end(51, self.now + timedelta(seconds=51))

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

    def test_pause_preserves_confirmed_violation_history(self):
        self.start()
        for cycle in range(6):
            start = cycle * 30
            if cycle:
                self.service.resume(start, self.now)
            for at in range(start, start+12):
                self.feed(at, 'phone_held')
            self.service.pause(start+12, self.now)
        result = self.service.end(200, self.now)
        self.assertEqual(result['deducted'], 0)
        self.assertGreater(result['violation_seconds'], 0)
        self.assertEqual(len(result['focus_incidents']), 6)

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
            self.feed(at, 'posture_deviation', 'ergonomic_risk')
        result = self.service.end(181, self.now)
        self.assertEqual(result['score'], 100)
        self.assertEqual(
            sum(a['kind'] == 'activated' for a in result['actions']), 1)
        self.assertEqual(sum(a['kind']=='alert' for a in result['actions']), 1)
        self.assertFalse(any(a['kind']=='penalty' for a in result['actions']))
        self.assertNotIn('ergonomic_risk', result['totals'])

    def test_product_has_one_posture_rule_and_merges_ergonomic_only_violation(self):
        self.assertEqual(
            [rule.rule_id for rule in product_rule_configs()
             if rule.rule_id in ('posture_deviation', 'ergonomic_risk')],
            ['posture_deviation'],
        )
        observation = Observation(0, {
            'posture_deviation': SignalValue.compliant(.8),
            'ergonomic_risk': SignalValue.violation(.7, reason='head_tilt'),
        })
        merged = unified_posture_signal(observation)
        self.assertEqual(merged.state.value, 'violation')
        self.assertEqual(merged.reason, 'head_tilt')

        self.start()
        for at in range(23):
            self.feed(at, 'ergonomic_risk')
        result = self.service.end(23, self.now + timedelta(seconds=23))
        posture = [action for action in result['actions']
                   if action['rule_id'] == 'posture_deviation']
        self.assertEqual(sum(action['kind'] == 'activated' for action in posture), 1)
        self.assertEqual(sum(action['kind'] == 'alert' for action in posture), 1)
        self.assertFalse(any(action['rule_id'] == 'ergonomic_risk'
                             for action in result['actions']))

    def test_health_reminder_cooldown_resets_for_each_session(self):
        first_id = self.start()
        for at in range(22):
            self.feed(at, 'posture_deviation')
        self.service.end(22, self.now + timedelta(seconds=22))

        second_id = self.service.start(
            '下一轮', 10, 30, self.now + timedelta(seconds=30), {'source': 'test'})
        for at in range(30, 52):
            self.feed(at, 'posture_deviation')
        self.service.end(52, self.now + timedelta(seconds=52))

        for identifier in (first_id, second_id):
            actions = self.store.summary(identifier)['actions']
            self.assertEqual(sum(action['kind'] == 'alert' for action in actions), 1)

    def test_calibrated_attention_event_uses_time_ratio_score(self):
        identifier = self.service.start(
            '离屏扣分', 10, 0, self.now,
            {'source': 'camera', 'soft_signals': 'calibrated'})
        for at in range(13):
            self.feed(at, 'attention_away')
        result = self.service.end(13, self.now + timedelta(seconds=13))
        penalties = [action for action in result['actions']
                     if action['rule_id'] == 'attention_away'
                     and action['kind'] == 'penalty']
        self.assertEqual(penalties, [])
        self.assertEqual(result['violation_seconds'], 12)
        self.assertEqual(result['score'], 7.7)
        self.assertEqual(result['focus_incidents'][0]['signals'], ['attention_away'])
        self.assertEqual(result['id'], identifier)

    def test_attention_and_phone_merge_into_one_timed_incident(self):
        self.service.start(
            '合并违规', 10, 0, self.now,
            {'source': 'camera', 'soft_signals': 'calibrated'})
        for at in range(21):
            violations = ['attention_away']
            if 3 <= at <= 13:
                violations.append('phone_held')
            self.feed(at, *violations)
        for at in range(21, 24):
            self.feed(at)
        result = self.service.end(24, self.now + timedelta(seconds=24))

        self.assertEqual(len(result['focus_incidents']), 1)
        incident = result['focus_incidents'][0]
        self.assertEqual(incident['signals'], ['phone_held', 'attention_away'])
        self.assertEqual(incident['violation_seconds'], 20)
        alerts = [action for action in result['actions'] if action['kind'] == 'alert']
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]['offset'], 13)
        self.assertEqual(result['score'], 16.7)

    def test_short_overlapping_phone_candidate_does_not_join_incident(self):
        self.service.start(
            '短候选', 10, 0, self.now,
            {'source': 'camera', 'soft_signals': 'calibrated'})
        for at in range(16):
            violations = ['attention_away']
            if 3 <= at <= 6:
                violations.append('phone_held')
            self.feed(at, *violations)
        for at in range(16, 19):
            self.feed(at)
        result = self.service.end(19, self.now + timedelta(seconds=19))

        self.assertEqual(result['focus_incidents'][0]['signals'], ['attention_away'])
        self.assertEqual(result['violation_seconds'], 15)
        alert = next(action for action in result['actions'] if action['kind'] == 'alert')
        self.assertEqual(alert['offset'], 10)

    def test_focus_alert_wait_is_bounded_and_late_signal_does_not_realert(self):
        self.service.start(
            '等待上限', 10, 0, self.now,
            {'source': 'camera', 'soft_signals': 'calibrated'})
        for at in range(26):
            violations = ['attention_away']
            if at >= 1:
                violations.append('away')
            self.feed(at, *violations)
        for at in range(26, 29):
            self.feed(at)
        result = self.service.end(29, self.now + timedelta(seconds=29))

        alerts = [action for action in result['actions'] if action['kind'] == 'alert']
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]['offset'], 20)
        self.assertEqual(result['focus_incidents'][0]['signals'],
                         ['away', 'attention_away'])

    def test_unavailable_attention_does_not_invalidate_other_scoring_signals(self):
        self.service.start(
            '无校准', 10, 0, self.now,
            {'source': 'camera', 'soft_signals': 'unavailable'})
        for at in range(10):
            signals = {name: SignalValue.compliant() for name in NAMES}
            signals['attention_away'] = SignalValue.unknown('未启用个人校准')
            self.service.process(
                Observation(at, signals), self.now + timedelta(seconds=at))
        result = self.service.end(10, self.now + timedelta(seconds=10))
        self.assertEqual(result['score'], 100)
        self.assertEqual(result['coverage'], .9)

    def test_allow_offscreen_suppresses_only_attention_soft_reminder(self):
        identifier = self.service.start(
            '资料阅读', 10, 0, self.now, {'source': 'test'},
            allow_offscreen=True)
        for at in range(21):
            self.feed(at, 'attention_away')
        result = self.service.end(21, self.now + timedelta(seconds=21))
        self.assertFalse(any(action['rule_id'] == 'attention_away'
                             for action in result['actions']))
        self.assertGreater(result['totals']['attention_away']['violation'], 0)
        self.assertTrue(result['allow_offscreen'])
        self.assertEqual(result['id'], identifier)

        self.service.start(
            '硬规则仍生效', 10, 30, self.now, {'source': 'test'},
            allow_offscreen=True)
        for at in range(30, 56):
            self.feed(at, 'away')
        result = self.service.end(56, self.now + timedelta(seconds=56))
        self.assertEqual(result['focus_incidents'][0]['signals'], ['away'])
        self.assertEqual(result['violation_seconds'], 25)

    def test_planned_break_tracks_rest_without_extending_deadline(self):
        self.service.start(
            '50/10缩短测试', 1, 0, self.now, {'source': 'test'},
            break_every_minutes=.1, break_minutes=.05)
        for at in range(7):
            self.feed(at)
        self.assertTrue(self.service.break_due())
        deadline = self.service.deadline
        self.service.start_break(6, self.now + timedelta(seconds=6))
        self.assertEqual(self.service.state, 'resting')
        self.service.finish_break(9, self.now + timedelta(seconds=9))
        self.assertEqual(self.service.state, 'monitoring')
        self.assertEqual(self.service.deadline, deadline)
        result = self.service.end(10, self.now + timedelta(seconds=10))
        self.assertEqual(result['break_seconds'], 3)
        self.assertEqual(result['break_every_minutes'], .1)
        self.assertEqual(result['break_minutes'], .05)

    def test_planned_break_resets_rule_continuity_and_is_not_eligible(self):
        self.service.start(
            '休息断开连续性', 1, 0, self.now, {'source': 'test'},
            break_every_minutes=.1, break_minutes=.05)
        for at in range(7):
            self.feed(at, 'phone_held')
        self.service.start_break(6, self.now + timedelta(seconds=6))
        self.service.finish_break(9, self.now + timedelta(seconds=9))
        for at in range(9, 16):
            self.feed(at, 'phone_held')
        result = self.service.end(15, self.now + timedelta(seconds=15))
        self.assertFalse(any(action['kind'] == 'penalty' for action in result['actions']))
        self.assertEqual(result['totals']['phone_held']['eligible'], 12)
        self.assertEqual(result['break_seconds'], 3)

    def test_extend_changes_only_current_session(self):
        task_id = self.store.add_task('一次任务', self.now, 1)
        task = self.service.due(self.now)
        session_id = self.service.start(
            task['title'], task['minutes'], 0, self.now, {'source': 'test'}, task_id)
        original_deadline = self.service.deadline
        self.service.extend(5, self.now + timedelta(seconds=1))
        self.assertEqual(self.service.deadline, original_deadline + 300)
        result = self.service.end(2, self.now + timedelta(seconds=2))
        self.assertEqual(result['id'], session_id)
        self.assertEqual(result['minutes'], 6)
        self.assertEqual(result['extended_minutes'], 5)
        self.assertEqual(next(item for item in self.store.tasks()
                              if item['id'] == task_id)['minutes'], 1)

    def test_today_summary_combines_plans_sessions_breaks_score_and_coverage(self):
        task_id = self.store.add_task('今日计划', self.now, 1)
        self.store.add_task('明日计划', self.now + timedelta(days=1), 1)
        task = self.service.due(self.now)
        self.service.start(
            task['title'], task['minutes'], 0, self.now, {'source': 'test'}, task_id,
            break_every_minutes=.2, break_minutes=.05)
        for at in range(13):
            self.feed(at, 'phone_held')
        self.service.start_break(12, self.now + timedelta(seconds=12))
        self.service.finish_break(15, self.now + timedelta(seconds=15))
        self.service.end(16, self.now + timedelta(seconds=16))

        summary = self.store.today_summary(self.now, 'America/Denver')
        self.assertEqual(summary['planned'], 1)
        self.assertEqual(summary['completed'], 1)
        self.assertEqual(summary['observed_seconds'], 13)
        self.assertEqual(summary['break_seconds'], 3)
        self.assertEqual(summary['deducted'], 0)
        self.assertEqual(summary['violation_seconds'], 12)
        self.assertEqual(summary['score'], 7.7)
        self.assertAlmostEqual(summary['coverage'], 12 / 13)

    def test_notification_delivery_is_idempotent_and_never_changes_score(self):
        identifier = self.start()
        for at in range(12):
            self.feed(at, 'attention_away')
        before = self.store.summary(identifier)
        alert = next(action for action in before['actions'] if action['kind'] == 'alert')
        self.assertEqual(alert['delivery_state'], 'pending')
        self.assertTrue(self.store.mark_notification(
            alert['id'], False, 'sound unavailable', self.now))
        self.assertEqual(next(action for action in self.store.summary(identifier)['actions']
                              if action['id'] == alert['id'])['delivery_state'], 'failed')
        self.assertTrue(self.store.mark_notification(alert['id'], True, now=self.now))
        self.assertFalse(self.store.mark_notification(alert['id'], True, now=self.now))
        after = self.store.summary(identifier)
        delivered = next(action for action in after['actions'] if action['id'] == alert['id'])
        self.assertEqual(delivered['delivery_state'], 'delivered')
        self.assertIsNone(delivered['delivery_error'])
        self.assertEqual(after['score'], before['score'])
        self.assertEqual(after['deducted'], before['deducted'])
        penalty = next(action for action in after['actions'] if action['kind'] != 'alert')
        with self.assertRaisesRegex(ValueError, '不需要提醒'):
            self.store.mark_notification(penalty['id'], True, now=self.now)

    def test_session_end_marks_unconfirmed_alert_failed_but_accepts_late_ack(self):
        identifier = self.start()
        for at in range(12):
            self.feed(at, 'attention_away')
        alert = next(action for action in self.store.summary(identifier)['actions']
                     if action['kind'] == 'alert')
        result = self.service.end(12, self.now+timedelta(seconds=12))
        ended_alert = next(action for action in result['actions'] if action['id'] == alert['id'])
        self.assertEqual(ended_alert['delivery_state'], 'failed')
        self.assertIn('会话结束前未确认', ended_alert['delivery_error'])
        self.assertTrue(self.store.mark_notification(
            alert['id'], True, now=self.now+timedelta(seconds=13)))
        final = self.store.summary(identifier)
        self.assertEqual(next(action for action in final['actions']
                              if action['id'] == alert['id'])['delivery_state'], 'delivered')
        self.assertEqual(final['deducted'], result['deducted'])

    def test_away_and_phone_merge_without_losing_either_signal(self):
        self.start()
        for at in range(25):
            self.feed(at, 'away', 'phone_held')
        result = self.service.end(25, self.now)
        self.assertEqual(result['deducted'], 0)
        self.assertEqual(result['violation_seconds'], 24)
        self.assertEqual(result['focus_incidents'][0]['signals'],
                         ['away', 'phone_held'])
        self.assertEqual(sum(a['kind'] == 'alert' for a in result['actions']), 1)

    def test_abandoned_session_keeps_ledger_and_stops_at_checkpoint(self):
        identifier = self.start()
        for at in range(12):
            self.feed(at, 'phone_held')
        self.store.interrupt_abandoned()
        summary = self.store.summary(identifier)
        self.assertEqual(summary['state'], 'interrupted')
        self.assertEqual(summary['ended'], summary['updated'])
        self.assertEqual(summary['deducted'], 0)
        self.assertEqual(summary['violation_seconds'], 11)
        alert = next(action for action in summary['actions'] if action['kind'] == 'alert')
        self.assertEqual(alert['delivery_state'], 'failed')
        self.assertIn('退出前未确认', alert['delivery_error'])

    def test_force_killed_process_releases_lock_and_recovers_committed_session(self):
        self.store.close()
        script = (
            "import os,sys,time; "
            "from pathlib import Path; "
            "from observer.product import ProductStore,RuntimeLock,SessionService,utc_now; "
            "root=Path(sys.argv[1]); lock=RuntimeLock(root); store=ProductStore(root); "
            "SessionService(store).start('强制终止测试',1,time.monotonic(),utc_now(),"
            "{'source':'subprocess-crash'}); os._exit(23)"
        )
        child = subprocess.run([sys.executable, '-c', script, str(self.root)],
                               capture_output=True, text=True, timeout=20)
        self.assertEqual(child.returncode, 23, child.stderr)
        lock = RuntimeLock(self.root)
        try:
            self.store = ProductStore(self.root)
            self.service = SessionService(self.store)
            self.assertEqual(self.store.interrupt_abandoned(), None)
        finally:
            lock.close()
        result = self.store.summary(self.store.history()[0]['id'])
        self.assertEqual(result['state'], 'interrupted')
        self.assertIn('上次退出', result['error'])
        self.assertIsNone(result['resource_summary'])

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

    def test_current_database_with_fake_single_session_index_is_rejected(self):
        self.store.close()
        damaged = sqlite3.connect(self.root / 'observer.sqlite3')
        damaged.execute('DROP INDEX one_active_session')
        damaged.execute('CREATE INDEX one_active_session ON sessions(state)')
        damaged.commit()
        damaged.close()
        with self.assertRaisesRegex(ValueError, '单会话约束损坏'):
            ProductStore(self.root)
        self.store = sqlite3.connect(self.root / 'observer.sqlite3')

    def test_v1_database_migrates_without_losing_history(self):
        identifier = self.start()
        self.service.end(1, self.now)
        self.store.close()
        legacy = sqlite3.connect(self.root / 'observer.sqlite3')
        self.drop_plan_schema(legacy)
        self.drop_resource_column(legacy)
        self.drop_delivery_columns(legacy)
        legacy.execute('ALTER TABLE tasks DROP COLUMN error')
        legacy.execute('ALTER TABLE sessions DROP COLUMN error')
        legacy.execute('DROP TABLE schema_migrations')
        legacy.execute('PRAGMA user_version=1')
        legacy.commit()
        legacy.close()
        self.store = ProductStore(self.root)
        self.service = SessionService(self.store)
        self.assertEqual(self.store.summary(identifier)['state'], 'ended')
        self.assertEqual(self.store.db.execute('PRAGMA user_version').fetchone()[0], 9)
        migrations = self.store.db.execute(
            'SELECT version FROM schema_migrations ORDER BY version').fetchall()
        self.assertEqual([row[0] for row in migrations], [1, 2, 3, 4, 5, 6, 7, 8, 9])
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
        self.drop_plan_schema(legacy)
        self.drop_resource_column(legacy)
        self.drop_delivery_columns(legacy)
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
        self.drop_plan_schema(legacy)
        self.drop_resource_column(legacy)
        self.drop_delivery_columns(legacy)
        legacy.execute('ALTER TABLE tasks DROP COLUMN error')
        legacy.execute('ALTER TABLE sessions DROP COLUMN error')
        legacy.execute('DELETE FROM schema_migrations WHERE version=3')
        legacy.execute('DELETE FROM schema_migrations WHERE version=4')
        legacy.execute('DELETE FROM schema_migrations WHERE version=5')
        legacy.execute('DELETE FROM schema_migrations WHERE version=6')
        legacy.execute('DELETE FROM schema_migrations WHERE version=7')
        legacy.execute('PRAGMA user_version=2')
        legacy.commit()
        legacy.close()
        self.store = ProductStore(self.root)
        self.service = SessionService(self.store)
        task = next(task for task in self.store.tasks() if task['id'] == identifier)
        self.assertIsNone(task['error'])
        self.assertEqual(self.store.db.execute('PRAGMA user_version').fetchone()[0], 9)
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
        self.drop_plan_schema(legacy)
        self.drop_resource_column(legacy)
        self.drop_delivery_columns(legacy)
        legacy.execute('ALTER TABLE sessions DROP COLUMN error')
        legacy.execute('DELETE FROM schema_migrations WHERE version=4')
        legacy.execute('DELETE FROM schema_migrations WHERE version=5')
        legacy.execute('DELETE FROM schema_migrations WHERE version=6')
        legacy.execute('DELETE FROM schema_migrations WHERE version=7')
        legacy.execute('PRAGMA user_version=3')
        legacy.commit()
        legacy.close()
        self.store = ProductStore(self.root)
        self.service = SessionService(self.store)
        self.assertIsNone(self.store.summary(identifier)['error'])
        self.assertEqual(self.store.db.execute('PRAGMA user_version').fetchone()[0], 9)
        backup = sqlite3.connect(self.root / 'observer.sqlite3.v3.backup')
        try:
            self.assertEqual(backup.execute('PRAGMA user_version').fetchone()[0], 3)
            self.assertNotIn('error', {row[1] for row in backup.execute('PRAGMA table_info(sessions)')})
        finally:
            backup.close()

    def test_v4_alert_delivery_migrates_as_historically_unknown(self):
        identifier = self.start()
        with self.store.db:
            self.store.db.execute(
                "INSERT INTO actions(id,session_id,event_id,rule_id,kind,offset,points,reason,delivery_state) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                ('legacy-alert', identifier, 'legacy-event', 'attention_away', 'alert',
                 1, 0, 'legacy', 'delivered'))
        self.service.end(2, self.now)
        self.store.close()
        legacy = sqlite3.connect(self.root / 'observer.sqlite3')
        self.drop_plan_schema(legacy)
        self.drop_resource_column(legacy)
        self.drop_delivery_columns(legacy)
        legacy.execute('DELETE FROM schema_migrations WHERE version=5')
        legacy.execute('DELETE FROM schema_migrations WHERE version=6')
        legacy.execute('DELETE FROM schema_migrations WHERE version=7')
        legacy.execute('PRAGMA user_version=4')
        legacy.commit()
        legacy.close()
        self.store = ProductStore(self.root)
        self.service = SessionService(self.store)
        alert = next(action for action in self.store.summary(identifier)['actions']
                     if action['id'] == 'legacy-alert')
        self.assertEqual(alert['delivery_state'], 'unknown')
        self.assertEqual(self.store.db.execute('PRAGMA user_version').fetchone()[0], 9)

    def test_v5_database_adds_empty_resource_summary_and_preserves_history(self):
        identifier = self.start()
        self.service.end(2, self.now)
        self.store.close()
        legacy = sqlite3.connect(self.root / 'observer.sqlite3')
        self.drop_plan_schema(legacy)
        self.drop_resource_column(legacy)
        legacy.execute('DELETE FROM schema_migrations WHERE version=6')
        legacy.execute('DELETE FROM schema_migrations WHERE version=7')
        legacy.execute('PRAGMA user_version=5')
        legacy.commit()
        legacy.close()
        self.store = ProductStore(self.root)
        self.service = SessionService(self.store)
        self.assertIsNone(self.store.summary(identifier)['resource_summary'])
        self.assertEqual(self.store.db.execute('PRAGMA user_version').fetchone()[0], 9)
        backup = sqlite3.connect(self.root / 'observer.sqlite3.v5.backup')
        try:
            columns = {row[1] for row in backup.execute('PRAGMA table_info(sessions)')}
            self.assertNotIn('resource_summary', columns)
        finally:
            backup.close()

    def test_v6_database_adds_plan_schema_and_preserves_existing_records(self):
        task_id = self.store.add_task('原有一次性任务', self.now, 1)
        task = self.service.due(self.now)
        session_id = self.service.start(
            task['title'], task['minutes'], 0, self.now, {'source': 'test'}, task_id)
        self.service.end(1, self.now + timedelta(seconds=1))
        self.store.close()
        legacy = sqlite3.connect(self.root / 'observer.sqlite3')
        self.drop_plan_schema(legacy)
        legacy.execute('DELETE FROM schema_migrations WHERE version=7')
        legacy.execute('PRAGMA user_version=6')
        legacy.commit()
        legacy.close()

        self.store = ProductStore(self.root)
        self.service = SessionService(self.store)
        task = next(item for item in self.store.tasks() if item['id'] == task_id)
        self.assertIsNone(task['plan_id'])
        self.assertEqual(self.store.summary(session_id)['state'], 'ended')
        self.assertEqual(self.store.plans(), [])
        self.assertEqual(self.store.db.execute('PRAGMA user_version').fetchone()[0], 9)
        backup = sqlite3.connect(self.root / 'observer.sqlite3.v6.backup')
        try:
            tables = {row[0] for row in backup.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertNotIn('plans', tables)
            columns = {row[1] for row in backup.execute('PRAGMA table_info(tasks)')}
            self.assertNotIn('plan_id', columns)
            self.assertEqual(backup.execute('SELECT count(*) FROM sessions').fetchone()[0], 1)
        finally:
            backup.close()

    def test_v7_database_marks_history_old_and_adds_focus_incidents(self):
        session_id = self.start()
        self.service.end(1, self.now + timedelta(seconds=1))
        self.store.close()
        legacy = sqlite3.connect(self.root / 'observer.sqlite3')
        legacy.execute('DROP TABLE focus_incidents')
        legacy.execute('ALTER TABLE sessions DROP COLUMN score_version')
        legacy.execute('DELETE FROM schema_migrations WHERE version=8')
        legacy.execute('PRAGMA user_version=7')
        legacy.commit()
        legacy.close()

        self.store = ProductStore(self.root)
        self.service = SessionService(self.store)
        result = self.store.summary(session_id)
        self.assertEqual(result['score_version'], 'points_v1')
        self.assertEqual(result['focus_incidents'], [])
        self.assertEqual(self.store.db.execute('PRAGMA user_version').fetchone()[0], 9)
        backup = sqlite3.connect(self.root / 'observer.sqlite3.v7.backup')
        try:
            columns = {row[1] for row in backup.execute('PRAGMA table_info(sessions)')}
            self.assertNotIn('score_version', columns)
        finally:
            backup.close()

    def test_plan_crud_normalizes_recurrence_breaks_and_flags(self):
        identifier = self.store.add_plan(
            ' 深度工作 ', '09:05', 'America/Denver', 'weekly', 90,
            weekdays=(4, 0, 4), break_every_minutes=50, break_minutes=10,
            allow_offscreen=True, now=self.now)
        plan = self.store.plans()[0]
        self.assertEqual(plan['id'], identifier)
        self.assertEqual(plan['title'], '深度工作')
        self.assertEqual(plan['weekdays'], (0, 4))
        self.assertTrue(plan['allow_offscreen'])
        self.assertTrue(plan['enabled'])
        self.assertEqual(plan['break_every_minutes'], 50)
        self.assertEqual(plan['break_minutes'], 10)

        updated = self.now + timedelta(minutes=1)
        self.store.update_plan(
            identifier, '日常专注', '08:30', 'America/Denver', 'daily', 25,
            weekdays=(6,), allow_offscreen=False, now=updated)
        plan = self.store.plans()[0]
        self.assertEqual(plan['weekdays'], tuple(range(7)))
        self.assertEqual(plan['created'], self.now.isoformat())
        self.assertEqual(plan['updated'], updated.isoformat())
        self.assertIsNone(plan['break_every_minutes'])
        self.assertFalse(plan['allow_offscreen'])

        self.store.set_plan_enabled(identifier, False, now=updated)
        self.assertFalse(self.store.plans()[0]['enabled'])
        self.store.set_plan_enabled(identifier, True, now=updated)
        self.assertTrue(self.store.plans()[0]['enabled'])
        self.store.delete_plan(identifier)
        self.assertEqual(self.store.plans(), [])
        with self.assertRaisesRegex(ValueError, '计划不存在'):
            self.store.delete_plan(identifier)

    def test_plan_rejects_invalid_definitions(self):
        valid = ('专注', '09:00', 'America/Denver', 'daily', 25)
        cases = (
            (('', *valid[1:]), '计划名'),
            ((valid[0], '9:00', *valid[2:]), 'HH:MM'),
            ((valid[0], valid[1], '', *valid[3:]), '时区'),
            ((valid[0], valid[1], 'Mars/Olympus', *valid[3:]), '时区'),
            ((valid[0], valid[1], valid[2], 'monthly', valid[4]), '重复规则'),
        )
        for arguments, message in cases:
            with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, message):
                self.store.add_plan(*arguments, now=self.now)
        with self.assertRaisesRegex(ValueError, '选择星期'):
            self.store.add_plan(*valid[:3], 'weekly', valid[4], now=self.now)
        with self.assertRaisesRegex(ValueError, '同时设置'):
            self.store.add_plan(*valid, break_every_minutes=50, now=self.now)
        with self.assertRaisesRegex(ValueError, '小于工作周期'):
            self.store.add_plan(
                *valid, break_every_minutes=10, break_minutes=10, now=self.now)
        with self.assertRaisesRegex(ValueError, '布尔值'):
            self.store.add_plan(*valid, allow_offscreen=1, now=self.now)
        with self.assertRaisesRegex(ValueError, '布尔值'):
            self.store.set_plan_enabled('missing', 1, now=self.now)

    def test_corrupt_persisted_plan_is_rejected_during_open(self):
        identifier = self.store.add_plan(
            '损坏计划', '09:00', 'America/Denver', 'weekly', 25,
            weekdays=(0,), now=self.now)
        self.store.close()
        damaged = sqlite3.connect(self.root / 'observer.sqlite3')
        damaged.execute("UPDATE plans SET weekdays='not-json' WHERE id=?", (identifier,))
        damaged.commit()
        damaged.close()
        with self.assertRaisesRegex(ValueError, '数据库计划数据损坏'):
            ProductStore(self.root)
        self.store = sqlite3.connect(self.root / 'observer.sqlite3')

    def test_recurring_plans_materialize_independent_idempotent_tasks(self):
        created = datetime(2026, 9, 9, 12, tzinfo=timezone.utc)
        daily = self.store.add_plan(
            '每天', '09:00', 'America/Denver', 'daily', 25, now=created)
        weekly = self.store.add_plan(
            '每周', '10:00', 'America/Denver', 'weekly', 50,
            weekdays=(3, 5), break_every_minutes=50, break_minutes=10,
            allow_offscreen=True, now=created)

        result = self.store.materialize_plans(self.now.replace(hour=12), horizon_days=3)
        self.assertEqual(result, {'created': 6, 'failed': 0})
        self.assertEqual(self.store.materialize_plans(
            self.now.replace(hour=12), horizon_days=3), {'created': 0, 'failed': 0})
        tasks = self.store.tasks()
        daily_tasks = [task for task in tasks if task['plan_id'] == daily]
        weekly_tasks = [task for task in tasks if task['plan_id'] == weekly]
        self.assertEqual(len(daily_tasks), 4)
        self.assertEqual(len(weekly_tasks), 2)
        self.assertTrue(all(task['id'] != other['id'] for index, task in enumerate(tasks)
                            for other in tasks[index + 1:]))
        self.assertTrue(all(task['allow_offscreen'] for task in weekly_tasks))
        self.assertTrue(all(task['break_every_minutes'] == 50 for task in weekly_tasks))

        self.store.close()
        self.store = ProductStore(self.root)
        self.service = SessionService(self.store)
        self.assertEqual(self.store.materialize_plans(
            self.now.replace(hour=12), horizon_days=3), {'created': 0, 'failed': 0})
        self.assertEqual(len(self.store.tasks()), 6)

    def test_overlapping_recurring_occurrences_keep_explicit_conflict(self):
        local_time = self.now.astimezone(ZoneInfo('America/Denver')).strftime('%H:%M')
        created = self.now - timedelta(minutes=1)
        first_plan = self.store.add_plan(
            '第一项', local_time, 'America/Denver', 'daily', 25, now=created)
        second_plan = self.store.add_plan(
            '第二项', local_time, 'America/Denver', 'daily', 25, now=created)
        self.store.materialize_plans(self.now, horizon_days=1)
        due = self.service.due(self.now)
        self.service.start(
            due['title'], due['minutes'], 0, self.now, {'source': 'test'}, due['id'])
        self.assertIsNone(self.service.due(self.now))
        current = [task for task in self.store.tasks()
                   if task['starts'] == self.now.isoformat()]
        self.assertEqual({task['plan_id'] for task in current}, {first_plan, second_plan})
        self.assertEqual(sorted(task['state'] for task in current), ['conflict', 'running'])

    def test_materialization_backfills_offline_occurrences_as_missed(self):
        created = datetime(2026, 9, 7, 12, tzinfo=timezone.utc)
        identifier = self.store.add_plan(
            '日计划', '09:00', 'America/Denver', 'daily', 25, now=created)
        now = datetime(2026, 9, 10, 18, tzinfo=timezone.utc)
        self.assertEqual(
            self.store.materialize_plans(now, horizon_days=1),
            {'created': 5, 'failed': 0},
        )
        self.assertEqual(self.store.expire_pending_tasks(now), 4)
        tasks = [task for task in self.store.tasks() if task['plan_id'] == identifier]
        self.assertEqual(sum(task['state'] == 'missed' for task in tasks), 4)
        self.assertEqual(sum(task['state'] == 'pending' for task in tasks), 1)

    def test_plan_changes_only_replace_future_occurrences(self):
        plan_id = self.store.add_plan(
            '旧计划', '09:00', 'America/Denver', 'daily', 25, now=self.now)
        self.store.materialize_plans(self.now, horizon_days=2)
        old = [task for task in self.store.tasks() if task['plan_id'] == plan_id]
        self.assertEqual(len(old), 2)
        with self.store.db:
            self.store.db.execute(
                "UPDATE tasks SET state='ended' WHERE id=?", (old[0]['id'],))
        changed_at = self.now + timedelta(hours=20)
        self.store.update_plan(
            plan_id, '新计划', '10:00', 'America/Denver', 'daily', 50,
            allow_offscreen=True, now=changed_at)
        self.store.materialize_plans(changed_at, horizon_days=2)
        tasks = [task for task in self.store.tasks() if task['plan_id'] == plan_id]
        historical = next(task for task in tasks if task['state'] == 'ended')
        self.assertEqual(historical['title'], '旧计划')
        self.assertFalse(historical['allow_offscreen'])
        self.assertIsNone(historical['break_every_minutes'])
        self.assertEqual(len([task for task in tasks if task['state'] == 'pending']), 2)
        self.assertTrue(all(task['title'] == '新计划' for task in tasks
                            if task['state'] == 'pending'))

        self.store.set_plan_enabled(plan_id, False, now=changed_at)
        tasks = [task for task in self.store.tasks() if task['plan_id'] == plan_id]
        self.assertEqual([task['state'] for task in tasks], ['ended'])
        self.store.set_plan_enabled(plan_id, True, now=changed_at)
        self.store.materialize_plans(changed_at, horizon_days=2)
        self.assertEqual(sum(task['state'] == 'pending' for task in self.store.tasks()), 2)
        self.store.delete_plan(plan_id, now=changed_at)
        tasks = self.store.tasks()
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]['state'], 'ended')
        self.assertIsNone(tasks[0]['plan_id'])

    def test_dst_gap_is_failed_and_repeated_time_occurs_once(self):
        spring = datetime(2026, 3, 7, tzinfo=timezone.utc)
        spring_plan = self.store.add_plan(
            '春季', '02:30', 'America/Denver', 'daily', 25, now=spring)
        result = self.store.materialize_plans(
            datetime(2026, 3, 8, tzinfo=timezone.utc), horizon_days=1)
        self.assertEqual(result['failed'], 1)
        failed = next(task for task in self.store.tasks()
                      if task['plan_id'] == spring_plan and task['state'] == 'failed')
        self.assertIn('夏令时', failed['error'])

        fall = datetime(2026, 10, 31, tzinfo=timezone.utc)
        fall_plan = self.store.add_plan(
            '秋季', '01:30', 'America/Denver', 'daily', 25, now=fall)
        self.store.materialize_plans(
            datetime(2026, 11, 1, tzinfo=timezone.utc), horizon_days=1)
        fall_tasks = [task for task in self.store.tasks() if task['plan_id'] == fall_plan]
        target = datetime(2026, 11, 1, 1, 30,
                          tzinfo=ZoneInfo('America/Denver'), fold=0)
        self.assertEqual(sum(task['starts'] == target.astimezone(timezone.utc).isoformat()
                             for task in fall_tasks), 1)

    def test_session_resource_summary_is_persisted(self):
        identifier = self.start()
        metrics = {'sample_count': 2, 'process_cpu_average_percent': 3.5,
                   'process_cpu_p95_percent': 5.0,
                   'process_memory_peak_mb': 120.0,
                   'process_memory_growth_mb_per_hour': None,
                   'within_budget': True, 'violations': [],
                   'sample_interval_seconds': 5.0, 'sampling_error': None}
        result = self.service.end(2, self.now, resource_summary=metrics)
        self.assertEqual(result['resource_summary'], metrics)
        self.store.close()
        self.store = ProductStore(self.root)
        self.service = SessionService(self.store)
        self.assertEqual(self.store.summary(identifier)['resource_summary'], metrics)

    def test_current_version_with_missing_structure_is_rejected(self):
        self.store.close()
        damaged = sqlite3.connect(self.root / 'observer.sqlite3')
        damaged.execute('DROP TABLE totals')
        damaged.commit()
        damaged.close()
        with self.assertRaisesRegex(ValueError, '缺失表'):
            ProductStore(self.root)
        self.store = sqlite3.connect(self.root / 'observer.sqlite3')

    def test_current_version_with_migration_ledger_gap_is_rejected(self):
        self.store.close()
        damaged = sqlite3.connect(self.root / 'observer.sqlite3')
        damaged.execute('DELETE FROM schema_migrations WHERE version=3')
        damaged.commit()
        damaged.close()
        with self.assertRaisesRegex(ValueError, '迁移记录与结构版本不一致'):
            ProductStore(self.root)
        self.store = sqlite3.connect(self.root / 'observer.sqlite3')

    def test_current_database_with_broken_foreign_key_is_rejected(self):
        self.store.close()
        damaged = sqlite3.connect(self.root / 'observer.sqlite3')
        damaged.execute(
            "INSERT INTO totals VALUES('missing-session','away',1,1,0)"
        )
        damaged.commit()
        damaged.close()
        with self.assertRaisesRegex(ValueError, '外键关系不一致'):
            ProductStore(self.root)
        self.store = sqlite3.connect(self.root / 'observer.sqlite3')

    def test_current_database_with_missing_foreign_key_definition_is_rejected(self):
        self.store.close()
        damaged = sqlite3.connect(self.root / 'observer.sqlite3')
        damaged.execute('DROP TABLE evidence')
        damaged.execute(
            'CREATE TABLE evidence(id TEXT PRIMARY KEY, session_id TEXT, event_id TEXT NOT NULL, '
            'rule_id TEXT, kind TEXT NOT NULL, offset REAL, signals TEXT NOT NULL, '
            'path TEXT NOT NULL, sha256 TEXT, error TEXT)'
        )
        damaged.commit()
        damaged.close()
        with self.assertRaisesRegex(ValueError, 'evidence.*外键定义损坏'):
            ProductStore(self.root)
        self.store = sqlite3.connect(self.root / 'observer.sqlite3')

    def test_invalid_product_settings_are_rejected_during_open(self):
        self.store.close()
        damaged = sqlite3.connect(self.root / 'observer.sqlite3')
        damaged.execute("UPDATE settings SET value='not-json' WHERE key='images'")
        damaged.commit()
        damaged.close()
        with self.assertRaisesRegex(ValueError, '设置损坏.*有效JSON'):
            ProductStore(self.root)

        damaged = sqlite3.connect(self.root / 'observer.sqlite3')
        damaged.execute("UPDATE settings SET value='false' WHERE key='images'")
        damaged.execute("UPDATE settings SET value='0' WHERE key='retention_days'")
        damaged.commit()
        damaged.close()
        with self.assertRaisesRegex(ValueError, 'retention_days须为1至3650天'):
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

    def test_console_rejects_second_instance_before_opening_database(self):
        from observer.cli import main
        from unittest.mock import patch
        self.store.close()
        lock = RuntimeLock(self.root)
        try:
            with patch('observer.product_cli.ProductStore') as product_store:
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    result = main([
                        'product', '--data-dir', str(self.root), 'run', '--demo',
                    ])
            self.assertEqual(result, 1)
            self.assertIn('已有 Observer 产品运行窗口', output.getvalue())
            product_store.assert_not_called()
        finally:
            lock.close()
            self.store = ProductStore(self.root)
            self.service = SessionService(self.store)

    def test_delete_all_rejects_running_instance_before_opening_database(self):
        from observer.cli import main
        from unittest.mock import patch
        identifier = self.start()
        self.service.end(1, self.now + timedelta(seconds=1))
        self.store.close()
        lock = RuntimeLock(self.root)
        try:
            with patch('observer.product_cli.ProductStore') as product_store:
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    result = main([
                        'product', '--data-dir', str(self.root), 'delete-all',
                    ])
            self.assertEqual(result, 1)
            self.assertIn('已有 Observer 产品运行窗口', output.getvalue())
            product_store.assert_not_called()
        finally:
            lock.close()
            self.store = ProductStore(self.root)
            self.service = SessionService(self.store)
        self.assertEqual(self.store.history()[0]['id'], identifier)

    def test_delete_all_removes_product_records_but_preserves_settings(self):
        from observer.cli import main
        import numpy as np
        identifier = self.start()
        self.store.set_images(True)
        self.store.capture(
            identifier, 'delete-all-image', np.zeros((16, 16, 3), dtype=np.uint8),
        )
        self.service.end(1, self.now + timedelta(seconds=1))
        self.store.add_task('待删除任务', self.now + timedelta(minutes=5), 1)
        self.store.close()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = main([
                'product', '--data-dir', str(self.root), 'delete-all',
            ])
        self.assertEqual(result, 0)
        self.assertIn('设置保留', output.getvalue())
        self.store = ProductStore(self.root)
        self.service = SessionService(self.store)
        status = self.store.data_status()
        self.assertEqual(status['sessions'], 0)
        self.assertEqual(status['tasks'], 0)
        self.assertEqual(status['images'], 0)
        self.assertTrue(status['images_enabled'])

    def test_delete_all_rejects_unsafe_path_before_removing_any_session(self):
        from observer.cli import main
        import numpy as np
        first = self.start()
        self.service.end(1, self.now + timedelta(seconds=1))
        second = self.start()
        self.store.set_images(True)
        self.store.capture(
            second, 'unsafe-delete-all', np.zeros((16, 16, 3), dtype=np.uint8),
        )
        self.service.end(2, self.now + timedelta(seconds=2))
        outside = self.root.parent / 'research.jpg'
        outside.write_bytes(b'keep')
        with self.store.db:
            self.store.db.execute(
                "UPDATE evidence SET path='../research.jpg' WHERE id='unsafe-delete-all'"
            )
        self.store.close()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = main([
                'product', '--data-dir', str(self.root), 'delete-all',
            ])
        self.assertEqual(result, 1)
        self.assertIn('证据路径越界', output.getvalue())
        self.store = ProductStore(self.root)
        self.service = SessionService(self.store)
        self.assertEqual({row['id'] for row in self.store.history()}, {first, second})
        self.assertEqual(outside.read_bytes(), b'keep')

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
        status = self.store.data_status()
        self.assertEqual(status['images'], 0)
        self.assertEqual(status['evidence_errors'], 1)
        with self.assertRaises(ValueError):
            self.store.delete_session(identifier)
        self.assertEqual(outside.read_bytes(), b'keep')
        self.assertEqual(len(self.store.history()), 1)
        with self.store.db:
            self.store.db.execute("UPDATE evidence SET path=''")
        self.store.delete_session(identifier)
        self.assertEqual(self.store.history(), [])

    def test_evidence_recovery_reconciles_atomic_write_crash_states(self):
        identifier = self.start()
        evidence = self.root / 'evidence'
        evidence.mkdir()
        completed = evidence / 'completed.jpg'
        completed.write_bytes(b'complete image bytes')
        (evidence / 'completed.jpg.tmp').write_bytes(b'stale temporary bytes')
        (evidence / 'unfinished.jpg.tmp').write_bytes(b'partial image bytes')
        changed = evidence / 'changed.jpg'
        changed.write_bytes(b'changed image bytes')
        with self.store.db:
            self.store.db.executemany(
                "INSERT INTO evidence(id,session_id,event_id,rule_id,kind,offset,signals,"
                "path,sha256,error) VALUES(?,?,?,?,?,?,?,?,?,?)",
                [
                    ('completed', identifier, 'completed', None, 'legacy', None, '[]',
                     'evidence/completed.jpg', None, 'pending'),
                    ('unfinished', identifier, 'unfinished', None, 'legacy', None, '[]',
                     'evidence/unfinished.jpg', None, 'pending'),
                    ('changed', identifier, 'changed', None, 'legacy', None, '[]',
                     'evidence/changed.jpg', 'wrong-hash', None),
                ],
            )

        result = self.store.recover_evidence()
        self.assertEqual(result, {'recovered': 1, 'failed': 2})
        records = {row['id']: dict(row) for row in self.store.db.execute(
            "SELECT id,sha256,error FROM evidence"
        )}
        self.assertEqual(
            records['completed']['sha256'], hashlib.sha256(completed.read_bytes()).hexdigest(),
        )
        self.assertIsNone(records['completed']['error'])
        self.assertIn('完成前中断', records['unfinished']['error'])
        self.assertIn('哈希不一致', records['changed']['error'])
        self.assertFalse((evidence / 'completed.jpg.tmp').exists())
        self.assertFalse((evidence / 'unfinished.jpg.tmp').exists())
        self.assertEqual(self.store.data_status()['images'], 1)
        self.assertEqual(self.store.data_status()['evidence_errors'], 2)

    def test_retention_only_deletes_ended_expired_product_sessions(self):
        self.start()
        self.service.end(0, self.now)
        self.start()
        self.assertEqual(self.store.purge(self.now+timedelta(days=31)), 1)
        self.assertEqual(len(self.store.history()), 1)

    def test_retention_isolates_unsafe_evidence_and_continues_cleanup(self):
        import numpy as np
        safe = self.start()
        self.service.end(1, self.now + timedelta(seconds=1))
        unsafe = self.start()
        self.store.set_images(True)
        self.store.capture(
            unsafe, 'unsafe-retention', np.zeros((16, 16, 3), dtype=np.uint8),
        )
        self.service.end(2, self.now + timedelta(seconds=2))
        outside = self.root.parent / 'research.jpg'
        outside.write_bytes(b'keep')
        with self.store.db:
            self.store.db.execute(
                "UPDATE evidence SET path='../research.jpg' WHERE id='unsafe-retention'"
            )

        self.assertEqual(self.store.purge(self.now + timedelta(days=31)), 1)
        self.assertEqual([row['id'] for row in self.store.history()], [unsafe])
        self.assertNotIn(safe, {row['id'] for row in self.store.history()})
        self.assertEqual(self.store.data_status()['evidence_errors'], 1)
        self.assertEqual(outside.read_bytes(), b'keep')

    def test_cli_history_and_parser_do_not_load_vision(self):
        from observer.cli import main, build_parser
        parsed = build_parser().parse_args(['product', 'run', '--demo'])
        self.assertTrue(parsed.demo)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['product', '--data-dir', str(self.root), 'history']), 0)
        self.assertIn('暂无', output.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['product', '--data-dir', str(self.root), 'data']), 0)
        self.assertIn('会话 0', output.getvalue())
        self.assertIn('截图 关闭', output.getvalue())
        parsed = build_parser().parse_args(['product', 'check', '--seconds', '2', '--demo'])
        self.assertEqual(parsed.seconds, 2)
        self.assertTrue(parsed.demo)
        self.assertEqual(
            build_parser().parse_args(['product', 'check', '--seconds', '3600']).seconds,
            3600,
        )

    def test_corrupt_session_metadata_does_not_break_history_or_summary(self):
        from observer.cli import main
        identifier = self.start()
        self.service.end(1, self.now + timedelta(seconds=1))
        with self.store.db:
            self.store.db.execute(
                "UPDATE sessions SET identity=?,resource_summary=? WHERE id=?",
                ('not-json', '{"sample_count":"many"}', identifier),
            )

        history = self.store.history()
        self.assertEqual(history[0]['mode'], 'unknown')
        summary = self.store.summary(identifier)
        self.assertIsNone(summary['resource_summary'])
        self.assertEqual(
            summary['data_warnings'],
            ['会话身份记录损坏', '资源汇总损坏'],
        )
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main([
                'product', '--data-dir', str(self.root), 'history',
            ]), 0)
        self.assertIn('[身份异常]', output.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main([
                'product', '--data-dir', str(self.root), 'show', identifier,
            ]), 0)
        self.assertIn('数据警告：会话身份记录损坏；资源汇总损坏', output.getvalue())

    def test_short_product_check_uses_temporary_data_and_reports_resources(self):
        from observer.cli import main
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['product', '--data-dir', str(self.root), 'check',
                                   '--seconds', '.1', '--demo']), 0)
        text = output.getvalue()
        self.assertIn('资源：CPU平均', text)
        self.assertIn('样本 2', text)
        self.assertIn('临时数据已清理', text)
        self.assertIn('不验证提醒交付', text)
        self.assertEqual(self.store.history(), [])

    def test_d5_resource_gate_requires_coverage_samples_growth_and_budget(self):
        from observer.product_cli import resource_gate_problems

        result = {
            'state': 'ended',
            'coverage': .99,
            'resource_summary': {
                'sample_count': 721,
                'process_memory_growth_mb_per_hour': 4.0,
                'within_budget': True,
                'violations': [],
                'sampling_error': None,
            },
        }
        self.assertIsNone(resource_gate_problems(result, 300))
        self.assertEqual(resource_gate_problems(result, 3600), [])

        failed = dict(result)
        failed['coverage'] = .79
        failed['resource_summary'] = {
            'sample_count': 20,
            'process_memory_growth_mb_per_hour': None,
            'within_budget': False,
            'violations': ['steady memory budget exceeded'],
            'sampling_error': 'metrics unavailable',
        }
        problems = resource_gate_problems(failed, 3600)
        self.assertIn('硬信号有效覆盖低于80%', problems)
        self.assertIn('资源采样失败：metrics unavailable', problems)
        self.assertIn('资源样本不足（少于360）', problems)
        self.assertIn('缺少每小时内存增长值', problems)
        self.assertIn('内存峰值超过500 MB', problems)

    def test_product_check_rejects_duration_above_long_gate(self):
        from observer.cli import main
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['product', 'check', '--seconds', '3600.1', '--demo']), 1)
        self.assertIn('0.1至3600秒', output.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['product', 'check', '--seconds', '3600', '--demo']), 1)
        self.assertIn('必须使用真实摄像头', output.getvalue())

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
        self.assertGreaterEqual(self.store.summary(rows[0]['id'])['resource_summary']['sample_count'], 1)

    def test_second_process_can_add_task_while_runtime_owns_database(self):
        active = self.start()
        future = datetime.now(timezone.utc) + timedelta(minutes=5)
        lock = RuntimeLock(self.root)
        try:
            child = subprocess.run(
                [sys.executable, '-m', 'observer', 'product', '--data-dir', str(self.root),
                 'task', '跨终端任务', '--at', future.isoformat(), '--minutes', '2'],
                capture_output=True, text=True, timeout=20,
            )
            self.assertEqual(child.returncode, 0, child.stderr + child.stdout)
            task_id = child.stdout.strip()
            task = next(task for task in self.store.tasks() if task['id'] == task_id)
            self.assertEqual(task['title'], '跨终端任务')
            self.assertEqual(task['state'], 'pending')
            self.assertEqual(self.store.summary(active)['state'], 'monitoring')
        finally:
            self.service.end(1, self.now + timedelta(seconds=1))
            lock.close()

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
        status = self.store.data_status()
        self.assertEqual(status['sessions'], 1)
        self.assertEqual(status['images'], 1)
        self.assertGreater(status['image_bytes'], 0)
        self.assertTrue(status['images_enabled'])
        self.service.end(0, self.now)
        self.store.delete_session(identifier)
        self.assertFalse(path.exists())
        status = self.store.data_status()
        self.assertEqual(status['sessions'], 0)
        self.assertEqual(status['images'], 0)
        self.assertEqual(status['image_bytes'], 0)

    def test_formal_focus_evidence_keeps_only_confirmed_start_and_end(self):
        import numpy as np
        from observer.product_cli import FocusEvidenceRecorder

        identifier = self.start()
        self.store.set_images(True)
        recorder = FocusEvidenceRecorder()
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        for second in range(13):
            observation = Observation(second, {
                name: (SignalValue.violation() if name == 'phone_held'
                       else SignalValue.compliant()) for name in NAMES
            })
            actions = self.service.process(
                observation, self.now + timedelta(seconds=second))
            recorder.update(self.store, self.service, frame, observation, actions)
        for second in range(13, 16):
            observation = Observation(second, {
                name: SignalValue.compliant() for name in NAMES
            })
            actions = self.service.process(
                observation, self.now + timedelta(seconds=second))
            recorder.update(self.store, self.service, frame, observation, actions)

        evidence = self.store.summary(identifier)['evidence']
        self.assertEqual([item['kind'] for item in evidence], ['start', 'end'])
        self.assertEqual([item['offset'] for item in evidence], [0, 15])
        self.assertEqual(len({item['event_id'] for item in evidence}), 1)
        self.assertTrue(all(item['rule_id'] == 'focus_violation' for item in evidence))

    def test_formal_focus_start_ignores_earlier_unconfirmed_signal(self):
        import numpy as np
        from observer.product_cli import FocusEvidenceRecorder

        identifier = self.start()
        self.store.set_images(True)
        recorder = FocusEvidenceRecorder()
        for second in range(14):
            states = {name: SignalValue.compliant() for name in NAMES}
            if second == 0:
                states['attention_away'] = SignalValue.violation()
            if 1 <= second <= 12:
                states['phone_held'] = SignalValue.violation()
            observation = Observation(second, states)
            actions = self.service.process(
                observation, self.now + timedelta(seconds=second))
            frame = np.full((24, 24, 3), second, dtype=np.uint8)
            recorder.update(self.store, self.service, frame, observation, actions)
        for second in range(14, 16):
            observation = Observation(second, {
                name: SignalValue.compliant() for name in NAMES
            })
            actions = self.service.process(
                observation, self.now + timedelta(seconds=second))
            recorder.update(
                self.store, self.service,
                np.full((24, 24, 3), second, dtype=np.uint8),
                observation, actions)

        evidence = self.store.summary(identifier)['evidence']
        self.assertEqual([item['kind'] for item in evidence], ['start', 'end'])
        self.assertEqual(evidence[0]['offset'], 1)
        self.assertEqual(json.loads(evidence[0]['signals']), ['phone_held'])

    def test_product_prewarms_with_frozen_phone_tracking_parameters(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch
        from observer.product_cli import prepare_camera_resources

        models = object()
        model_path = self.root / 'phone.onnx'
        manifest_path = self.root / 'manifest.json'
        config = SimpleNamespace(inference=object())
        backend = MagicMock()
        probe = MagicMock()
        with (
            patch('observer.product_cli.verify_frozen_models', return_value=(
                models, model_path, manifest_path)),
            patch('observer.product_cli.load_frozen_config', return_value=config),
            patch('observer.models.sha256_file', return_value='manifest-hash'),
            patch('observer.calibration.load_profile', return_value=None),
            patch('observer.object_detection.YoloOnnxBackend',
                  return_value=backend) as backend_type,
            patch('observer.vision.MediapipeVisionProbe',
                  return_value=probe) as probe_type,
        ):
            prepared = prepare_camera_resources(self.root, None)

        backend_type.assert_called_once_with(
            model_path, score_threshold=.25, single_class_name='cell_phone')
        probe_type.assert_called_once_with(
            models, config.inference, None,
            phone_score_threshold=.25,
            phone_retain_seconds=1.5,
            phone_center_distance=.18,
            object_backend=backend,
        )
        prepared.close()

    def test_test_evidence_captures_candidate_every_two_seconds_and_end(self):
        import numpy as np
        from observer.desktop import session_event_rows
        from observer.product_cli import FocusEvidenceRecorder

        identifier = self.start()
        self.store.set_images(True)
        self.store.set_test_evidence(True)
        recorder = FocusEvidenceRecorder()
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        for second in range(5):
            observation = Observation(second, {
                name: (SignalValue.violation() if name == 'phone_held'
                       else SignalValue.compliant()) for name in NAMES
            })
            actions = self.service.process(
                observation, self.now + timedelta(seconds=second))
            recorder.update(self.store, self.service, frame, observation, actions)
        observation = Observation(5, {
            name: SignalValue.compliant() for name in NAMES
        })
        actions = self.service.process(observation, self.now + timedelta(seconds=5))
        recorder.update(self.store, self.service, frame, observation, actions)

        evidence = self.store.summary(identifier)['evidence']
        self.assertEqual([item['kind'] for item in evidence],
                         ['start', 'sample', 'sample', 'end'])
        self.assertEqual([item['offset'] for item in evidence], [0, 2, 4, 5])
        self.assertTrue(all(item['rule_id'] == 'focus_candidate' for item in evidence))
        review = session_event_rows(self.store, self.store.summary(identifier))
        self.assertEqual(len(review), 1)
        self.assertTrue(review[0]['test_candidate'])
        self.assertEqual(len(review[0]['images']), 4)

    def test_v8_evidence_migrates_to_ordered_single_legacy_image(self):
        identifier = self.start()
        self.store.close()
        legacy = sqlite3.connect(self.root / 'observer.sqlite3')
        legacy.execute('ALTER TABLE evidence RENAME TO evidence_v9')
        legacy.execute(
            'CREATE TABLE evidence(id TEXT PRIMARY KEY, session_id TEXT '
            'REFERENCES sessions(id) ON DELETE CASCADE, path TEXT NOT NULL, '
            'sha256 TEXT, error TEXT)'
        )
        legacy.execute(
            "INSERT INTO evidence VALUES(?,?,?,?,?)",
            ('old-image', identifier, '', None, 'legacy error'),
        )
        legacy.execute('DROP TABLE evidence_v9')
        legacy.execute("DELETE FROM settings WHERE key='test_evidence'")
        legacy.execute('DELETE FROM schema_migrations WHERE version=9')
        legacy.execute('PRAGMA user_version=8')
        legacy.commit()
        legacy.close()

        self.store = ProductStore(self.root)
        self.service = SessionService(self.store)
        item = self.store.summary(identifier)['evidence'][0]
        self.assertEqual(item['event_id'], 'old-image')
        self.assertEqual(item['kind'], 'legacy')
        self.assertEqual(item['signals'], '[]')
        self.assertFalse(self.store.setting('test_evidence'))
        self.assertEqual(self.store.db.execute('PRAGMA user_version').fetchone()[0], 9)

    def test_opt_in_evidence_captures_soft_event_once(self):
        import numpy as np
        from observer.product_cli import capture_event_evidence

        identifier = self.start()
        self.store.set_images(True)
        self.feed(0)
        actions = []
        for second in range(1, 23):
            actions.extend(self.feed(second, 'posture_deviation'))
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        captured = [capture_event_evidence(
            self.store, identifier, frame, action
        ) for action in actions]

        self.assertEqual(sum(captured), 1)
        evidence = self.store.summary(identifier)['evidence']
        self.assertEqual(len(evidence), 1)
        activated = next(action for action in actions
                         if action.kind.value == 'activated')
        self.assertEqual(evidence[0]['id'], activated.event_id)
        self.assertIsNone(evidence[0]['error'])
        from observer.desktop import session_event_rows
        review = session_event_rows(self.store, self.store.summary(identifier))
        self.assertEqual(len(review), 1)
        self.assertEqual(review[0]['rule_id'], 'posture_deviation')
        self.assertEqual(review[0]['image_status'], '查看')
        self.assertTrue(Path(review[0]['image_path']).is_file())

    def test_camera_preflight_rejects_missing_model_before_camera_import(self):
        from observer.product_cli import CameraSource
        from unittest.mock import patch
        with patch('observer.camera.CameraWorker.start') as start:
            with self.assertRaises(FileNotFoundError):
                CameraSource(self.root / 'missing-models', None)
            start.assert_not_called()

    def test_camera_source_warms_first_frame_without_profile_before_ready(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch
        from observer.product_cli import CameraSource, load_frozen_config

        frame = object()
        snapshot = SimpleNamespace(frame=frame, captured_monotonic=time.monotonic(), sequence=1)
        camera = MagicMock()
        camera.start.return_value = True
        camera.stats.opened = True
        camera.buffer.latest.return_value = snapshot
        probe = MagicMock()
        prepared = SimpleNamespace(
            config=load_frozen_config(), profile=None, model_path=Path('model.onnx'),
            profile_sha256=None, manifest_sha256='manifest', background=False,
            timings={'verify': 0., 'models': 0., 'total': 0., 'details': {}},
            take_probe=MagicMock(return_value=probe),
        )

        with (patch('observer.camera.CameraWorker', return_value=camera),
              patch('observer.product_cli.INITIAL_VISION_WARMUP_SECONDS', 0)):
            source = CameraSource(self.root, None, prepared=prepared)
        probe.process_due.assert_called_once_with(frame, snapshot.captured_monotonic)
        self.assertIn('first_inference', source.identity['startup_seconds']['details'])
        source.close()
        camera.stop.assert_called_once_with()
        probe.close.assert_called_once_with()

    def test_camera_source_fails_if_frames_stop_during_warmup(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch
        from observer.product_cli import CameraSource, load_frozen_config

        first_frame = object()
        camera = MagicMock()
        camera.start.return_value = True
        camera.stats.opened = True
        camera.stats.last_error = None
        frames_seen = 0

        def latest(**_kwargs):
            nonlocal frames_seen
            frames_seen += 1
            if frames_seen == 1:
                return SimpleNamespace(
                    frame=first_frame, captured_monotonic=time.monotonic(), sequence=1)
            return None

        camera.buffer.latest.side_effect = latest
        probe = MagicMock()
        prepared = SimpleNamespace(
            config=load_frozen_config(), profile=None, model_path=Path('model.onnx'),
            profile_sha256=None, manifest_sha256='manifest', background=False,
            timings={'verify': 0., 'models': 0., 'total': 0., 'details': {}},
            take_probe=MagicMock(return_value=probe),
        )
        with (patch('observer.camera.CameraWorker', return_value=camera),
              patch('observer.product_cli.INITIAL_VISION_WARMUP_SECONDS', .1),
              patch('observer.product_cli.FRESH_FRAME_SECONDS', .03)):
            with self.assertRaisesRegex(RuntimeError, '准备期间画面中断'):
                CameraSource(self.root, None, prepared=prepared)
        camera.stop.assert_called_once_with()
        probe.close.assert_called_once_with()

    def test_background_camera_preloader_prepares_once_and_closes_unused_models(self):
        import threading
        import time
        from observer.product_cli import BackgroundCameraPreloader

        gate = threading.Event()
        invoked = threading.Event()
        prepared_items = []

        class Prepared:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        def prepare():
            invoked.set()
            gate.wait(2)
            item = Prepared()
            prepared_items.append(item)
            return item

        preloader = BackgroundCameraPreloader(prepare)
        self.assertTrue(preloader.start())
        self.assertTrue(invoked.wait(1))
        self.assertFalse(preloader.start())
        gate.set()
        deadline = time.monotonic() + 2
        while not preloader.ready and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(preloader.ready)
        self.assertIs(preloader.acquire(), prepared_items[0])

        self.assertTrue(preloader.start())
        deadline = time.monotonic() + 2
        while not preloader.ready and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(preloader.ready)
        preloader.close()
        self.assertTrue(prepared_items[1].closed)

    def test_frozen_product_identity_matches_c_closeout(self):
        from observer.product_cli import (
            FROZEN_CONFIG_SHA, FROZEN_MANIFEST_SHA, FROZEN_MODEL_SIZE,
            load_frozen_config, verify_frozen_models,
        )
        from observer.models import sha256_file
        config = load_frozen_config()
        _, model_path, manifest_path = verify_frozen_models(Path('models'))
        self.assertEqual(model_path.stat().st_size, FROZEN_MODEL_SIZE)
        self.assertEqual(sha256_file(manifest_path).lower(), FROZEN_MANIFEST_SHA)
        self.assertEqual(config.camera.capture_fps, 15)
        self.assertEqual(FROZEN_CONFIG_SHA,
                         '67a2d6a2890d8495147ab00390c02e7b329eb608829f7d3c796c214f57b41e06')

    def test_tampered_default_config_is_rejected_before_camera(self):
        from observer.product_cli import load_frozen_config
        candidate = self.root / 'changed-config.json'
        candidate.write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'C 冻结身份'):
            load_frozen_config(candidate)

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
        from types import SimpleNamespace
        source = CameraSource.__new__(CameraSource)
        original = MagicMock()
        replacement = MagicMock()
        replacement.start.return_value = True
        replacement.stats.opened = True
        replacement.buffer.latest.return_value = SimpleNamespace(
            captured_monotonic=time.monotonic())
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

    def test_camera_source_reconnect_requires_new_frame(self):
        from observer.product_cli import CameraSource, SourceStalled
        from unittest.mock import MagicMock, patch
        source = CameraSource.__new__(CameraSource)
        source.camera = MagicMock()
        source.camera_config = object()
        source.frame = object()
        source.stale_since = 1
        source.sequence = 9
        source.automatic_reconnects = 0
        replacement = MagicMock()
        replacement.start.return_value = True
        replacement.stats.opened = True
        replacement.stats.last_error = 'camera frame read failed'
        replacement.buffer.latest.return_value = None
        with (patch('observer.camera.CameraWorker', return_value=replacement),
              patch('observer.product_cli.CAMERA_FRAME_WAIT_SECONDS', .03)):
            with self.assertRaisesRegex(SourceStalled, 'camera frame read failed'):
                source.reconnect()
        replacement.stop.assert_called_once_with()
        self.assertEqual(source.automatic_reconnects, 1)
        self.assertIsNone(source.frame)

    def test_camera_frame_wait_rejects_stale_frame_and_checks_cancel(self):
        from observer.product_cli import SourceStalled, wait_for_camera_frame
        from types import SimpleNamespace
        stale = SimpleNamespace(captured_monotonic=time.monotonic() - 2)
        camera = SimpleNamespace(
            buffer=SimpleNamespace(latest=lambda: stale),
            stats=SimpleNamespace(opened=True, last_error=None))
        with self.assertRaisesRegex(SourceStalled, '没有新鲜画面'):
            wait_for_camera_frame(camera, timeout_seconds=.02)
        with self.assertRaisesRegex(SourceStalled, '已取消'):
            wait_for_camera_frame(camera, timeout_seconds=5, cancelled=lambda: True)


if __name__ == '__main__':
    unittest.main()
