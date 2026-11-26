"""Small local product layer. No vision dependencies and no research-data writes."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import stat
from fnmatch import fnmatch
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from observer.focus import FOCUS_RULE_ID, FOCUS_SIGNALS, FocusIncidentTracker
from observer.observations import Observation, SignalState, SignalValue
from observer.rules import ActionKind, default_rule_configs
from observer.scoring import RuleEngine

HARD_SIGNALS = ("away", "phone_held")
SCORE_VERSION = "time_ratio_v2"
SCHEMA_VERSION = 10
REQUIRED_SCHEMA = {
    'plans': {'id', 'title', 'local_time', 'timezone', 'recurrence', 'weekdays',
              'minutes', 'break_every_minutes', 'break_minutes', 'allow_offscreen',
              'enabled', 'created', 'updated'},
    'tasks': {'id', 'title', 'starts', 'minutes', 'state', 'error', 'plan_id',
              'break_every_minutes', 'break_minutes', 'allow_offscreen'},
    'sessions': {'id', 'title', 'task_id', 'started', 'updated', 'ended', 'state',
                 'minutes', 'identity', 'policy', 'error', 'resource_summary',
                 'break_every_minutes', 'break_minutes', 'allow_offscreen',
                 'extended_minutes', 'break_seconds', 'score_version'},
    'totals': {'session_id', 'signal', 'eligible', 'valid', 'violation'},
    'actions': {'id', 'session_id', 'event_id', 'rule_id', 'kind', 'offset', 'points', 'reason',
                'delivery_state', 'delivery_attempted', 'delivery_error'},
    'evidence': {'id', 'session_id', 'event_id', 'rule_id', 'kind', 'offset',
                 'signals', 'path', 'sha256', 'error', 'expired_at'},
    'focus_incidents': {'id', 'session_id', 'started', 'ended', 'signals', 'segments',
                        'violation_seconds', 'alert_offset'},
    'settings': {'key', 'value'},
    'session_work': {'session_id', 'sequence', 'started', 'ended', 'seconds'},
    'schema_migrations': {'version', 'applied', 'description'},
}
REQUIRED_FOREIGN_KEYS = {
    'tasks': {('plan_id', 'plans', 'id', 'SET NULL')},
    'sessions': {('task_id', 'tasks', 'id', 'NO ACTION')},
    'totals': {('session_id', 'sessions', 'id', 'CASCADE')},
    'actions': {('session_id', 'sessions', 'id', 'CASCADE')},
    'evidence': {('session_id', 'sessions', 'id', 'CASCADE')},
    'focus_incidents': {('session_id', 'sessions', 'id', 'CASCADE')},
    'session_work': {('session_id', 'sessions', 'id', 'CASCADE')},
}
NAMES = {"away": "离席", "phone_held": "可见手机", "attention_away": "离开屏幕",
         FOCUS_RULE_ID: "专注违规",
         "focus_candidate": "测试候选（未达到成立门槛）",
         "posture_deviation": "姿态（软提示）",
         "ergonomic_risk": "健康提示（旧版）"}


def _json_object(raw: object) -> dict | None:
    if not isinstance(raw, str):
        return None
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _session_mode(identity: object) -> str:
    value = _json_object(identity)
    if value is None:
        return 'unknown'
    return 'demo' if value.get('source') == 'synthetic-demo' else 'real'


def _session_scoring_signals(identity: object, policy: object,
                             allow_offscreen: bool = False,
                             score_version: str = 'points_v1') -> tuple[str, ...]:
    """Return scored signals for the policy frozen into this session."""
    signals = list(HARD_SIGNALS)
    identity_value = _json_object(identity)
    if identity_value and identity_value.get('soft_signals') == 'unavailable':
        return tuple(signals)
    if score_version == SCORE_VERSION:
        if not allow_offscreen:
            signals.append('attention_away')
        return tuple(signals)
    try:
        rules = json.loads(policy) if isinstance(policy, str) else None
    except (json.JSONDecodeError, TypeError):
        rules = None
    if isinstance(rules, list) and any(
        isinstance(rule, dict)
        and rule.get('signal_name') == 'attention_away'
        and isinstance(rule.get('maximum_penalty_points'), (int, float))
        and rule['maximum_penalty_points'] > 0
        for rule in rules
    ):
        signals.append('attention_away')
    return tuple(signals)


def _resource_summary(raw: object) -> dict | None:
    value = _json_object(raw)
    if value is None or type(value.get('sample_count')) is not int:
        return None
    if value['sample_count'] < 0:
        return None
    if value['sample_count']:
        required = ('process_cpu_average_percent', 'process_cpu_p95_percent',
                    'process_memory_peak_mb')
        if any(type(value.get(name)) not in (int, float) for name in required):
            return None
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("时间必须包含时区，例如 2026-09-15T09:00:00-06:00")
    return value.astimezone(timezone.utc).isoformat()


def minutes_valid(minutes: float) -> float:
    if not math.isfinite(minutes) or not 0 < minutes <= 1440:
        raise ValueError("时长须大于 0 且不超过 1440 分钟")
    return minutes


def product_rule_configs():
    """Apply current product consequences without changing the frozen C config."""
    configs = []
    for config in default_rule_configs():
        # D exposes one posture concept.  The research pipeline may still emit
        # ergonomic_risk for explanation, but product sessions merge it into
        # posture_deviation instead of creating a second event.
        if config.signal_name == 'ergonomic_risk':
            continue
        if config.signal_name in FOCUS_SIGNALS:
            configs.append(replace(
                config,
                trigger_after_seconds=(10.0 if config.signal_name == 'attention_away'
                                       else config.trigger_after_seconds),
                maximum_pending_sample_gap_seconds=(
                    config.trigger_after_seconds if config.signal_name == 'away' else None
                ),
                initial_penalty_points=0.0,
                repeat_penalty_points=0.0,
                maximum_penalty_points=0.0,
            ))
        else:
            configs.append(config)
    return tuple(configs)


def unified_posture_signal(observation: Observation) -> SignalValue:
    """Merge the two calibrated posture interpretations for D product use."""
    posture = observation.signal('posture_deviation')
    ergonomic = observation.signal('ergonomic_risk')
    values = (posture, ergonomic)
    metrics = {key: value for signal in values for key, value in signal.metrics.items()}
    violations = [signal for signal in values if signal.state is SignalState.VIOLATION]
    if violations:
        reasons = [signal.reason for signal in violations if signal.reason]
        confidences = [signal.confidence for signal in violations
                       if signal.confidence is not None]
        return SignalValue.violation(
            max(confidences) if confidences else None,
            reason='; '.join(dict.fromkeys(reasons)) or None,
            **metrics,
        )
    compliant = [signal for signal in values if signal.state is SignalState.COMPLIANT]
    if compliant:
        confidences = [signal.confidence for signal in compliant
                       if signal.confidence is not None]
        return SignalValue.compliant(
            max(confidences) if confidences else None, **metrics)
    exempt = [signal for signal in values if signal.state is SignalState.EXEMPT]
    if exempt:
        return SignalValue.exempt(
            '; '.join(dict.fromkeys(signal.reason for signal in exempt
                                    if signal.reason)) or '姿态观察已豁免')
    reasons = [signal.reason for signal in values if signal.reason]
    return SignalValue.unknown(
        '; '.join(dict.fromkeys(reasons)) or '姿态当前不可判断', **metrics)


def break_values(break_every_minutes: float | None,
                 break_minutes: float | None) -> tuple[float | None, float | None]:
    if (break_every_minutes is None) != (break_minutes is None):
        raise ValueError("休息周期和时长须同时设置")
    if break_every_minutes is None:
        return None, None
    every = minutes_valid(float(break_every_minutes))
    rest = minutes_valid(float(break_minutes))
    if rest >= every:
        raise ValueError("休息时长须小于工作周期")
    return every, rest


def plan_values(title: str, local_time: str, timezone_name: str, recurrence: str,
                weekdays=(), break_every_minutes: float | None = None,
                break_minutes: float | None = None,
                allow_offscreen: bool = False) -> tuple:
    """Validate and normalize one deliberately small recurring-plan definition."""
    title = str(title).strip()
    if not title:
        raise ValueError("计划名不能为空")
    if re.fullmatch(r'[0-9]{2}:[0-9]{2}', str(local_time)) is None:
        raise ValueError("计划时间须为HH:MM")
    try:
        parsed_time = datetime.strptime(str(local_time), '%H:%M')
    except (TypeError, ValueError) as exc:
        raise ValueError("计划时间须为HH:MM") from exc
    local_time = parsed_time.strftime('%H:%M')
    timezone_name = str(timezone_name).strip()
    if not timezone_name or len(timezone_name) > 100:
        raise ValueError("计划时区无效")
    try:
        ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError("计划时区无效") from exc
    if recurrence not in ('daily', 'weekdays', 'weekly'):
        raise ValueError("重复规则只支持每天、工作日或每周选定星期")
    if recurrence == 'daily':
        normalized_days = tuple(range(7))
    elif recurrence == 'weekdays':
        normalized_days = tuple(range(5))
    else:
        try:
            normalized_days = tuple(sorted({int(day) for day in weekdays}))
        except (TypeError, ValueError) as exc:
            raise ValueError("每周计划须选择星期") from exc
        if not normalized_days or any(day < 0 or day > 6 for day in normalized_days):
            raise ValueError("每周计划须选择星期一至星期日")
    break_every_minutes, break_minutes = break_values(
        break_every_minutes, break_minutes)
    if type(allow_offscreen) is not bool:
        raise ValueError("允许离屏资料须为布尔值")
    return (title, local_time, timezone_name, recurrence, normalized_days,
            break_every_minutes, break_minutes, allow_offscreen)


class ProductStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "observer.sqlite3", timeout=5)
        self.db.row_factory = sqlite3.Row
        try:
            self.db.execute("PRAGMA foreign_keys=ON")
            self.db.execute("PRAGMA journal_mode=WAL")
            version = self.db.execute("PRAGMA user_version").fetchone()[0]
            if version < 0 or version > SCHEMA_VERSION:
                raise ValueError(f"不支持数据库版本 {version}")
            if version == 0:
                self._create_schema()
            elif version < SCHEMA_VERSION:
                self._backup_before_migration(version)
                while version < SCHEMA_VERSION:
                    previous = version
                    if version == 1:
                        self._migrate_v1_to_v2()
                    elif version == 2:
                        self._migrate_v2_to_v3()
                    elif version == 3:
                        self._migrate_v3_to_v4()
                    elif version == 4:
                        self._migrate_v4_to_v5()
                    elif version == 5:
                        self._migrate_v5_to_v6()
                    elif version == 6:
                        self._migrate_v6_to_v7()
                    elif version == 7:
                        self._migrate_v7_to_v8()
                    elif version == 8:
                        self._migrate_v8_to_v9()
                    elif version == 9:
                        self._migrate_v9_to_v10()
                    version = self.db.execute("PRAGMA user_version").fetchone()[0]
                    if version != previous + 1:
                        raise RuntimeError(
                            f"数据库迁移未按顺序推进：{previous} → {version}"
                        )
            self._validate_schema()
        except BaseException:
            self.db.close()
            raise

    def _backup_before_migration(self, version: int):
        target = self.root / f'observer.sqlite3.v{version}.backup'
        temporary = self.root / f'observer.sqlite3.v{version}.backup.tmp'
        temporary.unlink(missing_ok=True)
        backup = sqlite3.connect(temporary)
        try:
            self.db.backup(backup)
        finally:
            backup.close()
        temporary.replace(target)

    def _validate_schema(self):
        quick_check = [row[0] for row in self.db.execute("PRAGMA quick_check")]
        if quick_check != ['ok']:
            detail = '; '.join(str(value) for value in quick_check[:3])
            raise ValueError(f"数据库完整性检查失败：{detail}")
        foreign_key_error = self.db.execute("PRAGMA foreign_key_check").fetchone()
        if foreign_key_error is not None:
            raise ValueError("数据库外键关系不一致")
        tables = {row[0] for row in self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        missing = set(REQUIRED_SCHEMA) - tables
        if missing:
            raise ValueError(f"数据库结构缺失表：{', '.join(sorted(missing))}")
        for table, expected in REQUIRED_SCHEMA.items():
            columns = {row[1] for row in self.db.execute(f'PRAGMA table_info({table})')}
            missing_columns = expected - columns
            if missing_columns:
                raise ValueError(
                    f"数据库表 {table} 缺失字段：{', '.join(sorted(missing_columns))}"
                )
        for table, expected in REQUIRED_FOREIGN_KEYS.items():
            actual = {
                (row[3], row[2], row[4], row[6])
                for row in self.db.execute(f'PRAGMA foreign_key_list({table})')
            }
            if actual != expected:
                raise ValueError(f"数据库表 {table} 的外键定义损坏")
        migrations = [row[0] for row in self.db.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        )]
        expected_migrations = list(range(1, SCHEMA_VERSION + 1))
        if migrations != expected_migrations:
            raise ValueError("数据库迁移记录与结构版本不一致")
        index_row = self.db.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name='one_active_session'"
        ).fetchone()
        if index_row is None:
            raise ValueError("数据库缺失单会话约束")
        index_sql = ''.join((index_row[0] or '').lower().split())
        expected_index_sql = (
            "createuniqueindexone_active_sessiononsessions((1))"
            "wherestatein('monitoring','paused','resting')"
        )
        if index_sql != expected_index_sql:
            raise ValueError("数据库单会话约束损坏")
        plan_index = self.db.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' "
            "AND name='unique_plan_occurrence'"
        ).fetchone()
        plan_index_sql = ''.join(((plan_index[0] if plan_index else '') or '')
                                 .lower().split())
        expected_plan_index = (
            'createuniqueindexunique_plan_occurrenceontasks(plan_id,starts)'
            'whereplan_idisnotnull'
        )
        if plan_index_sql != expected_plan_index:
            raise ValueError("数据库计划执行唯一约束损坏")
        raw_settings = {row['key']: row['value'] for row in self.db.execute(
            "SELECT key,value FROM settings "
            "WHERE key IN ('images','test_evidence','retention_days')"
        )}
        missing_settings = {'images', 'test_evidence', 'retention_days'} - set(raw_settings)
        if missing_settings:
            raise ValueError(
                "数据库设置缺失：" + ', '.join(sorted(missing_settings))
            )
        try:
            images = json.loads(raw_settings['images'])
            test_evidence = json.loads(raw_settings['test_evidence'])
            retention_days = json.loads(raw_settings['retention_days'])
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("数据库设置损坏：不是有效JSON") from exc
        if not isinstance(images, bool):
            raise ValueError("数据库设置损坏：images必须为布尔值")
        if not isinstance(test_evidence, bool):
            raise ValueError("数据库设置损坏：test_evidence必须为布尔值")
        if type(retention_days) is not int or not 1 <= retention_days <= 3650:
            raise ValueError("数据库设置损坏：retention_days须为1至3650天")
        try:
            for row in self.db.execute("SELECT * FROM plans"):
                raw_days = json.loads(row['weekdays'])
                if (not isinstance(raw_days, list)
                        or any(type(day) is not int for day in raw_days)):
                    raise ValueError("星期数据无效")
                normalized = plan_values(
                    row['title'], row['local_time'], row['timezone'], row['recurrence'],
                    raw_days, row['break_every_minutes'], row['break_minutes'],
                    bool(row['allow_offscreen']) if row['allow_offscreen'] in (0, 1) else None)
                if tuple(raw_days) != normalized[4] or row['enabled'] not in (0, 1):
                    raise ValueError("计划枚举值无效")
                minutes_valid(row['minutes'])
                created = datetime.fromisoformat(row['created'])
                updated = datetime.fromisoformat(row['updated'])
                utc_text(created)
                utc_text(updated)
                if updated < created:
                    raise ValueError("计划更新时间早于创建时间")
            for row in self.db.execute(
                    "SELECT minutes,break_every_minutes,break_minutes,allow_offscreen FROM tasks"):
                minutes_valid(row['minutes'])
                break_values(row['break_every_minutes'], row['break_minutes'])
                if row['allow_offscreen'] not in (0, 1):
                    raise ValueError("任务离屏选项无效")
            for row in self.db.execute(
                    "SELECT minutes,break_every_minutes,break_minutes,allow_offscreen,"
                    "extended_minutes,break_seconds,score_version FROM sessions"):
                minutes_valid(row['minutes'])
                break_values(row['break_every_minutes'], row['break_minutes'])
                if row['allow_offscreen'] not in (0, 1):
                    raise ValueError("会话离屏选项无效")
                if (not math.isfinite(row['extended_minutes']) or row['extended_minutes'] < 0
                        or not math.isfinite(row['break_seconds']) or row['break_seconds'] < 0):
                    raise ValueError("会话时长数据无效")
                if row['score_version'] not in ('points_v1', SCORE_VERSION):
                    raise ValueError("会话计分版本无效")
            for row in self.db.execute("SELECT * FROM focus_incidents"):
                signals = json.loads(row['signals'])
                segments = json.loads(row['segments'])
                if (not isinstance(signals, list) or not signals
                        or any(signal not in FOCUS_SIGNALS for signal in signals)):
                    raise ValueError("合并事件信号无效")
                if not isinstance(segments, dict) or not set(segments).issubset(FOCUS_SIGNALS):
                    raise ValueError("合并事件区间无效")
                if (not math.isfinite(row['started']) or row['started'] < 0
                        or (row['ended'] is not None and
                            (not math.isfinite(row['ended']) or row['ended'] < row['started']))
                        or not math.isfinite(row['violation_seconds'])
                        or row['violation_seconds'] < 0):
                    raise ValueError("合并事件时长无效")
            for row in self.db.execute("SELECT event_id,kind,offset,signals FROM evidence"):
                signals = json.loads(row['signals'])
                if (not row['event_id'] or row['kind'] not in
                        ('legacy', 'start', 'sample', 'end', 'event')
                        or (row['offset'] is not None and
                            (not math.isfinite(row['offset']) or row['offset'] < 0))
                        or not isinstance(signals, list)
                        or any(signal not in FOCUS_SIGNALS for signal in signals)):
                    raise ValueError("事件图片元数据无效")
            for row in self.db.execute('SELECT sequence,started,ended,seconds FROM session_work'):
                a, b = datetime.fromisoformat(row['started']), datetime.fromisoformat(row['ended'])
                utc_text(a)
                utc_text(b)
                if (type(row['sequence']) is not int or row['sequence'] < 0
                        or not math.isfinite(row['seconds']) or row['seconds'] < 0
                        or not math.isclose((b-a).total_seconds(), row['seconds'],
                                            rel_tol=1e-9, abs_tol=2e-6)):
                    raise ValueError('历史计时时段无效')
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"数据库计划数据损坏：{exc}") from exc

    def _create_schema(self):
        applied = utc_text(utc_now())
        try:
            self.db.executescript("""
                BEGIN IMMEDIATE;
                CREATE TABLE plans(id TEXT PRIMARY KEY, title TEXT NOT NULL,
                    local_time TEXT NOT NULL, timezone TEXT NOT NULL,
                    recurrence TEXT NOT NULL, weekdays TEXT NOT NULL,
                    minutes REAL NOT NULL, break_every_minutes REAL,
                    break_minutes REAL, allow_offscreen INTEGER NOT NULL DEFAULT 0,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created TEXT NOT NULL, updated TEXT NOT NULL);
                CREATE TABLE tasks(id TEXT PRIMARY KEY, title TEXT NOT NULL,
                    starts TEXT NOT NULL, minutes REAL NOT NULL, state TEXT NOT NULL DEFAULT 'pending',
                    error TEXT, plan_id TEXT REFERENCES plans(id) ON DELETE SET NULL,
                    break_every_minutes REAL, break_minutes REAL,
                    allow_offscreen INTEGER NOT NULL DEFAULT 0);
                CREATE UNIQUE INDEX unique_plan_occurrence ON tasks(plan_id,starts)
                    WHERE plan_id IS NOT NULL;
                CREATE TABLE sessions(id TEXT PRIMARY KEY, title TEXT NOT NULL,
                    task_id TEXT UNIQUE REFERENCES tasks(id), started TEXT NOT NULL,
                    updated TEXT NOT NULL, ended TEXT, state TEXT NOT NULL,
                    minutes REAL NOT NULL, identity TEXT NOT NULL, policy TEXT NOT NULL,
                    error TEXT, resource_summary TEXT, break_every_minutes REAL,
                    break_minutes REAL, allow_offscreen INTEGER NOT NULL DEFAULT 0,
                    extended_minutes REAL NOT NULL DEFAULT 0,
                    break_seconds REAL NOT NULL DEFAULT 0,
                    score_version TEXT NOT NULL DEFAULT 'time_ratio_v2');
                CREATE UNIQUE INDEX one_active_session ON sessions((1))
                    WHERE state IN ('monitoring','paused','resting');
                CREATE TABLE totals(session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
                    signal TEXT, eligible REAL NOT NULL, valid REAL NOT NULL,
                    violation REAL NOT NULL, PRIMARY KEY(session_id,signal));
                CREATE TABLE actions(id TEXT PRIMARY KEY,
                    session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
                    event_id TEXT NOT NULL, rule_id TEXT NOT NULL, kind TEXT NOT NULL,
                    offset REAL NOT NULL, points REAL NOT NULL, reason TEXT,
                    delivery_state TEXT NOT NULL, delivery_attempted TEXT,
                    delivery_error TEXT);
                CREATE TABLE evidence(id TEXT PRIMARY KEY,
                    session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
                    event_id TEXT NOT NULL, rule_id TEXT, kind TEXT NOT NULL,
                    offset REAL, signals TEXT NOT NULL,
                    path TEXT NOT NULL, sha256 TEXT, error TEXT, expired_at TEXT);
                CREATE INDEX evidence_event ON evidence(session_id,event_id,offset,id);
                CREATE INDEX evidence_unexpired_session ON evidence(session_id)
                    WHERE expired_at IS NULL;
                CREATE TABLE focus_incidents(id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    started REAL NOT NULL, ended REAL, signals TEXT NOT NULL,
                    segments TEXT NOT NULL, violation_seconds REAL NOT NULL,
                    alert_offset REAL);
                CREATE TABLE settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE session_work(
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL,
                    started TEXT NOT NULL, ended TEXT NOT NULL, seconds REAL NOT NULL,
                    PRIMARY KEY(session_id,sequence));
                CREATE INDEX session_work_dates ON session_work(ended,started);
                CREATE INDEX sessions_history_started ON sessions(started);
                CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY,
                    applied TEXT NOT NULL, description TEXT NOT NULL);
                INSERT INTO settings VALUES('images','false'),
                    ('test_evidence','false'),('retention_days','30');
            """)
            self.db.executemany(
                "INSERT INTO schema_migrations VALUES(?,?,?)",
                ((1, applied, 'initial product schema'),
                 (2, applied, 'versioned migration ledger'),
                 (3, applied, 'persist task outcome details'),
                 (4, applied, 'persist session interruption details'),
                 (5, applied, 'track reminder delivery'),
                 (6, applied, 'persist bounded session resource summary'),
                 (7, applied, 'add recurring plan templates and occurrence links'),
                 (8, applied, 'merge focus incidents and version time-ratio scoring'),
                 (9, applied, 'store ordered multi-frame test evidence'),
                 (10, applied, 'persist work intervals; retain logs independently of screenshots')),
            )
            self.db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self.db.commit()
        except sqlite3.Error:
            self.db.rollback()
            raise

    def _migrate_v1_to_v2(self):
        applied = utc_text(utc_now())
        try:
            self.db.execute("BEGIN IMMEDIATE")
            self.db.execute("CREATE TABLE IF NOT EXISTS schema_migrations("
                            "version INTEGER PRIMARY KEY, applied TEXT NOT NULL, "
                            "description TEXT NOT NULL)")
            self.db.execute("INSERT OR IGNORE INTO schema_migrations VALUES(?,?,?)",
                            (1, applied, 'legacy schema observed during v2 migration'))
            self.db.execute("INSERT OR IGNORE INTO schema_migrations VALUES(?,?,?)",
                            (2, applied, 'versioned migration ledger'))
            self.db.execute("PRAGMA user_version=2")
            self.db.commit()
        except sqlite3.Error:
            self.db.rollback()
            raise

    def _migrate_v2_to_v3(self):
        applied = utc_text(utc_now())
        try:
            self.db.execute("BEGIN IMMEDIATE")
            self.db.execute("ALTER TABLE tasks ADD COLUMN error TEXT")
            self.db.execute("INSERT INTO schema_migrations VALUES(?,?,?)",
                            (3, applied, 'persist task outcome details'))
            self.db.execute("PRAGMA user_version=3")
            self.db.commit()
        except sqlite3.Error:
            self.db.rollback()
            raise

    def _migrate_v3_to_v4(self):
        applied = utc_text(utc_now())
        try:
            self.db.execute("BEGIN IMMEDIATE")
            self.db.execute("ALTER TABLE sessions ADD COLUMN error TEXT")
            self.db.execute("INSERT INTO schema_migrations VALUES(?,?,?)",
                            (4, applied, 'persist session interruption details'))
            self.db.execute("PRAGMA user_version=4")
            self.db.commit()
        except sqlite3.Error:
            self.db.rollback()
            raise

    def _migrate_v4_to_v5(self):
        applied = utc_text(utc_now())
        try:
            self.db.execute("BEGIN IMMEDIATE")
            self.db.execute("ALTER TABLE actions ADD COLUMN delivery_state TEXT NOT NULL "
                            "DEFAULT 'not_applicable'")
            self.db.execute("ALTER TABLE actions ADD COLUMN delivery_attempted TEXT")
            self.db.execute("ALTER TABLE actions ADD COLUMN delivery_error TEXT")
            self.db.execute("UPDATE actions SET delivery_state='unknown' WHERE kind='alert'")
            self.db.execute("INSERT INTO schema_migrations VALUES(?,?,?)",
                            (5, applied, 'track reminder delivery'))
            self.db.execute("PRAGMA user_version=5")
            self.db.commit()
        except sqlite3.Error:
            self.db.rollback()
            raise

    def _migrate_v5_to_v6(self):
        applied = utc_text(utc_now())
        try:
            self.db.execute("BEGIN IMMEDIATE")
            self.db.execute("ALTER TABLE sessions ADD COLUMN resource_summary TEXT")
            self.db.execute("INSERT INTO schema_migrations VALUES(?,?,?)",
                            (6, applied, 'persist bounded session resource summary'))
            self.db.execute("PRAGMA user_version=6")
            self.db.commit()
        except sqlite3.Error:
            self.db.rollback()
            raise

    def _migrate_v6_to_v7(self):
        applied = utc_text(utc_now())
        try:
            self.db.execute("BEGIN IMMEDIATE")
            self.db.execute(
                "CREATE TABLE plans(id TEXT PRIMARY KEY, title TEXT NOT NULL, "
                "local_time TEXT NOT NULL, timezone TEXT NOT NULL, "
                "recurrence TEXT NOT NULL, weekdays TEXT NOT NULL, "
                "minutes REAL NOT NULL, break_every_minutes REAL, break_minutes REAL, "
                "allow_offscreen INTEGER NOT NULL DEFAULT 0, "
                "enabled INTEGER NOT NULL DEFAULT 1, created TEXT NOT NULL, updated TEXT NOT NULL)"
            )
            self.db.execute(
                "ALTER TABLE tasks ADD COLUMN plan_id TEXT REFERENCES plans(id) ON DELETE SET NULL"
            )
            self.db.execute("ALTER TABLE tasks ADD COLUMN break_every_minutes REAL")
            self.db.execute("ALTER TABLE tasks ADD COLUMN break_minutes REAL")
            self.db.execute(
                "ALTER TABLE tasks ADD COLUMN allow_offscreen INTEGER NOT NULL DEFAULT 0"
            )
            self.db.execute("ALTER TABLE sessions ADD COLUMN break_every_minutes REAL")
            self.db.execute("ALTER TABLE sessions ADD COLUMN break_minutes REAL")
            self.db.execute(
                "ALTER TABLE sessions ADD COLUMN allow_offscreen INTEGER NOT NULL DEFAULT 0"
            )
            self.db.execute(
                "ALTER TABLE sessions ADD COLUMN extended_minutes REAL NOT NULL DEFAULT 0"
            )
            self.db.execute(
                "ALTER TABLE sessions ADD COLUMN break_seconds REAL NOT NULL DEFAULT 0"
            )
            self.db.execute("DROP INDEX one_active_session")
            self.db.execute(
                "CREATE UNIQUE INDEX one_active_session ON sessions((1)) "
                "WHERE state IN ('monitoring','paused','resting')"
            )
            self.db.execute(
                "CREATE UNIQUE INDEX unique_plan_occurrence ON tasks(plan_id,starts) "
                "WHERE plan_id IS NOT NULL"
            )
            self.db.execute("INSERT INTO schema_migrations VALUES(?,?,?)",
                            (7, applied, 'add recurring plan templates and occurrence links'))
            self.db.execute("PRAGMA user_version=7")
            self.db.commit()
        except sqlite3.Error:
            self.db.rollback()
            raise

    def _migrate_v7_to_v8(self):
        applied = utc_text(utc_now())
        try:
            self.db.execute("BEGIN IMMEDIATE")
            self.db.execute(
                "ALTER TABLE sessions ADD COLUMN score_version TEXT NOT NULL "
                "DEFAULT 'points_v1'"
            )
            self.db.execute(
                "CREATE TABLE focus_incidents(id TEXT PRIMARY KEY, "
                "session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, "
                "started REAL NOT NULL, ended REAL, signals TEXT NOT NULL, "
                "segments TEXT NOT NULL, violation_seconds REAL NOT NULL, "
                "alert_offset REAL)"
            )
            self.db.execute("INSERT INTO schema_migrations VALUES(?,?,?)",
                            (8, applied,
                             'merge focus incidents and version time-ratio scoring'))
            self.db.execute("PRAGMA user_version=8")
            self.db.commit()
        except sqlite3.Error:
            self.db.rollback()
            raise

    def _migrate_v8_to_v9(self):
        applied = utc_text(utc_now())
        try:
            self.db.execute("BEGIN IMMEDIATE")
            self.db.execute("ALTER TABLE evidence RENAME TO evidence_v8")
            self.db.execute(
                "CREATE TABLE evidence(id TEXT PRIMARY KEY, "
                "session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE, "
                "event_id TEXT NOT NULL, rule_id TEXT, kind TEXT NOT NULL, "
                "offset REAL, signals TEXT NOT NULL, path TEXT NOT NULL, "
                "sha256 TEXT, error TEXT)"
            )
            self.db.execute(
                "INSERT INTO evidence(id,session_id,event_id,rule_id,kind,offset,signals,"
                "path,sha256,error) SELECT id,session_id,id,NULL,'legacy',NULL,'[]',"
                "path,sha256,error FROM evidence_v8"
            )
            self.db.execute("DROP TABLE evidence_v8")
            self.db.execute(
                "CREATE INDEX evidence_event ON evidence(session_id,event_id,offset,id)"
            )
            self.db.execute(
                "INSERT OR IGNORE INTO settings(key,value) VALUES('test_evidence','false')"
            )
            self.db.execute("INSERT OR REPLACE INTO schema_migrations VALUES(?,?,?)",
                            (9, applied, 'store ordered multi-frame test evidence'))
            self.db.execute("PRAGMA user_version=9")
            self.db.commit()
        except sqlite3.Error:
            self.db.rollback()
            raise

    def _migrate_v9_to_v10(self):
        # Old totals remain intact. No invented pause times or daily splits.
        try:
            self.db.execute('BEGIN IMMEDIATE')
            self.db.execute(
                'CREATE TABLE IF NOT EXISTS session_work('
                'session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, '
                'sequence INTEGER NOT NULL, '
                'started TEXT NOT NULL, ended TEXT NOT NULL, seconds REAL NOT NULL, '
                'PRIMARY KEY(session_id,sequence))')
            self.db.execute('CREATE INDEX IF NOT EXISTS session_work_dates '
                            'ON session_work(ended,started)')
            self.db.execute('CREATE INDEX IF NOT EXISTS sessions_history_started ON sessions(started)')
            columns = {row[1] for row in self.db.execute('PRAGMA table_info(evidence)')}
            if 'expired_at' not in columns:
                self.db.execute('ALTER TABLE evidence ADD COLUMN expired_at TEXT')
            self.db.execute('CREATE INDEX IF NOT EXISTS evidence_unexpired_session '
                            'ON evidence(session_id) WHERE expired_at IS NULL')
            self.db.execute('INSERT OR REPLACE INTO schema_migrations VALUES(?,?,?)',
                            (10, utc_text(utc_now()),
                             'persist work intervals; retain logs independently of screenshots'))
            self.db.execute('PRAGMA user_version=10')
            self.db.commit()
        except sqlite3.Error:
            self.db.rollback()
            raise

    def close(self):
        self.db.close()

    def setting(self, key: str):
        row = self.db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        if row is None:
            raise ValueError("未知设置")
        return json.loads(row[0])

    def set_images(self, enabled: bool):
        with self.db:
            self.db.execute("UPDATE settings SET value=? WHERE key='images'", (json.dumps(enabled),))

    def set_test_evidence(self, enabled: bool):
        with self.db:
            self.db.execute(
                "UPDATE settings SET value=? WHERE key='test_evidence'",
                (json.dumps(enabled),),
            )

    def add_plan(self, title: str, local_time: str, timezone_name: str,
                 recurrence: str, minutes: float, weekdays=(),
                 break_every_minutes: float | None = None,
                 break_minutes: float | None = None,
                 allow_offscreen: bool = False, now: datetime | None = None) -> str:
        values = plan_values(title, local_time, timezone_name, recurrence, weekdays,
                             break_every_minutes, break_minutes, allow_offscreen)
        identifier = str(uuid4())
        stamp = utc_text(now or utc_now())
        title, local_time, timezone_name, recurrence, weekdays, every, rest, offscreen = values
        with self.db:
            self.db.execute(
                "INSERT INTO plans VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (identifier, title, local_time, timezone_name, recurrence,
                 json.dumps(weekdays), minutes_valid(minutes), every, rest,
                 int(offscreen), 1, stamp, stamp),
            )
        return identifier

    def plans(self):
        result = []
        for row in self.db.execute("SELECT * FROM plans ORDER BY title,id"):
            item = dict(row)
            item['weekdays'] = tuple(json.loads(item['weekdays']))
            item['allow_offscreen'] = bool(item['allow_offscreen'])
            item['enabled'] = bool(item['enabled'])
            result.append(item)
        return result

    def update_plan(self, identifier: str, title: str, local_time: str,
                    timezone_name: str, recurrence: str, minutes: float, weekdays=(),
                    break_every_minutes: float | None = None,
                    break_minutes: float | None = None,
                    allow_offscreen: bool = False, now: datetime | None = None):
        values = plan_values(title, local_time, timezone_name, recurrence, weekdays,
                             break_every_minutes, break_minutes, allow_offscreen)
        title, local_time, timezone_name, recurrence, weekdays, every, rest, offscreen = values
        stamp = now or utc_now()
        with self.db:
            self._delete_future_occurrences(identifier, stamp)
            changed = self.db.execute(
                "UPDATE plans SET title=?,local_time=?,timezone=?,recurrence=?,weekdays=?,"
                "minutes=?,break_every_minutes=?,break_minutes=?,allow_offscreen=?,updated=? "
                "WHERE id=?",
                (title, local_time, timezone_name, recurrence, json.dumps(weekdays),
                 minutes_valid(minutes), every, rest, int(offscreen),
                 utc_text(stamp), identifier),
            ).rowcount
        if not changed:
            raise ValueError("计划不存在")

    def set_plan_enabled(self, identifier: str, enabled: bool,
                         now: datetime | None = None):
        if type(enabled) is not bool:
            raise ValueError("计划启用状态须为布尔值")
        stamp = now or utc_now()
        with self.db:
            if not enabled:
                self._delete_future_occurrences(identifier, stamp)
            changed = self.db.execute(
                "UPDATE plans SET enabled=?,updated=? WHERE id=?",
                (int(enabled), utc_text(stamp), identifier),
            ).rowcount
        if not changed:
            raise ValueError("计划不存在")

    def delete_plan(self, identifier: str, now: datetime | None = None):
        with self.db:
            self._delete_future_occurrences(identifier, now or utc_now())
            changed = self.db.execute("DELETE FROM plans WHERE id=?", (identifier,)).rowcount
        if not changed:
            raise ValueError("计划不存在")

    def _delete_future_occurrences(self, identifier: str, now: datetime):
        self.db.execute(
            "DELETE FROM tasks WHERE plan_id=? AND state='pending' AND starts>=?",
            (identifier, utc_text(now)),
        )

    @staticmethod
    def _plan_occurrence(day, local_time: str, zone: ZoneInfo):
        wall_time = datetime.strptime(local_time, '%H:%M').time()
        naive = datetime.combine(day, wall_time)
        local = naive.replace(tzinfo=zone, fold=0)
        instant = local.astimezone(timezone.utc)
        if instant.astimezone(zone).replace(tzinfo=None) != naive:
            return instant, '本地计划时间因夏令时切换不存在，已跳过本次执行'
        return instant, None

    def materialize_plans(self, now: datetime, horizon_days: int = 14) -> dict:
        """Create durable task occurrences through a short rolling horizon."""
        utc_text(now)
        if type(horizon_days) is not int or not 1 <= horizon_days <= 366:
            raise ValueError("计划生成范围须为1至366天")
        horizon = now.astimezone(timezone.utc) + timedelta(days=horizon_days)
        created_count = failed_count = 0
        with self.db:
            for plan in self.plans():
                if not plan['enabled']:
                    continue
                zone = ZoneInfo(plan['timezone'])
                updated = datetime.fromisoformat(plan['updated'])
                start_day = updated.astimezone(zone).date()
                last = self.db.execute(
                    "SELECT max(starts) FROM tasks WHERE plan_id=? "
                    "AND (starts<? OR state='pending')",
                    (plan['id'], utc_text(updated)),
                ).fetchone()[0]
                if last:
                    start_day = max(
                        start_day,
                        datetime.fromisoformat(last).astimezone(zone).date() + timedelta(days=1),
                    )
                end_day = horizon.astimezone(zone).date()
                day = start_day
                while day <= end_day:
                    if day.weekday() in plan['weekdays']:
                        occurrence, error = self._plan_occurrence(
                            day, plan['local_time'], zone)
                        if updated <= occurrence <= horizon:
                            changed = self.db.execute(
                                "INSERT OR IGNORE INTO tasks("
                                "id,title,starts,minutes,state,error,plan_id,"
                                "break_every_minutes,break_minutes,allow_offscreen) "
                                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                                (str(uuid4()), plan['title'], utc_text(occurrence),
                                 plan['minutes'], 'failed' if error else 'pending', error,
                                 plan['id'], plan['break_every_minutes'],
                                 plan['break_minutes'], int(plan['allow_offscreen'])),
                            ).rowcount
                            created_count += changed
                            failed_count += changed if error else 0
                    day += timedelta(days=1)
        return {'created': created_count, 'failed': failed_count}

    def add_task(self, title: str, starts: datetime, minutes: float) -> str:
        if not title.strip():
            raise ValueError("任务名不能为空")
        identifier = str(uuid4())
        with self.db:
            self.db.execute("INSERT INTO tasks(id,title,starts,minutes) VALUES(?,?,?,?)",
                            (identifier, title.strip(), utc_text(starts), minutes_valid(minutes)))
        return identifier

    def tasks(self):
        result = []
        for row in self.db.execute("SELECT * FROM tasks ORDER BY starts"):
            item = dict(row)
            item['allow_offscreen'] = bool(item['allow_offscreen'])
            result.append(item)
        return result

    def cancel_task(self, identifier: str):
        with self.db:
            changed = self.db.execute("UPDATE tasks SET state='cancelled',error=NULL "
                                      "WHERE id=? AND state IN ('pending','conflict','failed','missed')",
                                      (identifier,)).rowcount
            if not changed:
                raise ValueError("任务正在运行或已经结束")

    def retry_task(self, identifier: str, now: datetime):
        """Return a non-started terminal task to pending without creating a duplicate."""
        starts = utc_text(now)
        with self.db:
            if self.db.execute(
                "SELECT 1 FROM sessions WHERE state IN ('monitoring','paused','resting')"
            ).fetchone():
                raise ValueError("请先结束当前会话再重试任务")
            if self.db.execute(
                "SELECT 1 FROM sessions WHERE task_id=?", (identifier,)
            ).fetchone():
                raise ValueError("该任务已经建立过会话")
            changed = self.db.execute(
                "UPDATE tasks SET state='pending',starts=?,error=NULL "
                "WHERE id=? AND state IN ('conflict','failed','missed')",
                (starts, identifier),
            ).rowcount
            if not changed:
                raise ValueError("只有冲突、失败或错过的任务可以重试")

    def fail_task(self, identifier: str, reason: str):
        detail = str(reason).strip()[:500] or "启动失败"
        with self.db:
            changed = self.db.execute(
                "UPDATE tasks SET state='failed',error=? WHERE id=? AND state='pending'",
                (detail, identifier),
            ).rowcount
            if not changed:
                raise ValueError("任务已处理")

    def expire_pending_tasks(self, now: datetime, grace_seconds: float = 60) -> int:
        """Mark tasks older than the restart grace window; keep recent tasks runnable."""
        if not math.isfinite(grace_seconds) or grace_seconds < 0:
            raise ValueError("迟到宽限时间无效")
        cutoff = utc_text(now - timedelta(seconds=grace_seconds))
        with self.db:
            return self.db.execute(
                "UPDATE tasks SET state='missed',error=? WHERE state='pending' AND starts < ?",
                (f"程序启动时已超过{grace_seconds:g}秒补启动窗口", cutoff),
            ).rowcount

    def history(self, limit=None):
        rows = []
        query = (
                "SELECT id,title,started,ended,state,minutes,identity "
                "FROM sessions ORDER BY started DESC")
        parameters = ()
        if limit is not None:
            query += ' LIMIT ?'
            parameters = (limit,)
        for row in self.db.execute(query, parameters):
            item = dict(row)
            item['mode'] = _session_mode(row['identity'])
            rows.append(item)
        return rows

    def history_calendar(self, now, timezone_name):
        from observer.history import calendar_history
        return calendar_history(self.db, now, timezone_name)

    def today_summary(self, now: datetime, timezone_name: str) -> dict:
        utc_text(now)
        try:
            zone = ZoneInfo(timezone_name)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError("摘要时区无效") from exc
        local_day = now.astimezone(zone).date()
        local_start = datetime.combine(local_day, datetime.min.time(), tzinfo=zone)
        local_end = local_start + timedelta(days=1)
        start = utc_text(local_start)
        end = utc_text(local_end)
        planned, completed = self.db.execute(
            "SELECT count(*),sum(state='ended') FROM tasks WHERE starts>=? AND starts<?",
            (start, end),
        ).fetchone()
        session_rows = list(self.db.execute(
            "SELECT id,identity,policy,allow_offscreen,score_version FROM sessions "
            "WHERE started>=? AND started<?", (start, end)))
        session_ids = [row['id'] for row in session_rows]
        observed = break_seconds = legacy_deducted = violation_seconds = 0.
        required = set(HARD_SIGNALS)
        scoring_by_session = {}
        for row in session_rows:
            scoring_by_session[row['id']] = set(
                _session_scoring_signals(
                    row['identity'], row['policy'], bool(row['allow_offscreen']),
                    row['score_version']))
            required.update(scoring_by_session[row['id']])
        valid = {signal: 0. for signal in required}
        eligible = {signal: 0. for signal in required}
        v2_valid = {signal: 0. for signal in required}
        v2_eligible = {signal: 0. for signal in required}
        v2_ids = {row['id'] for row in session_rows
                  if row['score_version'] == SCORE_VERSION}
        legacy_ids = set(session_ids) - v2_ids
        if session_ids:
            placeholders = ','.join('?' for _ in session_ids)
            for row in self.db.execute(
                f"SELECT session_id,signal,eligible,valid FROM totals "
                f"WHERE session_id IN ({placeholders}) "
                "AND signal IN ('away','phone_held','attention_away')",
                session_ids,
            ):
                if row['signal'] in scoring_by_session[row['session_id']]:
                    eligible[row['signal']] += float(row['eligible'] or 0)
                    valid[row['signal']] += float(row['valid'] or 0)
                    if row['session_id'] in v2_ids:
                        v2_eligible[row['signal']] += float(row['eligible'] or 0)
                        v2_valid[row['signal']] += float(row['valid'] or 0)
            observed = eligible['away']
            break_seconds = float(self.db.execute(
                f"SELECT sum(break_seconds) FROM sessions WHERE id IN ({placeholders})",
                session_ids,
            ).fetchone()[0] or 0)
            if legacy_ids:
                legacy_placeholders = ','.join('?' for _ in legacy_ids)
                legacy_deducted = float(self.db.execute(
                    f"SELECT sum(points) FROM actions WHERE kind='penalty' "
                    f"AND session_id IN ({legacy_placeholders})",
                    tuple(legacy_ids),
                ).fetchone()[0] or 0)
            if v2_ids:
                v2_placeholders = ','.join('?' for _ in v2_ids)
                violation_seconds = float(self.db.execute(
                    f"SELECT sum(violation_seconds) FROM focus_incidents "
                    f"WHERE session_id IN ({v2_placeholders})",
                    tuple(v2_ids),
                ).fetchone()[0] or 0)
        coverage = None
        if all(eligible[signal] > 0 for signal in required):
            coverage = min(valid[signal] / eligible[signal] for signal in required)
        v2_required = set().union(*(scoring_by_session[sid] for sid in v2_ids)) \
            if v2_ids else set()
        scoring_seconds = v2_eligible.get('away', 0.)
        v2_coverage = None
        if v2_required and all(v2_eligible[signal] > 0 for signal in v2_required):
            v2_coverage = min(
                v2_valid[signal] / v2_eligible[signal] for signal in v2_required
            )
        score = None
        if scoring_seconds > 0 and v2_coverage is not None and v2_coverage >= .8:
            score = round(max(0., scoring_seconds - violation_seconds)
                          / scoring_seconds * 100, 1)
        return {
            'date': local_day.isoformat(),
            'planned': planned,
            'completed': int(completed or 0),
            'observed_seconds': observed,
            'break_seconds': break_seconds,
            'deducted': legacy_deducted,
            'legacy_deducted': legacy_deducted,
            'violation_seconds': violation_seconds,
            'scoring_seconds': scoring_seconds,
            'score': score,
            'coverage': coverage,
        }

    def summary(self, identifier: str):
        row = self.db.execute("SELECT * FROM sessions WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise ValueError("未找到会话")
        totals = {r['signal']: dict(r) for r in self.db.execute(
            "SELECT * FROM totals WHERE session_id=?", (identifier,))}
        scoring_signals = _session_scoring_signals(
            row['identity'], row['policy'], bool(row['allow_offscreen']),
            row['score_version'])
        coverage = min((totals.get(s, {}).get('valid', 0) /
                        max(totals.get(s, {}).get('eligible', 0), 1e-9)
                        for s in scoring_signals))
        actions = [dict(r) for r in self.db.execute(
            "SELECT * FROM actions WHERE session_id=? ORDER BY offset,rowid", (identifier,))]
        points = sum(a['points'] for a in actions)
        incidents = []
        for incident_row in self.db.execute(
                "SELECT * FROM focus_incidents WHERE session_id=? ORDER BY started,id",
                (identifier,)):
            incident = dict(incident_row)
            incident['signals'] = json.loads(incident['signals'])
            incident['segments'] = json.loads(incident['segments'])
            incidents.append(incident)
        violation_seconds = sum(
            float(incident['violation_seconds']) for incident in incidents
        )
        scoring_seconds = float(totals.get('away', {}).get('eligible', 0) or 0)
        warnings = []
        mode = _session_mode(row['identity'])
        if mode == 'unknown':
            warnings.append('会话身份记录损坏')
        resource_summary = None
        if row['resource_summary']:
            resource_summary = _resource_summary(row['resource_summary'])
            if resource_summary is None:
                warnings.append('资源汇总损坏')
        if row['score_version'] == SCORE_VERSION:
            score = (round(max(0., scoring_seconds - violation_seconds)
                           / scoring_seconds * 100, 1)
                     if scoring_seconds > 0 and coverage >= .8 else None)
            deducted = 0.0
        else:
            score = max(0, 100-points) if coverage >= .8 else None
            deducted = points
        return {**dict(row), "mode": mode, "data_warnings": warnings,
                "resource_summary": resource_summary,
                "coverage": coverage, "score": score, "deducted": deducted,
                "scoring_seconds": scoring_seconds,
                "violation_seconds": violation_seconds,
                "compliant_seconds": max(0., scoring_seconds - violation_seconds),
                "focus_incidents": incidents, "totals": totals, "actions": actions,
                "evidence": [dict(r) for r in self.db.execute(
                    "SELECT * FROM evidence WHERE session_id=? "
                    "ORDER BY COALESCE(offset,0),id", (identifier,))]}

    def data_status(self):
        sessions = self.db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        tasks = self.db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        plans = self.db.execute("SELECT COUNT(*) FROM plans").fetchone()[0]
        image_count = image_bytes = evidence_errors = 0
        sizes = self._evidence_file_sizes()
        for row in self.db.execute("SELECT path,sha256,error FROM evidence WHERE expired_at IS NULL"):
            if row['error'] or not row['path'] or not row['sha256']:
                evidence_errors += 1
                continue
            try:
                size = sizes.get(row['path'])
                if size is None:
                    size = self.safe_image_path(row['path']).stat().st_size
            except (OSError, ValueError):
                evidence_errors += 1
            else:
                image_count += 1
                image_bytes += size
        return {
            'sessions': sessions,
            'tasks': tasks,
            'plans': plans,
            'images': image_count,
            'image_bytes': image_bytes,
            'evidence_errors': evidence_errors,
            'images_enabled': self.setting('images'),
            'test_evidence_enabled': self.setting('test_evidence'),
            'retention_days': self.setting('retention_days'),
        }

    def _evidence_file_sizes(self):
        """Read flat evidence files together, keeping links on the validated path."""
        base = self.root / 'evidence'
        if base.resolve() != base:
            return {}
        sizes = {}
        try:
            with os.scandir(base) as entries:
                for entry in entries:
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    if (stat.S_ISREG(info.st_mode) and not
                            getattr(info, 'st_file_attributes', 0)
                            & stat.FILE_ATTRIBUTE_REPARSE_POINT):
                        sizes[f'evidence/{entry.name}'] = info.st_size
        except OSError:
            return {}
        return sizes

    def recover_evidence(self):
        """Reconcile crash leftovers after the caller has acquired RuntimeLock."""
        recovered = failed = 0
        rows = list(self.db.execute("SELECT id,path,sha256,error FROM evidence WHERE expired_at IS NULL"))
        for row in rows:
            try:
                if not row['path']:
                    if row['error']:
                        continue
                    raise ValueError('证据索引缺少文件路径')
                target = self.safe_image_path(row['path'])
                temporary = self.safe_image_path(row['path'] + '.tmp')
                if row['error'] == 'pending':
                    if target.is_file():
                        digest = _file_sha256(target)
                        temporary.unlink(missing_ok=True)
                        with self.db:
                            self.db.execute(
                                "UPDATE evidence SET sha256=?,error=NULL WHERE id=?",
                                (digest, row['id']),
                            )
                        recovered += 1
                    else:
                        temporary.unlink(missing_ok=True)
                        raise ValueError('事件图写入在完成前中断')
                elif row['error']:
                    temporary.unlink(missing_ok=True)
                elif not target.is_file():
                    raise ValueError('事件图文件缺失')
                elif not row['sha256'] or _file_sha256(target) != row['sha256']:
                    raise ValueError('事件图哈希不一致')
                else:
                    temporary.unlink(missing_ok=True)
            except (OSError, TypeError, ValueError) as exc:
                detail = f"证据恢复失败：{str(exc).strip()[:450] or type(exc).__name__}"
                with self.db:
                    self.db.execute(
                        "UPDATE evidence SET error=? WHERE id=?",
                        (detail, row['id']),
                    )
                failed += 1
        return {'recovered': recovered, 'failed': failed}

    def mark_notification(self, action_id: str, delivered: bool,
                          error: str | None = None, now: datetime | None = None) -> bool:
        state = 'delivered' if delivered else 'failed'
        detail = None if delivered else (str(error).strip()[:500] or '提醒呈现失败')
        attempted = utc_text(now or utc_now())
        with self.db:
            row = self.db.execute(
                "SELECT kind,delivery_state FROM actions WHERE id=?", (action_id,)
            ).fetchone()
            if row is None:
                raise ValueError("未找到提醒事件")
            if row['kind'] != ActionKind.ALERT.value:
                raise ValueError("该事件不需要提醒交付")
            if row['delivery_state'] == 'delivered':
                return False
            self.db.execute(
                "UPDATE actions SET delivery_state=?,delivery_attempted=?,delivery_error=? "
                "WHERE id=?",
                (state, attempted, detail, action_id),
            )
        return True

    def interrupt_abandoned(self):
        """Only call after acquiring the runtime lock. Do not bridge process clocks."""
        attempted = utc_text(utc_now())
        with self.db:
            self.db.execute(
                "UPDATE actions SET delivery_state='failed',delivery_attempted=?,delivery_error=? "
                "WHERE delivery_state='pending' AND session_id IN "
                "(SELECT id FROM sessions WHERE state IN ('monitoring','paused','resting'))",
                (attempted, "程序退出前未确认提醒呈现"))
            self.db.execute("UPDATE tasks SET state='interrupted',error=? WHERE id IN "
                            "(SELECT task_id FROM sessions WHERE state IN ('monitoring','paused','resting'))",
                            ("程序上次退出时会话未正常结束",))
            self.db.execute("UPDATE sessions SET state='interrupted',ended=updated,error=? "
                            "WHERE state IN ('monitoring','paused','resting')",
                            ("程序上次退出时会话未正常结束",))

    def delete_session(self, identifier: str):
        # File removal precedes row removal: failures leave a retryable index.
        with self.db:
            row = self.db.execute("SELECT state FROM sessions WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise ValueError("未找到会话")
            if row[0] in ('monitoring', 'paused', 'resting'):
                raise ValueError("请先结束会话")
            for item in self.db.execute("SELECT path FROM evidence WHERE session_id=?", (identifier,)):
                if item[0]:
                    path = self.safe_image_path(item[0])
                    self.safe_image_path(item[0]+'.tmp').unlink(missing_ok=True)
                    path.unlink(missing_ok=True)
            self.db.execute("DELETE FROM sessions WHERE id=?", (identifier,))

    def delete_all(self) -> int:
        """Delete product records after validating every persisted evidence path."""
        if self.db.execute(
            "SELECT 1 FROM sessions WHERE state IN ('monitoring','paused','resting')"
        ).fetchone():
            raise ValueError("请先结束运行窗口中的会话")
        evidence_paths = [row[0] for row in self.db.execute(
            "SELECT path FROM evidence WHERE path != ''"
        )]
        # Reject the whole request before deleting anything when an index is unsafe.
        for relative in evidence_paths:
            self.safe_image_path(relative)
            self.safe_image_path(relative + '.tmp')
        identifiers = [row[0] for row in self.db.execute("SELECT id FROM sessions")]
        for identifier in identifiers:
            self.delete_session(identifier)
        with self.db:
            self.db.execute("DELETE FROM tasks")
            self.db.execute("DELETE FROM plans")
        return len(identifiers)

    def safe_image_path(self, relative: str) -> Path:
        base = self.root / "evidence"
        if base.resolve() != base:
            raise ValueError("证据目录不能通过链接指向其他位置")
        target = (self.root / relative).resolve()
        if not target.is_relative_to(base) or target == base:
            raise ValueError("证据路径越界")
        return target

    def purge(self, now: datetime):
        """Expire screenshots only. Sessions and text logs are retained."""
        cutoff = utc_text(now - timedelta(days=self.setting('retention_days')))
        rows = list(self.db.execute(
            "SELECT e.id,e.path FROM evidence e JOIN sessions s ON s.id=e.session_id "
            "WHERE e.expired_at IS NULL AND "
            "CASE WHEN e.offset IS NOT NULL THEN julianday(s.started)+e.offset/86400.0 "
            "ELSE julianday(COALESCE(s.ended,s.updated)) END < julianday(?)", (cutoff,)))
        removed = 0
        for row in rows:
            try:
                if row['path']:
                    self.safe_image_path(row['path']).unlink(missing_ok=True)
                    self.safe_image_path(row['path'] + '.tmp').unlink(missing_ok=True)
            except (OSError, ValueError) as exc:
                detail = f"清理失败：{str(exc).strip()[:450] or type(exc).__name__}"
                with self.db:
                    self.db.execute(
                        "UPDATE evidence SET error=COALESCE(error,?) WHERE id=?",
                        (detail, row['id']),
                    )
            else:
                with self.db:
                    self.db.execute("UPDATE evidence SET path='',expired_at=? WHERE id=?",
                                    (utc_text(now), row['id']))
                removed += 1
        return removed

    def capture(self, session_id: str, event_id: str, frame, *,
                rule_id: str | None = None, kind: str = 'event',
                offset: float | None = None, signals=(), capture_id: str | None = None):
        if not self.setting('images'):
            return False
        identifier = capture_id or event_id
        if self.db.execute("SELECT 1 FROM evidence WHERE id=?", (identifier,)).fetchone():
            return False
        if kind not in ('legacy', 'start', 'sample', 'end', 'event'):
            raise ValueError('事件图片类型无效')
        if offset is not None and (not math.isfinite(offset) or offset < 0):
            raise ValueError('事件图片时间无效')
        signals = tuple(dict.fromkeys(signals))
        if any(signal not in FOCUS_SIGNALS for signal in signals):
            raise ValueError('事件图片信号无效')
        relative, digest, error = '', None, None
        try:
            if frame is None:
                raise ValueError("事件发生时没有新鲜图像")
            target = self.safe_image_path(f"evidence/{identifier}.jpg")
            target.parent.mkdir(exist_ok=True)
            import cv2
            try:
                ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            except cv2.error as exc:
                raise ValueError('证据图编码失败') from exc
            if not ok:
                raise ValueError("证据图编码失败")
            raw = encoded.tobytes()
            with os.scandir(target.parent) as images:
                existing_bytes = sum(entry.stat().st_size for entry in images
                                     if fnmatch(entry.name, '*.jpg'))
            if len(raw) + existing_bytes > 500 * 1024**2:
                raise ValueError("证据图配额已满")
            relative = target.relative_to(self.root).as_posix()
            # Persist intent first: a crash leaves a known, deletable path.
            with self.db:
                self.db.execute(
                    "INSERT INTO evidence(id,session_id,event_id,rule_id,kind,offset,signals,"
                    "path,sha256,error) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (identifier, session_id, event_id, rule_id, kind, offset,
                     json.dumps(signals, ensure_ascii=False), relative, None, 'pending'),
                )
            temporary = self.safe_image_path(relative+'.tmp')
            temporary.write_bytes(raw)
            temporary.replace(target)
            digest = hashlib.sha256(raw).hexdigest()
        except (OSError, ValueError) as exc:
            error = str(exc)
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO evidence(id,session_id,event_id,rule_id,kind,offset,"
                "signals,path,sha256,error) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (identifier, session_id, event_id, rule_id, kind, offset,
                 json.dumps(signals, ensure_ascii=False), relative, digest, error),
            )
        return error is None

    def reassign_evidence_event(self, session_id: str, previous_event_id: str,
                                event_id: str, rule_id: str):
        with self.db:
            self.db.execute(
                "UPDATE evidence SET event_id=?,rule_id=? "
                "WHERE session_id=? AND event_id=?",
                (event_id, rule_id, session_id, previous_event_id),
            )


class RuntimeLock:
    """OS releases the lock on process death; the file itself may remain."""
    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.file = (root / 'runtime.lock').open('a+b')
        self.file.write(b'0')
        self.file.flush()
        self.file.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            raise ValueError("已有 Observer 产品运行窗口") from exc

    def close(self):
        self.file.close()


class SessionService:
    def __init__(self, store: ProductStore):
        self.store = store
        self.identifier: str | None = None
        self.state = 'idle'
        self.rules = product_rule_configs()
        self.engine = RuleEngine(self.rules)
        self.last = None
        self.totals = {}
        self.work_seconds = 0.
        self.break_seconds = 0.
        self.break_every_seconds = None
        self.break_duration_seconds = None
        self.next_break_work = None
        self.break_deadline = None
        self.allow_offscreen = False
        self.total_minutes = 0.
        self.paused_remaining = None
        self.focus: FocusIncidentTracker | None = None

    def start(self, title: str, minutes: float, at: float, now: datetime,
              identity: dict, task_id: str | None = None,
              break_every_minutes: float | None = None,
              break_minutes: float | None = None,
              allow_offscreen: bool = False):
        if self.identifier:
            raise ValueError("已有会话，请先结束")
        if not title.strip() or not math.isfinite(at) or at < 0:
            raise ValueError("任务名或时间无效")
        minutes = minutes_valid(minutes)
        break_every_minutes, break_minutes = break_values(
            break_every_minutes, break_minutes)
        if type(allow_offscreen) is not bool:
            raise ValueError("允许离屏资料须为布尔值")
        identifier = str(uuid4())
        with self.store.db:
            if task_id and not self.store.db.execute(
                "UPDATE tasks SET state='running',error=NULL WHERE id=? AND state='pending'",
                (task_id,)).rowcount:
                raise ValueError("任务已处理")
            self.store.db.execute(
                "INSERT INTO sessions("
                "id,title,task_id,started,updated,ended,state,minutes,identity,policy,error,"
                "break_every_minutes,break_minutes,allow_offscreen,extended_minutes,break_seconds,"
                "score_version) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (identifier, title.strip(), task_id, utc_text(now), utc_text(now), None,
                 'monitoring', minutes, json.dumps(identity, ensure_ascii=False),
                 json.dumps([asdict(r) for r in self.rules]), None,
                 break_every_minutes, break_minutes, int(allow_offscreen), 0, 0,
                 SCORE_VERSION))
            self.store.db.execute('INSERT INTO session_work VALUES(?,0,?,?,0)',
                                  (identifier, utc_text(now), utc_text(now)))
        self.identifier, self.state = identifier, 'monitoring'
        self.started = self.last_at = self.flushed_at = at
        self.deadline = at + minutes * 60
        self.paused_remaining = None
        self.total_minutes = minutes
        self.work_seconds = self.break_seconds = 0.
        self.break_every_seconds = (break_every_minutes * 60
                                    if break_every_minutes is not None else None)
        self.break_duration_seconds = (break_minutes * 60
                                       if break_minutes is not None else None)
        self.next_break_work = self.break_every_seconds
        self.break_deadline = None
        self.allow_offscreen = allow_offscreen
        self.last = None
        self.engine = RuleEngine(self.rules)
        self.totals = {r.signal_name: [0., 0., 0.] for r in self.rules}
        scoring_signals = _session_scoring_signals(
            identity, [asdict(rule) for rule in self.rules], allow_offscreen, SCORE_VERSION
        )
        self.focus = FocusIncidentTracker(self.rules, at, scoring_signals)
        self.work_sequence = -1
        self._start_work_interval(at, now)
        return identifier

    def _start_work_interval(self, at, now):
        self.work_sequence += 1
        self.work_started_at = at
        self.work_started_utc = now
        self.work_interval_seconds = 0.

    def _account(self, at: float, current: Observation | None = None):
        delta = at - self.last_at
        if delta < 0 or not math.isfinite(delta):
            raise ValueError("时间不能倒退或无效")
        if self.state == 'monitoring':
            self.work_seconds += delta
            self.work_interval_seconds = at - self.work_started_at
            for name, values in self.totals.items():
                previous = self.last.signal(name).state if self.last else SignalState.UNKNOWN
                following = current.signal(name).state if current else SignalState.UNKNOWN
                values[0] += delta
                # Never assume an unobserved gap was compliant or a violation.
                if delta <= 2 and previous == following and previous in (
                        SignalState.COMPLIANT, SignalState.VIOLATION):
                    values[1] += delta
                    values[2] += delta * (previous == SignalState.VIOLATION)
        elif self.state == 'resting':
            self.break_seconds += delta
        self.last_at = at

    def _flush(self, now: datetime, actions=(), penalties=()):
        if not self.identifier:
            return
        points = {(p.event_id, p.at_monotonic): p.points for p in penalties}
        try:
            with self.store.db:
                self.store.db.execute(
                    "UPDATE sessions SET updated=?,state=?,break_seconds=? WHERE id=?",
                    (utc_text(now), self.state, self.break_seconds, self.identifier))
                self.store.db.execute(
                    'INSERT INTO session_work VALUES(?,?,?,?,?) '
                    'ON CONFLICT(session_id,sequence) DO UPDATE SET '
                    'ended=excluded.ended,seconds=excluded.seconds',
                    (self.identifier, self.work_sequence, utc_text(self.work_started_utc),
                     utc_text(self.work_started_utc + timedelta(seconds=self.work_interval_seconds)),
                     self.work_interval_seconds))
                for name, values in self.totals.items():
                    self.store.db.execute("INSERT OR REPLACE INTO totals VALUES(?,?,?,?,?)",
                                          (self.identifier, name, *values))
                for action in actions:
                    key = self.action_identifier(action)
                    delivery = 'pending' if action.kind is ActionKind.ALERT else 'not_applicable'
                    self.store.db.execute(
                        "INSERT OR IGNORE INTO actions("
                        "id,session_id,event_id,rule_id,kind,offset,points,reason,delivery_state) "
                        "VALUES(?,?,?,?,?,?,?,?,?)",
                        (key, self.identifier, action.event_id, action.rule_id, action.kind.value,
                         action.at_monotonic-self.started,
                         points.get((action.event_id, action.at_monotonic), 0.)
                         if action.kind is ActionKind.PENALTY else 0., action.reason, delivery))
                if self.focus is not None:
                    for incident in self.focus.rows():
                        self.store.db.execute(
                            "INSERT INTO focus_incidents("
                            "id,session_id,started,ended,signals,segments,violation_seconds,"
                            "alert_offset) VALUES(?,?,?,?,?,?,?,?) "
                            "ON CONFLICT(id) DO UPDATE SET ended=excluded.ended,"
                            "signals=excluded.signals,segments=excluded.segments,"
                            "violation_seconds=excluded.violation_seconds,"
                            "alert_offset=excluded.alert_offset",
                            (incident['id'], self.identifier, incident['started'],
                             incident['ended'], incident['signals'], incident['segments'],
                             incident['violation_seconds'], incident['alert_offset']),
                        )
        except sqlite3.Error:
            self.state = 'faulted'
            raise
        self.flushed_at = self.last_at

    @staticmethod
    def action_identifier(action) -> str:
        return f'{action.event_id}:{action.kind.value}:{action.at_monotonic:.9f}'

    def _process_rules(self, observation: Observation):
        result = self.engine.process(observation)
        if self.focus is None:
            return result.actions, result.penalties
        focus_actions = tuple(
            action for action in result.actions if action.rule_id in FOCUS_SIGNALS
        )
        ordinary_actions = tuple(
            action for action in result.actions if action.rule_id not in FOCUS_SIGNALS
        )
        merged_actions = self.focus.update(
            observation, focus_actions, self.engine.phases
        )
        ordinary_penalties = tuple(
            penalty for penalty in result.penalties if penalty.rule_id not in FOCUS_SIGNALS
        )
        return ordinary_actions + merged_actions, ordinary_penalties

    def process(self, observation: Observation, now: datetime):
        if not self.identifier or self.state != 'monitoring':
            return ()
        at = observation.monotonic_seconds
        if self.last is not None and at == self.last.monotonic_seconds:
            return ()
        signals = dict(observation.signals)
        signals['posture_deviation'] = unified_posture_signal(observation)
        product_observation = Observation(
            at, signals, captured_at_utc=observation.captured_at_utc)
        self._account(at, product_observation)
        rule_signals = dict(signals)
        if self.allow_offscreen:
            rule_signals['attention_away'] = SignalValue.exempt('本次允许离屏查看资料')
        actions, penalties = self._process_rules(Observation(
            at, rule_signals, captured_at_utc=observation.captured_at_utc))
        self.last = product_observation
        if actions or at-self.flushed_at >= 1:
            self._flush(now, actions, penalties)
        return actions

    def pause(self, at: float, now: datetime):
        if self.state != 'monitoring':
            raise ValueError("当前没有运行中的会话")
        self._account(at)
        actions, penalties = self._process_rules(Observation(at, {
            r.signal_name: SignalValue.exempt('会话暂停') for r in self.rules}))
        self.paused_remaining = max(0., self.deadline - at)
        self.state, self.last = 'paused', None
        self._flush(now, actions, penalties)

    def resume(self, at: float, now: datetime):
        if self.state != 'paused':
            raise ValueError("会话没有暂停")
        self._account(at)
        self.deadline = at + self.paused_remaining
        self.paused_remaining = None
        self.state, self.last = 'monitoring', None
        self._start_work_interval(at, now)
        self._flush(now)

    def remaining_seconds(self, at: float) -> float:
        if self.state == 'paused' and self.paused_remaining is not None:
            return self.paused_remaining
        return max(0., self.deadline - at) if self.identifier else 0.

    def break_due(self) -> bool:
        return (self.state == 'monitoring' and self.next_break_work is not None
                and self.work_seconds >= self.next_break_work)

    def start_break(self, at: float, now: datetime):
        if not self.break_due():
            raise ValueError("当前未到计划休息时间")
        self._account(at)
        actions, penalties = self._process_rules(Observation(at, {
            rule.signal_name: SignalValue.exempt('计划休息') for rule in self.rules
        }))
        self.state, self.last = 'resting', None
        self.break_deadline = at + self.break_duration_seconds
        self._flush(now, actions, penalties)

    def finish_break(self, at: float, now: datetime):
        if self.state != 'resting':
            raise ValueError("当前不在计划休息中")
        self._account(at)
        self.state, self.last = 'monitoring', None
        self.break_deadline = None
        self.next_break_work += self.break_every_seconds
        self._start_work_interval(at, now)
        self._flush(now)

    def extend(self, minutes: float, now: datetime):
        if not self.identifier or self.state not in ('monitoring', 'paused', 'resting'):
            raise ValueError("当前没有可延长的会话")
        minutes = minutes_valid(minutes)
        new_total = minutes_valid(self.total_minutes + minutes)
        self.total_minutes = new_total
        self.deadline += minutes * 60
        if self.state == 'paused' and self.paused_remaining is not None:
            self.paused_remaining += minutes * 60
        with self.store.db:
            self.store.db.execute(
                "UPDATE sessions SET minutes=?,extended_minutes=extended_minutes+?,updated=? "
                "WHERE id=?",
                (new_total, minutes, utc_text(now), self.identifier),
            )

    def source_gap(self, at: float, now: datetime, reason: str):
        """Break rule continuity while keeping unavailable time in coverage totals."""
        if self.state != 'monitoring':
            raise ValueError("当前没有运行中的会话")
        self._account(at)
        actions, penalties = self._process_rules(Observation(at, {
            rule.signal_name: SignalValue.exempt('摄像头重连：'+reason)
            for rule in self.rules
        }))
        self.last = None
        self._flush(now, actions, penalties)
        return actions

    def end(self, at: float, now: datetime, reason='ended', detail: str | None = None,
            resource_summary: dict | None = None):
        if not self.identifier:
            raise ValueError("没有会话")
        if self.state == 'monitoring':
            self.pause(at, now)
        elif self.state == 'resting':
            self._account(at)
            self._flush(now)
        identifier = self.identifier
        ended = utc_text(now)
        with self.store.db:
            resources = (json.dumps(resource_summary, ensure_ascii=False, sort_keys=True)
                         if resource_summary is not None else None)
            self.store.db.execute("UPDATE sessions SET state=?,ended=?,updated=?,error=?,"
                                  "resource_summary=? WHERE id=?",
                                  (reason, ended, ended,
                                   str(detail).strip()[:500] if detail else None,
                                   resources, identifier))
            self.store.db.execute(
                "UPDATE actions SET delivery_state='failed',delivery_attempted=?,delivery_error=? "
                "WHERE session_id=? AND delivery_state='pending'",
                (ended, "会话结束前未确认提醒呈现", identifier))
            self.store.db.execute("UPDATE tasks SET state=?,error=NULL "
                                  "WHERE id=(SELECT task_id FROM sessions WHERE id=?)",
                                  (reason, identifier))
        self.identifier, self.state = None, 'idle'
        self.paused_remaining = None
        self.focus = None
        return self.store.summary(identifier)

    def due(self, now: datetime):
        """Return one eligible task; caller starts only after source preflight."""
        for task in self.store.tasks():
            if task['state'] != 'pending' or task['starts'] > utc_text(now):
                continue
            late = (now - datetime.fromisoformat(task['starts'])).total_seconds()
            state = 'conflict' if self.identifier else 'missed' if late > 60 else None
            if state:
                with self.store.db:
                    detail = ('已有会话正在运行' if state == 'conflict' else
                              '任务到点后超过60秒，未自动打开摄像头')
                    self.store.db.execute(
                        "UPDATE tasks SET state=?,error=? WHERE id=? AND state='pending'",
                        (state, detail, task['id']))
            else:
                return task
        return None
