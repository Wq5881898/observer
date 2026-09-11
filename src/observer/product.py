"""Small local product layer. No vision dependencies and no research-data writes."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from observer.observations import Observation, SignalState, SignalValue
from observer.rules import ActionKind, default_rule_configs
from observer.scoring import RuleEngine

HARD_SIGNALS = ("away", "phone_held")
SCHEMA_VERSION = 4
REQUIRED_SCHEMA = {
    'tasks': {'id', 'title', 'starts', 'minutes', 'state', 'error'},
    'sessions': {'id', 'title', 'task_id', 'started', 'updated', 'ended', 'state',
                 'minutes', 'identity', 'policy', 'error'},
    'totals': {'session_id', 'signal', 'eligible', 'valid', 'violation'},
    'actions': {'id', 'session_id', 'event_id', 'rule_id', 'kind', 'offset', 'points', 'reason'},
    'evidence': {'id', 'session_id', 'path', 'sha256', 'error'},
    'settings': {'key', 'value'},
    'schema_migrations': {'version', 'applied', 'description'},
}
NAMES = {"away": "离席", "phone_held": "可见手机", "attention_away": "离开屏幕（软提示）",
         "posture_deviation": "姿态（软提示）", "ergonomic_risk": "健康提示"}


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
        last = self.db.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
        if last != SCHEMA_VERSION:
            raise ValueError("数据库迁移记录与结构版本不一致")
        if not self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name='one_active_session'"
        ).fetchone():
            raise ValueError("数据库缺失单会话约束")

    def _create_schema(self):
        applied = utc_text(utc_now())
        try:
            self.db.executescript("""
                BEGIN IMMEDIATE;
                CREATE TABLE tasks(id TEXT PRIMARY KEY, title TEXT NOT NULL,
                    starts TEXT NOT NULL, minutes REAL NOT NULL, state TEXT NOT NULL DEFAULT 'pending',
                    error TEXT);
                CREATE TABLE sessions(id TEXT PRIMARY KEY, title TEXT NOT NULL,
                    task_id TEXT UNIQUE REFERENCES tasks(id), started TEXT NOT NULL,
                    updated TEXT NOT NULL, ended TEXT, state TEXT NOT NULL,
                    minutes REAL NOT NULL, identity TEXT NOT NULL, policy TEXT NOT NULL,
                    error TEXT);
                CREATE UNIQUE INDEX one_active_session ON sessions((1))
                    WHERE state IN ('monitoring','paused');
                CREATE TABLE totals(session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
                    signal TEXT, eligible REAL NOT NULL, valid REAL NOT NULL,
                    violation REAL NOT NULL, PRIMARY KEY(session_id,signal));
                CREATE TABLE actions(id TEXT PRIMARY KEY,
                    session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
                    event_id TEXT NOT NULL, rule_id TEXT NOT NULL, kind TEXT NOT NULL,
                    offset REAL NOT NULL, points REAL NOT NULL, reason TEXT);
                CREATE TABLE evidence(id TEXT PRIMARY KEY,
                    session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
                    path TEXT NOT NULL, sha256 TEXT, error TEXT);
                CREATE TABLE settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY,
                    applied TEXT NOT NULL, description TEXT NOT NULL);
                INSERT INTO settings VALUES('images','false'),('retention_days','30');
            """)
            self.db.executemany(
                "INSERT INTO schema_migrations VALUES(?,?,?)",
                ((1, applied, 'initial product schema'),
                 (2, applied, 'versioned migration ledger'),
                 (3, applied, 'persist task outcome details'),
                 (4, applied, 'persist session interruption details')),
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

    def add_task(self, title: str, starts: datetime, minutes: float) -> str:
        if not title.strip():
            raise ValueError("任务名不能为空")
        identifier = str(uuid4())
        with self.db:
            self.db.execute("INSERT INTO tasks(id,title,starts,minutes) VALUES(?,?,?,?)",
                            (identifier, title.strip(), utc_text(starts), minutes_valid(minutes)))
        return identifier

    def tasks(self):
        return [dict(row) for row in self.db.execute("SELECT * FROM tasks ORDER BY starts")]

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
                "SELECT 1 FROM sessions WHERE state IN ('monitoring','paused')"
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

    def history(self):
        return [dict(row) for row in self.db.execute(
            "SELECT id,title,started,ended,state,minutes,identity FROM sessions ORDER BY started DESC LIMIT 100")]

    def summary(self, identifier: str):
        row = self.db.execute("SELECT * FROM sessions WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise ValueError("未找到会话")
        totals = {r['signal']: dict(r) for r in self.db.execute(
            "SELECT * FROM totals WHERE session_id=?", (identifier,))}
        coverage = min((totals.get(s, {}).get('valid', 0) /
                        max(totals.get(s, {}).get('eligible', 0), 1e-9) for s in HARD_SIGNALS))
        actions = [dict(r) for r in self.db.execute(
            "SELECT * FROM actions WHERE session_id=? ORDER BY offset,rowid", (identifier,))]
        points = sum(a['points'] for a in actions)
        return {**dict(row), "coverage": coverage, "score": max(0, 100-points) if coverage >= .8 else None,
                "deducted": points, "totals": totals, "actions": actions,
                "evidence": [dict(r) for r in self.db.execute(
                    "SELECT * FROM evidence WHERE session_id=?", (identifier,))]}

    def interrupt_abandoned(self):
        """Only call after acquiring the runtime lock. Do not bridge process clocks."""
        with self.db:
            self.db.execute("UPDATE tasks SET state='interrupted',error=? WHERE id IN "
                            "(SELECT task_id FROM sessions WHERE state IN ('monitoring','paused'))",
                            ("程序上次退出时会话未正常结束",))
            self.db.execute("UPDATE sessions SET state='interrupted',ended=updated,error=? "
                            "WHERE state IN ('monitoring','paused')",
                            ("程序上次退出时会话未正常结束",))

    def delete_session(self, identifier: str):
        # File removal precedes row removal: failures leave a retryable index.
        with self.db:
            row = self.db.execute("SELECT state FROM sessions WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise ValueError("未找到会话")
            if row[0] in ('monitoring', 'paused'):
                raise ValueError("请先结束会话")
            for item in self.db.execute("SELECT path FROM evidence WHERE session_id=?", (identifier,)):
                if item[0]:
                    path = self.safe_image_path(item[0])
                    self.safe_image_path(item[0]+'.tmp').unlink(missing_ok=True)
                    path.unlink(missing_ok=True)
            self.db.execute("DELETE FROM sessions WHERE id=?", (identifier,))

    def safe_image_path(self, relative: str) -> Path:
        base = self.root / "evidence"
        if base.resolve() != base:
            raise ValueError("证据目录不能通过链接指向其他位置")
        target = (self.root / relative).resolve()
        if not target.is_relative_to(base) or target == base:
            raise ValueError("证据路径越界")
        return target

    def purge(self, now: datetime):
        cutoff = utc_text(now - timedelta(days=self.setting('retention_days')))
        ids = [r[0] for r in self.db.execute(
            "SELECT id FROM sessions WHERE ended < ? AND state NOT IN ('monitoring','paused')", (cutoff,))]
        for identifier in ids:
            self.delete_session(identifier)
        return len(ids)

    def capture(self, session_id: str, event_id: str, frame):
        if not self.setting('images'):
            return
        if self.db.execute("SELECT 1 FROM evidence WHERE id=?", (event_id,)).fetchone():
            return
        relative, digest, error = '', None, None
        try:
            if frame is None:
                raise ValueError("事件发生时没有新鲜图像")
            target = self.safe_image_path(f"evidence/{event_id}.jpg")
            target.parent.mkdir(exist_ok=True)
            import cv2
            try:
                ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            except cv2.error as exc:
                raise ValueError('证据图编码失败') from exc
            if not ok:
                raise ValueError("证据图编码失败")
            raw = encoded.tobytes()
            if len(raw)+sum(p.stat().st_size for p in target.parent.glob('*.jpg')) > 500 * 1024**2:
                raise ValueError("证据图配额已满")
            relative = target.relative_to(self.root).as_posix()
            # Persist intent first: a crash leaves a known, deletable path.
            with self.db:
                self.db.execute("INSERT INTO evidence VALUES(?,?,?,?,?)",
                                (event_id, session_id, relative, None, 'pending'))
            temporary = self.safe_image_path(relative+'.tmp')
            temporary.write_bytes(raw)
            temporary.replace(target)
            digest = hashlib.sha256(raw).hexdigest()
        except (OSError, ValueError) as exc:
            error = str(exc)
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO evidence VALUES(?,?,?,?,?)",
                            (event_id, session_id, relative, digest, error))


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
        self.rules = default_rule_configs()
        self.engine = RuleEngine(self.rules)
        self.last = None
        self.totals = {}

    def start(self, title: str, minutes: float, at: float, now: datetime,
              identity: dict, task_id: str | None = None):
        if self.identifier:
            raise ValueError("已有会话，请先结束")
        if not title.strip() or not math.isfinite(at) or at < 0:
            raise ValueError("任务名或时间无效")
        minutes_valid(minutes)
        identifier = str(uuid4())
        with self.store.db:
            if task_id and not self.store.db.execute(
                "UPDATE tasks SET state='running',error=NULL WHERE id=? AND state='pending'",
                (task_id,)).rowcount:
                raise ValueError("任务已处理")
            self.store.db.execute(
                "INSERT INTO sessions(id,title,task_id,started,updated,ended,state,minutes,identity,policy,error) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (identifier, title.strip(), task_id, utc_text(now), utc_text(now), None,
                 'monitoring', minutes, json.dumps(identity, ensure_ascii=False),
                 json.dumps([asdict(r) for r in self.rules]), None))
        self.identifier, self.state = identifier, 'monitoring'
        self.started = self.last_at = self.flushed_at = at
        self.deadline = at + minutes * 60
        self.last = None
        self.engine = RuleEngine(self.rules)
        self.totals = {r.signal_name: [0., 0., 0.] for r in self.rules}
        return identifier

    def _account(self, at: float, current: Observation | None = None):
        delta = at - self.last_at
        if delta < 0 or not math.isfinite(delta):
            raise ValueError("时间不能倒退或无效")
        if self.state == 'monitoring':
            for name, values in self.totals.items():
                previous = self.last.signal(name).state if self.last else SignalState.UNKNOWN
                following = current.signal(name).state if current else SignalState.UNKNOWN
                values[0] += delta
                # Never assume an unobserved gap was compliant or a violation.
                if delta <= 2 and previous == following and previous in (
                        SignalState.COMPLIANT, SignalState.VIOLATION):
                    values[1] += delta
                    values[2] += delta * (previous == SignalState.VIOLATION)
        self.last_at = at

    def _flush(self, now: datetime, actions=(), penalties=()):
        if not self.identifier:
            return
        points = {(p.event_id, p.at_monotonic): p.points for p in penalties}
        try:
            with self.store.db:
                self.store.db.execute("UPDATE sessions SET updated=?,state=? WHERE id=?",
                                      (utc_text(now), self.state, self.identifier))
                for name, values in self.totals.items():
                    self.store.db.execute("INSERT OR REPLACE INTO totals VALUES(?,?,?,?,?)",
                                          (self.identifier, name, *values))
                for action in actions:
                    key = f'{action.event_id}:{action.kind.value}:{action.at_monotonic:.9f}'
                    self.store.db.execute("INSERT OR IGNORE INTO actions VALUES(?,?,?,?,?,?,?,?)",
                        (key, self.identifier, action.event_id, action.rule_id, action.kind.value,
                         action.at_monotonic-self.started,
                         points.get((action.event_id, action.at_monotonic), 0.)
                         if action.kind is ActionKind.PENALTY else 0., action.reason))
        except sqlite3.Error:
            self.state = 'faulted'
            raise
        self.flushed_at = self.last_at

    def process(self, observation: Observation, now: datetime):
        if not self.identifier or self.state != 'monitoring':
            return ()
        at = observation.monotonic_seconds
        if self.last is not None and at == self.last.monotonic_seconds:
            return ()
        self._account(at, observation)
        # Absence is the primary fact; do not punish a visible unattended phone too.
        signals = dict(observation.signals)
        if observation.signal('away').state is SignalState.VIOLATION:
            signals['phone_held'] = SignalValue.exempt('离席期间避免重复处罚')
        result = self.engine.process(Observation(at, signals))
        self.last = observation
        if result.actions or at-self.flushed_at >= 1:
            self._flush(now, result.actions, result.penalties)
        return result.actions

    def pause(self, at: float, now: datetime):
        if self.state != 'monitoring':
            raise ValueError("当前没有运行中的会话")
        self._account(at)
        result = self.engine.process(Observation(at, {
            r.signal_name: SignalValue.exempt('会话暂停') for r in self.rules}))
        self.state, self.last = 'paused', None
        self._flush(now, result.actions, result.penalties)

    def resume(self, at: float, now: datetime):
        if self.state != 'paused':
            raise ValueError("会话没有暂停")
        self._account(at)
        self.state, self.last = 'monitoring', None
        self._flush(now)

    def source_gap(self, at: float, now: datetime, reason: str):
        """Break rule continuity while keeping unavailable time in coverage totals."""
        if self.state != 'monitoring':
            raise ValueError("当前没有运行中的会话")
        self._account(at)
        result = self.engine.process(Observation(at, {
            rule.signal_name: SignalValue.exempt('摄像头重连：'+reason)
            for rule in self.rules
        }))
        self.last = None
        self._flush(now, result.actions, result.penalties)
        return result.actions

    def end(self, at: float, now: datetime, reason='ended', detail: str | None = None):
        if not self.identifier:
            raise ValueError("没有会话")
        if self.state == 'monitoring':
            self.pause(at, now)
        identifier = self.identifier
        with self.store.db:
            self.store.db.execute("UPDATE sessions SET state=?,ended=?,updated=?,error=? WHERE id=?",
                                  (reason, utc_text(now), utc_text(now),
                                   str(detail).strip()[:500] if detail else None, identifier))
            self.store.db.execute("UPDATE tasks SET state=?,error=NULL "
                                  "WHERE id=(SELECT task_id FROM sessions WHERE id=?)",
                                  (reason, identifier))
        self.identifier, self.state = None, 'idle'
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
