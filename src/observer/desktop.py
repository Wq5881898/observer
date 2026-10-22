"""One-window Qt frontend. SQLite and inference belong to the worker thread."""
from __future__ import annotations

import html
import io
import json
import math
import os
import queue
import sqlite3
import struct
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QDateTime, QSettings, QThread, QTime, QTimer, Signal, Slot, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDateTimeEdit, QFormLayout,
    QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QPushButton, QScrollArea, QSpinBox, QTabWidget, QTextBrowser, QTextEdit, QTimeEdit,
    QVBoxLayout, QWidget,
)
from observer.local_timezone import get_localzone_name

from observer.product import NAMES, ProductStore, RuntimeLock, SessionService, minutes_valid, utc_now
from observer.product_cli import (
    BackgroundCameraPreloader, CameraSource, DemoSource, FocusEvidenceRecorder, SourceStalled,
    capture_event_evidence, prepare_camera_resources,
)
from observer.config import load_config
from observer.resources import SessionResourceTracker
from observer.rules import ActionKind
from observer.desktop_tray import DesktopTray
from observer.desktop_ipc import SingleInstanceWake
from observer.runtime_watch import RuntimeWatch
from observer.power_events import WindowsPowerEventFilter
from observer.deployment import source_identity

STATES = {'idle': '等待开始', 'monitoring': '观察中', 'paused': '已暂停',
          'resting': '计划休息中',
          'ended': '已结束', 'interrupted': '已中断', 'pending': '待开始',
          'running': '进行中', 'conflict': '任务冲突', 'missed': '已错过',
          'cancelled': '已取消', 'failed': '启动失败', 'faulted': '运行异常'}
HEALTH_REMINDER_RULES = frozenset({
    'posture_deviation',
})
HARD_REMINDER_STYLE = (
    'background:#fee2e2; color:#991b1b; border:1px solid #fca5a5; '
    'border-radius:5px; padding:10px; font-size:16px; font-weight:600;'
)
HEALTH_REMINDER_STYLE = (
    'background:#f0fdfa; color:#115e59; border:1px solid #99f6e4; '
    'border-radius:5px; padding:8px; font-size:14px; font-weight:500;'
)
HEALTH_REMINDER_BANNER_MS = 4_000
STARTUP_LOG_MAX_BYTES = 512 * 1024
ACTION_BUTTON_ENABLED_STYLE = (
    'QPushButton {background:#2563eb; color:white; border:1px solid #2563eb; '
    'border-radius:5px;} QPushButton:hover {background:#1d4ed8;} '
    'QPushButton:pressed {background:#1e40af;}'
)
ACTION_BUTTON_DISABLED_STYLE = (
    'QPushButton {background:#e2e8f0; color:#94a3b8; '
    'border:1px solid #cbd5e1; border-radius:5px;}'
)


def append_startup_log(data_dir: Path, record: dict) -> Path:
    """Append one privacy-safe startup result to a bounded local JSONL log."""
    log_dir = Path(data_dir).resolve() / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / 'startup.jsonl'
    backup = log_dir / 'startup.previous.jsonl'
    encoded = (json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n').encode('utf-8')
    if path.is_file() and path.stat().st_size + len(encoded) > STARTUP_LOG_MAX_BYTES:
        backup.unlink(missing_ok=True)
        path.replace(backup)
    with path.open('ab') as stream:
        stream.write(encoded)
    return path


def summary_text(result):
    score = (f"{result['score']:.1f} 分" if result['score'] is not None
             and result.get('score_version') == 'time_ratio_v2'
             else f"{result['score']:g} 分" if result['score'] is not None
             else '证据不足')
    mode = {'demo': '模拟', 'real': '真实', 'unknown': '身份异常'}[result['mode']]
    lines = [f"{result['title']} · {mode}",
             f"{STATES.get(result['state'],result['state'])}  |  {score}  |  有效覆盖 {result['coverage']:.0%}", '']
    if result.get('data_warnings'):
        lines.extend(['数据警告：' + '；'.join(result['data_warnings']), ''])
    options = []
    if result.get('extended_minutes'):
        options.append(f"本次延长 {result['extended_minutes']:g} 分钟")
    if result.get('break_seconds'):
        options.append(f"计划休息 {result['break_seconds']/60:.1f} 分钟")
    if result.get('allow_offscreen'):
        options.append('允许离屏资料')
    if options:
        lines.extend([' · '.join(options), ''])
    if result.get('score_version') == 'time_ratio_v2':
        lines.extend([
            f"计分 {result.get('scoring_seconds', 0):.1f} 秒 · "
            f"违规 {result.get('violation_seconds', 0):.1f} 秒 · "
            f"合规 {result.get('compliant_seconds', 0):.1f} 秒",
            '',
        ])
    incidents = {item['id']: item for item in result.get('focus_incidents', [])}
    for action in result['actions']:
        if action['kind'] in ('alert', 'penalty'):
            incident = incidents.get(action.get('event_id'))
            name = (' + '.join(NAMES.get(signal, signal)
                               for signal in incident['signals'])
                    if incident else NAMES.get(action['rule_id'], action['rule_id']))
            detail = f"扣 {action['points']:g} 分" if action['kind']=='penalty' else '提醒'
            if incident:
                detail += f" · 违规 {incident['violation_seconds']:.1f} 秒"
            lines.append(f"{action['offset']:.1f} 秒  {name} · {detail}")
    if not result['actions']:
        lines.append('无提醒或计分事件')
    attention_seconds = result['totals'].get('attention_away', {}).get('violation', 0)
    if attention_seconds >= .1:
        lines.extend(['', f"离开屏幕累计异常：{attention_seconds:.1f}秒",
                      '累计时间不等于连续触发时间'])
    posture_seconds = result['totals'].get(
        'posture_deviation', {}).get('violation', 0)
    legacy_ergonomic_seconds = result['totals'].get(
        'ergonomic_risk', {}).get('violation', 0)
    if posture_seconds >= .1:
        lines.extend(['', f'姿态累计偏离：{posture_seconds:.1f}秒',
                      '累计时间不等于连续触发时间'])
    if legacy_ergonomic_seconds >= .1:
        lines.extend(['', f'旧版健康提示累计异常：{legacy_ergonomic_seconds:.1f}秒',
                      '仅用于查看合并前的历史会话'])
    alerts = [action for action in result['actions'] if action['kind'] == 'alert']
    if alerts:
        delivered = sum(action.get('delivery_state') == 'delivered' for action in alerts)
        unresolved = sum(action.get('delivery_state') in ('pending', 'failed') for action in alerts)
        unknown = sum(action.get('delivery_state') == 'unknown' for action in alerts)
        parts = [f'{delivered}次已确认']
        if unresolved:
            parts.append(f'{unresolved}次未确认')
        if unknown:
            parts.append(f'{unknown}次历史状态未知')
        lines.extend(['', '提醒交付：' + ' · '.join(parts)])
    if result.get('error'):
        lines.extend(['', '中断原因：' + result['error']])
    resources = result.get('resource_summary')
    if resources:
        if resources['sample_count']:
            lines.extend(['',
                          f"资源：CPU平均 {resources['process_cpu_average_percent']:.1f}% · "
                          f"P95 {resources['process_cpu_p95_percent']:.1f}% · "
                          f"内存峰值 {resources['process_memory_peak_mb']:.0f} MB"])
        elif resources.get('sampling_error'):
            lines.extend(['', '资源采样不可用：' + resources['sampling_error']])
    return '\n'.join(lines)


def _reminder_wave(health: bool) -> bytes:
    """Build one gentle health tone or two distinct violation tones."""
    sample_rate = 16_000
    frequency = 660.0 if health else 880.0
    duration = .12
    amplitude = .10 if health else .28
    sample_count = round(sample_rate * duration)
    fade_samples = max(1, round(sample_rate * .012))
    frames = bytearray()
    for index in range(sample_count):
        envelope = min(1.0, index / fade_samples,
                       (sample_count-index-1) / fade_samples)
        value = round(32767 * amplitude * envelope *
                      math.sin(2 * math.pi * frequency * index / sample_rate))
        frames.extend(struct.pack('<h', value))
    if not health:
        # Keep both beeps in one wave so playback cannot overlap itself.
        silence = bytes(round(sample_rate * .08) * 2)
        frames = frames + silence + frames
    output = io.BytesIO()
    with wave.open(output, 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(frames)
    return output.getvalue()


def play_reminder_sound(rule_id: str):
    """Play a short hard or gentle reminder; keep failures visible to the caller."""
    health = rule_id in HEALTH_REMINDER_RULES
    if os.name == 'nt':
        try:
            import winsound
            sound = _reminder_wave(health)
            # SND_MEMORY playback is synchronous. The tones are capped at 320 ms,
            # so play here instead of acknowledging delivery before a background
            # thread has actually attempted the Windows audio call.
            winsound.PlaySound(sound, winsound.SND_MEMORY | winsound.SND_NODEFAULT)
            return
        except (ImportError, RuntimeError):
            pass
    QApplication.beep()
    if not health:
        QTimer.singleShot(200, QApplication.beep)


def data_status_text(status):
    size = status['image_bytes']
    if size < 1024:
        storage = f'{size} B'
    elif size < 1024**2:
        storage = f'{size/1024:.1f} KB'
    else:
        storage = f'{size/1024**2:.1f} MB'
    mode = '开启' if status['images_enabled'] else '关闭'
    test_mode = ' · 测试取证开启' if status.get('test_evidence_enabled') else ''
    problem = f" · 证据异常 {status['evidence_errors']}" if status['evidence_errors'] else ''
    return (f"本地：{status['sessions']} 条记录 · {status['tasks']} 个任务 · "
            f"{status.get('plans', 0)} 个重复计划 · "
            f"{status['images']} 张图片/{storage} · "
            f"截图{mode}{test_mode} · 保留{status['retention_days']}天{problem}")


def user_error_message(error) -> str:
    """Keep technical diagnostics in storage while presenting actionable UI text."""
    text = str(error or '').strip()
    if text.startswith('session attention setup timed out'):
        return ('未能完成头眼基线。请正对摄像头，确保完整脸部和双眼可见，'
                '并看屏幕中央后重试。')
    return text


def live_other_event_text(signals: dict, state: str) -> str:
    """Describe current focus disturbances without implying event activation."""
    if state in ('paused', 'resting'):
        return '已暂停'
    phone = signals.get('phone_held', 'unknown')
    away = signals.get('away', 'unknown')
    active = []
    if phone == 'violation':
        active.append('发现手机')
    if away == 'violation':
        active.append('离席候选')
    if active:
        return ' · '.join(active)
    unknown = []
    if phone == 'unknown':
        unknown.append('手机不可判断')
    if away == 'unknown':
        unknown.append('离席不可判断')
    if len(unknown) == 2:
        return '不可判断'
    if unknown:
        return unknown[0]
    return '未发现'


def session_event_rows(store: ProductStore, result: dict | None) -> list[dict]:
    """Return activated events and test candidates with validated image sequences."""
    if not result:
        return []
    grouped = {}
    for action in result.get('actions', []):
        grouped.setdefault(action['event_id'], []).append(action)
    evidence = {}
    for source in result.get('evidence', []):
        item = dict(source)
        raw_signals = item.get('signals', [])
        try:
            item['signals'] = (json.loads(raw_signals)
                               if isinstance(raw_signals, str) else list(raw_signals))
        except (json.JSONDecodeError, TypeError):
            item['signals'] = []
        item['image_path'] = None
        item['image_status'] = '未保存图片'
        if item.get('error'):
            item['image_status'] = '图片保存失败：' + item['error']
        elif item.get('path'):
            try:
                target = store.safe_image_path(item['path'])
            except ValueError as exc:
                item['image_status'] = '图片路径异常：' + str(exc)
            else:
                if target.is_file():
                    item['image_path'] = str(target)
                    item['image_status'] = '查看'
                else:
                    item['image_status'] = '图片文件缺失'
        evidence.setdefault(item.get('event_id') or item['id'], []).append(item)
    for images in evidence.values():
        images.sort(key=lambda item: (
            float(item['offset']) if item.get('offset') is not None else float('inf'),
            item['id']))
    incidents = {item['id']: item for item in result.get('focus_incidents', [])}
    rows = []
    for action in result.get('actions', []):
        if action['kind'] != 'activated':
            continue
        related = grouped[action['event_id']]
        incident = incidents.get(action['event_id'])
        alert = next((item for item in related if item['kind'] == 'alert'), None)
        resolved = next((item for item in related if item['kind'] == 'resolved'), None)
        points = sum(float(item['points']) for item in related
                     if item['kind'] == 'penalty')
        records = evidence.get(action['event_id'], [])
        valid_images = [item for item in records if item.get('image_path')]
        image_path = valid_images[0]['image_path'] if valid_images else None
        image_status = '未保存图片'
        image_sha256 = valid_images[0].get('sha256') if valid_images else None
        if valid_images:
            image_status = ('查看' if len(valid_images) == 1
                            else f'查看 {len(valid_images)}张')
        elif records:
            image_status = records[0]['image_status']
        rows.append({
            'event_id': action['event_id'],
            'rule_id': action['rule_id'],
            'offset': float(incident['started']) if incident else float(action['offset']),
            'resolved_offset': (float(incident['ended']) if incident and
                                incident.get('ended') is not None else
                                float(resolved['offset']) if resolved else None),
            'resolved_reason': resolved.get('reason') if resolved else None,
            'points': points,
            'delivery_state': alert.get('delivery_state') if alert else None,
            'image_path': image_path,
            'image_status': image_status,
            'image_sha256': image_sha256,
            'images': valid_images,
            'signals': incident.get('signals', []) if incident else [],
            'violation_seconds': (float(incident['violation_seconds'])
                                  if incident else None),
        })
    known = set(grouped)
    for event_id, records in evidence.items():
        if event_id in known:
            continue
        valid_images = [item for item in records if item.get('image_path')]
        if not records:
            continue
        offsets = [float(item['offset']) for item in records if item.get('offset') is not None]
        signals = tuple(dict.fromkeys(
            signal for item in records for signal in item.get('signals', [])
        ))
        rows.append({
            'event_id': event_id,
            'rule_id': 'focus_candidate',
            'offset': min(offsets) if offsets else 0.,
            'resolved_offset': max(offsets) if len(offsets) > 1 else None,
            'resolved_reason': 'candidate',
            'points': 0.,
            'delivery_state': None,
            'image_path': valid_images[0]['image_path'] if valid_images else None,
            'image_status': (('查看' if len(valid_images) == 1
                              else f'查看 {len(valid_images)}张') if valid_images
                             else records[0]['image_status']),
            'image_sha256': valid_images[0].get('sha256') if valid_images else None,
            'images': valid_images,
            'signals': list(signals),
            'violation_seconds': None,
            'test_candidate': True,
        })
    rows.sort(key=lambda item: (item['offset'], item['event_id']))
    return rows


class DesktopWorker(QThread):
    snapshot = Signal(dict)
    notice = Signal(str)
    detail = Signal(str)
    busy = Signal(bool)
    reminder = Signal(str, str, str)
    task_notification = Signal(str, str)
    BUSY_COMMANDS = frozenset({
        'start', 'pause', 'resume', 'end', 'task', 'delete-session', 'purge',
        'plan-save', 'plan-toggle', 'plan-delete',
        'extend',
    })

    def __init__(self, args, source_factory=None):
        super().__init__()
        self.args = args
        self.preloader = None
        if source_factory is not None:
            self.factory = source_factory
        elif args.demo:
            self.factory = DemoSource
        else:
            self.preloader = BackgroundCameraPreloader(
                lambda: prepare_camera_resources(
                    args.models_dir, args.profile, background=True))
            self.factory = self.create_camera_source
        self.commands = queue.Queue(maxsize=32)
        self.runtime_watch = RuntimeWatch()
        self.power_event_lock = threading.Lock()
        self.power_event_reason = None
        self.ui_visible = True
        self.source = None
        self.store = None
        self.service = None
        self.storage_snapshot = None
        self.today_snapshot = None
        self.next_storage_refresh = 0.
        self.resource_tracker = None
        self.last_session_id = None
        self.startup_started = None
        self.startup_progress = []
        self.evidence_recorder = FocusEvidenceRecorder()

    def create_camera_source(self):
        started = time.monotonic()
        if not self.preloader.ready:
            self.source_progress('启动中：正在等待后台视觉模型准备…')
        prepared = self.preloader.acquire(self.isInterruptionRequested)
        return CameraSource(
            self.args.models_dir,
            self.args.profile,
            self.source_progress,
            setup_preview=False,
            cancelled=self.isInterruptionRequested,
            prepared=prepared,
            startup_started=started,
        )

    def start_model_prewarm(self):
        if (self.preloader is None or self.isInterruptionRequested()
                or (self.service is not None and self.service.state == 'monitoring')):
            return

        def completed(error):
            if self.isInterruptionRequested():
                return
            if error is not None:
                self.notice.emit(f'后台视觉模型准备失败：{error}')
            elif self.preloader.ready and self.service and not self.service.identifier:
                self.notice.emit('视觉模型已在后台准备好；开始时不会再等待模型加载。')

        self.preloader.start(completed)

    def source_progress(self, message):
        if self.startup_started is not None:
            self.startup_progress.append({
                'elapsed_seconds': round(time.monotonic()-self.startup_started, 3),
                'message': message,
            })
        self.notice.emit(message)

    def record_startup(self, status, elapsed, *, identity=None,
                       session_id=None, error=None):
        if self.args.demo:
            return
        record = {
            'timestamp': utc_now().isoformat(),
            'status': status,
            'elapsed_seconds': round(elapsed, 3),
            'session_id': session_id,
            'phases': (identity or {}).get('startup_seconds'),
            'progress': list(self.startup_progress),
        }
        if error is not None:
            record['error_type'] = type(error).__name__
            record['error'] = str(error)
        try:
            append_startup_log(self.args.data_dir, record)
        except OSError:
            pass

    def invalidate_storage_status(self):
        self.storage_snapshot = None
        self.today_snapshot = None
        self.next_storage_refresh = 0.

    def submit(self, command, *values):
        if not self.isRunning():
            return False
        try:
            self.commands.put_nowait((command, values))
        except queue.Full:
            self.notice.emit('正在处理，请稍候')
            return False
        return True

    def stop_source(self):
        source, self.source = self.source, None
        if source:
            source.close()

    def begin(self, title, minutes, task_id=None, break_every_minutes=None,
              break_minutes=None, allow_offscreen=False):
        if self.service.identifier:
            raise ValueError('请先结束当前会话')
        minutes_valid(minutes)
        if not title.strip():
            raise ValueError('请输入任务名称')
        self.busy.emit(True)
        self.evidence_recorder.reset()
        self.notice.emit('正在准备观察…')
        preparation_started = time.monotonic()
        self.startup_started = preparation_started
        self.startup_progress = []
        try:
            self.source = self.factory()
            if self.isInterruptionRequested():
                self.stop_source()
                self.record_startup(
                    'cancelled', time.monotonic()-preparation_started)
                return
            identity = dict(self.source.identity)
            identity.update(source_identity())
            preparation_seconds = time.monotonic()-preparation_started
            identity['product_startup_seconds'] = round(preparation_seconds, 3)
            self.last_session_id = self.service.start(
                title, minutes, time.monotonic(), utc_now(), identity, task_id,
                break_every_minutes, break_minutes, allow_offscreen)
            self.resource_tracker.begin(time.monotonic())
            self.record_startup(
                'success', time.monotonic()-preparation_started,
                identity=identity, session_id=self.last_session_id)
            self.invalidate_storage_status()
            self.notice.emit(f'已开始 · 准备耗时 {preparation_seconds:.1f} 秒。可以自然使用电脑。')
        except Exception as exc:
            self.record_startup(
                'failed', time.monotonic()-preparation_started, error=exc)
            self.stop_source()
            raise
        finally:
            self.startup_started = None
            try:
                self.publish()
            finally:
                self.busy.emit(False)

    def finish_session(self, reason='ended', detail=None):
        try:
            at = time.monotonic()
            if (self.service.identifier and self.source and
                    self.evidence_recorder.finish(
                        self.store, self.service, self.source.frame, at)):
                self.invalidate_storage_status()
            self.last_session_id = self.service.identifier
            resources = self.resource_tracker.finish(at) if self.resource_tracker else None
            result = self.service.end(at, utc_now(), reason, detail, resources)
            self.invalidate_storage_status()
            self.detail.emit(summary_text(result))
        finally:
            self.stop_source()
            self.start_model_prewarm()

    def check_runtime_continuity(self):
        clock_reason = self.runtime_watch.check(time.monotonic(), time.time())
        with self.power_event_lock:
            power_reason = self.power_event_reason
            self.power_event_reason = None
        reason = power_reason or clock_reason
        if not reason or not self.service.identifier:
            return False
        if self.service.state == 'paused':
            return False
        if self.service.state == 'monitoring':
            # Break at the last accounted instant, never extend a pre-gap signal.
            self.service.source_gap(self.service.last_at, utc_now(), reason)
        self.evidence_recorder.reset()
        self.finish_session('interrupted', reason)
        self.notice.emit(f'会话已中断：{reason}。已保存记录，请重新开始。')
        return True

    def note_power_transition(self, reason):
        """May be called from Qt's UI thread; worker consumes it at a safe point."""
        with self.power_event_lock:
            self.power_event_reason = reason

    def publish(self):
        service = self.service
        at = time.monotonic()
        if self.storage_snapshot is None or at >= self.next_storage_refresh:
            self.storage_snapshot = self.store.data_status()
            self.today_snapshot = self.store.today_summary(utc_now(), get_localzone_name())
            self.next_storage_refresh = at + 5
        history = self.store.history()
        current = self.store.summary(service.identifier) if service.identifier else None
        focus_session = current
        if focus_session is None and self.last_session_id:
            try:
                focus_session = self.store.summary(self.last_session_id)
            except ValueError:
                self.last_session_id = None
        if focus_session is None and history:
            self.last_session_id = history[0]['id']
            focus_session = self.store.summary(self.last_session_id)
        signals = ({name: service.last.signal(name).state.value for name in NAMES}
                   if service.last is not None else {})
        self.snapshot.emit({'state': service.state, 'current': current,
                            'remaining': service.remaining_seconds(at) if current else 0,
                            'break_remaining': max(
                                0, service.break_deadline-time.monotonic())
                            if service.state == 'resting' else 0,
                            'tasks': self.store.tasks(), 'plans': self.store.plans(),
                            'history': history,
                            'focus_session': focus_session,
                            'focus_events': session_event_rows(self.store, focus_session),
                            'images': self.store.setting('images'),
                            'test_evidence': self.store.setting('test_evidence'),
                            'data_status': self.storage_snapshot,
                            'today': self.today_snapshot,
                            'signals': signals,
                            'demo_signal': getattr(self.source, 'signal', 'normal')})

    def execute(self, command, values):
        service, store = self.service, self.store
        if command == 'start':
            self.begin(*values)
        elif command == 'pause':
            if (self.source and self.evidence_recorder.finish(
                    store, service, self.source.frame, time.monotonic())):
                self.invalidate_storage_status()
            service.pause(time.monotonic(), utc_now())
            self.stop_source()
            self.start_model_prewarm()
            self.notice.emit('已暂停。倒计时已冻结。')
        elif command == 'resume':
            if service.state != 'paused':
                raise ValueError('会话没有暂停')
            self.busy.emit(True)
            try:
                self.source = self.factory()
                if self.isInterruptionRequested():
                    self.stop_source()
                    return
                service.resume(time.monotonic(), utc_now())
                self.notice.emit('已继续')
            except Exception:
                self.stop_source()
                self.start_model_prewarm()
                raise
            finally:
                try:
                    self.publish()
                finally:
                    self.busy.emit(False)
        elif command == 'end':
            self.finish_session()
            self.notice.emit('会话已保存')
        elif command == 'extend':
            service.extend(values[0], utc_now())
            extended = store.summary(service.identifier)['extended_minutes']
            self.notice.emit(
                f'当前会话已延长 {values[0]:g} 分钟（本次累计延长 {extended:g} 分钟）')
        elif command == 'task':
            title, when, minutes = values
            if when <= utc_now():
                raise ValueError('请选择未来的开始时间')
            store.add_task(title, when, minutes)
            self.notice.emit('已安排。保持程序运行（可隐藏到托盘），到点自动开始。')
        elif command == 'plan-save':
            identifier, title, local_time, timezone_name, recurrence, minutes, weekdays, every, rest, offscreen = values
            now = utc_now()
            if identifier:
                store.update_plan(identifier, title, local_time, timezone_name,
                                  recurrence, minutes, weekdays, every, rest,
                                  offscreen, now)
                message = '重复计划已更新'
            else:
                store.add_plan(title, local_time, timezone_name, recurrence,
                               minutes, weekdays, every, rest, offscreen, now)
                message = '重复计划已添加'
            store.materialize_plans(now)
            self.invalidate_storage_status()
            self.notice.emit(message)
        elif command == 'plan-toggle':
            identifier, enabled = values
            now = utc_now()
            store.set_plan_enabled(identifier, enabled, now)
            if enabled:
                store.materialize_plans(now)
            self.invalidate_storage_status()
            self.notice.emit('重复计划已启用' if enabled else '重复计划已停用')
        elif command == 'plan-delete':
            store.delete_plan(values[0], utc_now())
            self.invalidate_storage_status()
            self.notice.emit('重复计划已删除，历史执行记录已保留')
        elif command == 'cancel':
            store.cancel_task(values[0])
        elif command == 'retry':
            store.retry_task(values[0], utc_now())
            self.notice.emit('任务已重新进入待开始状态')
        elif command == 'show':
            self.detail.emit(summary_text(store.summary(values[0])))
        elif command == 'delete-session':
            store.delete_session(values[0])
            self.invalidate_storage_status()
            self.detail.emit('')
            self.notice.emit('记录及其事件图片已删除')
        elif command == 'purge':
            removed = store.purge(utc_now())
            self.invalidate_storage_status()
            self.notice.emit(f'已清理 {removed} 条过期记录')
        elif command == 'images':
            store.set_images(values[0])
            if not values[0]:
                self.evidence_recorder.reset()
            self.invalidate_storage_status()
        elif command == 'test-evidence':
            store.set_test_evidence(values[0])
            self.invalidate_storage_status()
        elif command == 'notification':
            store.mark_notification(*values)
        elif command == 'signal':
            if isinstance(self.source, DemoSource):
                self.source.signal = values[0]
        elif command == 'visibility':
            self.ui_visible = bool(values[0])

    def run(self):
        lock = None
        failure_detail = None
        try:
            lock = RuntimeLock(Path(self.args.data_dir).resolve())
            self.store = ProductStore(self.args.data_dir)
            self.service = SessionService(self.store)
            self.resource_tracker = SessionResourceTracker(load_config().resources)
            self.store.recover_evidence()
            self.store.interrupt_abandoned()
            startup_now = utc_now()
            self.store.materialize_plans(startup_now)
            self.store.expire_pending_tasks(startup_now)
            self.store.purge(startup_now)
            history = self.store.history()
            self.last_session_id = history[0]['id'] if history else None
            self.publish()
            self.busy.emit(False)
            self.start_model_prewarm()
            next_publish = time.monotonic() + .5
            next_purge = time.monotonic() + 3600
            next_plan_refresh = time.monotonic() + 60
            while not self.isInterruptionRequested():
                at, now = time.monotonic(), utc_now()
                if self.check_runtime_continuity():
                    next_publish = 0
                if at >= next_plan_refresh:
                    result = self.store.materialize_plans(now)
                    if result['created']:
                        next_publish = 0
                    next_plan_refresh = at + 60
                if (self.service.identifier and self.service.state != 'paused'
                        and at >= self.service.deadline):
                    self.finish_session()
                    self.notice.emit('计划时间已到，会话已保存')
                    next_publish = 0
                elif self.service.break_due():
                    if (self.source and self.evidence_recorder.finish(
                            self.store, self.service, self.source.frame, at)):
                        self.invalidate_storage_status()
                    self.service.start_break(at, now)
                    self.stop_source()
                    self.start_model_prewarm()
                    self.notice.emit(
                        f'计划休息 {self.service.break_duration_seconds/60:g} 分钟，摄像头已关闭')
                    next_publish = 0
                elif (self.service.state == 'resting'
                      and at >= self.service.break_deadline):
                    self.notice.emit('计划休息结束，正在重新打开摄像头…')
                    self.busy.emit(True)
                    try:
                        self.source = self.factory()
                        self.service.finish_break(time.monotonic(), utc_now())
                    except Exception as exc:
                        self.finish_session('interrupted', f'计划休息后恢复失败：{exc}')
                        self.notice.emit(f'会话已中断：计划休息后恢复失败：{exc}')
                    else:
                        self.notice.emit('计划休息结束，观察已继续')
                    finally:
                        self.busy.emit(False)
                    next_publish = 0
                task = self.service.due(now)
                if task:
                    attempt_id = f"{task['id']}:{time.monotonic_ns()}"
                    self.task_notification.emit(
                        attempt_id + ':due',
                        f"学习时间到了：{task['title']}（{task['minutes']:g} 分钟），正在准备开始。")
                    try:
                        self.begin(
                            task['title'], task['minutes'], task['id'],
                            task['break_every_minutes'], task['break_minutes'],
                            task['allow_offscreen'])
                    except Exception as exc:
                        self.store.fail_task(task['id'], str(exc))
                        self.notice.emit('任务启动失败：' + user_error_message(exc))
                        self.task_notification.emit(
                            attempt_id + ':failed',
                            f"任务“{task['title']}”启动失败：{user_error_message(exc)}")
                    next_publish = 0
                try:
                    command, values = self.commands.get_nowait()
                except queue.Empty:
                    command = None
                if command:
                    try:
                        self.execute(command, values)
                    except (ValueError, OSError, RuntimeError, sqlite3.Error) as exc:
                        self.notice.emit(user_error_message(exc))
                    finally:
                        if command in self.BUSY_COMMANDS:
                            try:
                                self.publish()
                            finally:
                                self.busy.emit(False)
                    next_publish = 0
                if self.service.state == 'monitoring' and self.source:
                    # Model startup/resume is deliberately outside observation time.
                    self.runtime_watch.check(time.monotonic(), time.time())
                    self.resource_tracker.sample(at)
                    try:
                        observation = self.source.read(time.monotonic())
                    except SourceStalled as exc:
                        if self.check_runtime_continuity():
                            self.publish()
                            continue
                        if self.evidence_recorder.finish(
                                self.store, self.service, self.source.frame, at):
                            self.invalidate_storage_status()
                        self.service.source_gap(time.monotonic(), utc_now(), str(exc))
                        self.notice.emit('摄像头断开，正在自动恢复（仅尝试一次）…')
                        try:
                            self.source.reconnect()
                        except Exception as reconnect_exc:
                            detail = f'{exc}；自动重连失败：{reconnect_exc}'
                            self.finish_session('interrupted', detail)
                            self.notice.emit(f'会话已中断：{detail}。可以重新开始。')
                        else:
                            if self.check_runtime_continuity():
                                self.publish()
                                continue
                            self.notice.emit('摄像头已恢复，会话继续。')
                        next_publish = 0
                    except Exception as exc:
                        if self.check_runtime_continuity():
                            self.publish()
                            continue
                        self.finish_session('interrupted', str(exc))
                        self.notice.emit(f'会话已中断：{exc}。可以重新开始。')
                        next_publish = 0
                    else:
                        if self.check_runtime_continuity():
                            self.publish()
                            next_publish = time.monotonic() + .5
                            continue
                        actions = self.service.process(observation, utc_now())
                        for action in actions:
                            if action.kind is ActionKind.ALERT:
                                reminder_name = (action.reason if
                                                 action.rule_id == 'focus_violation'
                                                 and action.reason else
                                                 NAMES.get(action.rule_id, action.rule_id))
                                self.reminder.emit(
                                    self.service.action_identifier(action),
                                    action.rule_id,
                                    '提醒：'+reminder_name)
                            if (action.rule_id != 'focus_violation' and capture_event_evidence(
                                self.store, self.service.identifier, self.source.frame, action
                            )):
                                self.invalidate_storage_status()
                        if self.evidence_recorder.update(
                                self.store, self.service, self.source.frame,
                                observation, actions):
                            self.invalidate_storage_status()
                if at >= next_publish:
                    self.publish()
                    next_publish = at + (.5 if self.ui_visible else 5.)
                if at >= next_purge:
                    self.store.purge(now)
                    next_purge = at + 3600
                self.msleep(50)
        except Exception as exc:
            failure_detail = str(exc).strip()[:500] or type(exc).__name__
            self.notice.emit(f'运行已停止：{exc}')
        finally:
            # Never start another prewarm while cleaning up, including fatal errors.
            self.requestInterruption()
            try:
                if self.service and self.service.identifier:
                    self.finish_session('interrupted', failure_detail)
                else:
                    self.stop_source()
            except Exception as exc:
                self.notice.emit(f'结束时发生错误：{exc}')
            finally:
                try:
                    if self.preloader:
                        self.preloader.close(wait=True)
                except Exception as exc:
                    self.notice.emit(f'释放视觉模型时发生错误：{exc}')
                finally:
                    try:
                        if self.store:
                            self.store.close()
                    finally:
                        if lock:
                            lock.close()


class ObserverWindow(QMainWindow):
    exit_ready = Signal()

    def __init__(self, args, source_factory=None):
        super().__init__()
        self.setWindowTitle('Observer · 模拟' if args.demo else 'Observer')
        available = QApplication.primaryScreen().availableGeometry()
        full_height_layout = available.height() >= 952
        self.setMinimumSize(500, 920 if full_height_layout else 650)
        self.resize(620, min(960, max(self.minimumHeight(), available.height() - 32)))
        self.closing = False
        self.exit_completed = False
        self.hidden_to_tray = False
        self.presented_reminders = set()
        self.presented_task_notifications = set()
        self.desktop_settings = QSettings(
            str(Path(args.data_dir).resolve() / 'desktop.ini'), QSettings.Format.IniFormat)
        self.blocked = True
        self.pending_command = None
        self.state = 'idle'
        self.worker = DesktopWorker(args, source_factory)
        self.tray = DesktopTray(self)
        self.tray.show_requested.connect(self.restore_window)
        self.tray.pause_requested.connect(
            lambda: self.send('resume' if self.state == 'paused' else 'pause'))
        self.tray.end_requested.connect(lambda: self.send('end'))
        self.tray.quit_requested.connect(self.request_exit)
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setSpacing(14)
        root_layout.addWidget(content, 1)
        header = QLabel('Observer')
        header.setStyleSheet('font-size: 24px; font-weight: 600;')
        layout.addWidget(header)
        if args.demo:
            subtitle_text = '模拟模式 · 不使用摄像头'
        elif args.profile:
            subtitle_text = '个人校准已启用 · 默认无摄像头预览'
        else:
            subtitle_text = '仅离席与手机 · 眼神/姿态提醒未启用'
        subtitle = QLabel(subtitle_text)
        subtitle.setStyleSheet('color: #64748b;')
        layout.addWidget(subtitle)
        tabs = self.tabs = QTabWidget()
        tabs.tabBar().setObjectName('mainPageTabs')
        layout.addWidget(tabs, 1)
        lifecycle = QHBoxLayout()
        self.lifecycle_hint = QLabel()
        self.lifecycle_hint.setWordWrap(True)
        self.lifecycle_hint.setStyleSheet('color:#64748b;')
        self.exit_button = QPushButton('退出程序')
        self.exit_button.clicked.connect(self.request_exit)
        lifecycle.addWidget(self.lifecycle_hint, 1)
        lifecycle.addWidget(self.exit_button)
        layout.addLayout(lifecycle)
        self.task_notifications = QCheckBox('定时任务系统通知（到点及启动失败）')
        self.task_notifications.setChecked(self.desktop_settings.value(
            'task_notifications', True, type=bool))
        self.task_notifications.toggled.connect(self.set_task_notifications)
        layout.addWidget(self.task_notifications)
        self.notification_status = QLabel(
            '任务通知已开启；Windows 可能阻止显示。' if self.task_notifications.isChecked()
            else '任务系统通知已关闭。')
        self.notification_status.setWordWrap(True)
        self.notification_status.setStyleSheet('color:#64748b;')
        layout.addWidget(self.notification_status)
        self.tray_watch = QTimer(self)
        self.tray_watch.setInterval(1000)
        self.tray_watch.timeout.connect(self.check_tray)
        self.tray_watch.start()
        self.check_tray()
        focus_page = QWidget()
        focus_layout = QVBoxLayout(focus_page)
        focus_layout.setContentsMargins(12, 16, 12, 12)
        focus_layout.setSpacing(14)
        form = QFormLayout()
        form.setVerticalSpacing(10)
        self.title = QLineEdit('专注')
        self.title.setObjectName('focusTaskTitle')
        self.title.setMinimumHeight(40)
        self.minutes = QSpinBox()
        self.minutes.setObjectName('focusDuration')
        self.minutes.setRange(1, 1440)
        self.minutes.setValue(25)
        self.minutes.setMinimumHeight(40)
        duration_box = QWidget()
        duration_layout = QHBoxLayout(duration_box)
        duration_layout.setContentsMargins(0, 0, 0, 0)
        duration_layout.setSpacing(8)
        duration_box.setMinimumHeight(40)
        duration_layout.addWidget(self.minutes, 1)
        self.duration_unit = QLabel('分钟')
        self.duration_unit.setStyleSheet('font-size:14px;')
        duration_layout.addWidget(self.duration_unit)
        form.addRow('任务', self.title)
        form.addRow('时长', duration_box)
        self.scheduled = QCheckBox('指定开始时间')
        self.scheduled.setObjectName('focusScheduled')
        self.scheduled.setMinimumHeight(40)
        self.scheduled.setStyleSheet('font-size:14px;')
        self.when = QDateTimeEdit(QDateTime.currentDateTime().addSecs(60))
        self.when.setObjectName('focusStartTime')
        self.when.setMinimumHeight(40)
        self.when.setCalendarPopup(True)
        self.when.setDisplayFormat('yyyy-MM-dd HH:mm:ss')
        self.when.setEnabled(False)
        self.scheduled.toggled.connect(self.when.setEnabled)
        form.addRow(self.scheduled, self.when)
        focus_layout.addLayout(form)
        self.status = QLabel('正在准备…')
        self.status.setWordWrap(True)
        self.status.setStyleSheet('font-size: 17px; font-weight: 600; padding: 10px 0;')
        self.status.setMinimumHeight(120)
        focus_layout.addWidget(self.status)
        self.alert_banner = QLabel('')
        self.alert_banner.setWordWrap(True)
        self.alert_banner.setStyleSheet(HARD_REMINDER_STYLE)
        self.alert_banner.hide()
        focus_layout.addWidget(self.alert_banner)
        self.health_banner_timer = QTimer(self)
        self.health_banner_timer.setSingleShot(True)
        self.health_banner_timer.timeout.connect(self.alert_banner.hide)
        buttons = QHBoxLayout()
        self.start = QPushButton('开始')
        self.pause = QPushButton('暂停')
        self.end = QPushButton('结束')
        for button in (self.start, self.pause, self.end):
            button.setMinimumHeight(38)
            buttons.addWidget(button)
        focus_layout.addLayout(buttons)
        self.start.clicked.connect(self.start_clicked)
        self.pause.clicked.connect(lambda: self.send('resume' if self.state == 'paused' else 'pause'))
        self.end.clicked.connect(lambda: self.send('end'))
        self.scheduled.toggled.connect(lambda _on: self.update_buttons())
        self.images = QCheckBox('保存事件截图（保留30天）')
        self.images.toggled.connect(self.images_changed)
        focus_layout.addWidget(self.images)
        self.test_evidence = QCheckBox('测试取证（违规期间每2秒截图）')
        self.test_evidence.toggled.connect(
            lambda enabled: self.worker.submit('test-evidence', enabled))
        focus_layout.addWidget(self.test_evidence)
        if args.demo:
            self.demo_signal = QComboBox()
            for label, signal in [('正常', 'normal'), ('可见手机', 'phone_held'), ('离席', 'away'), ('无法判断', 'unknown')]:
                self.demo_signal.addItem('模拟：'+label, signal)
            self.demo_signal.currentIndexChanged.connect(lambda: self.worker.submit('signal', self.demo_signal.currentData()))
            focus_layout.addWidget(self.demo_signal)
        if args.demo:
            message = '保持程序运行，定时任务才能执行；可关闭窗口隐藏到托盘。'
        elif args.profile:
            message = ('离屏、可见手机和离席按成立区间合并计时；'
                       '总分按工作时间比例计算。姿态提示不计分。')
        else:
            message = '未找到个人校准；当前只检测离席和可见手机。'
        self.message = QLabel(message)
        self.message.setWordWrap(True)
        focus_layout.addWidget(self.message)
        event_header = QLabel('本轮事件')
        event_header.setStyleSheet('font-weight:600; padding-top:4px;')
        focus_layout.addWidget(event_header)
        self.focus_events = QTextBrowser()
        self.focus_events.setOpenLinks(False)
        self.focus_events.setPlaceholderText('本轮尚无成立事件')
        self.focus_events.setMinimumHeight(130)
        self.focus_events.anchorClicked.connect(self.open_event_image)
        focus_layout.addWidget(self.focus_events, 1)
        if full_height_layout:
            tabs.addTab(focus_page, '专注')
        else:
            focus_scroll = QScrollArea()
            focus_scroll.setWidgetResizable(True)
            focus_scroll.setFrameShape(QFrame.Shape.NoFrame)
            focus_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            focus_scroll.setWidget(focus_page)
            tabs.addTab(focus_scroll, '专注')
        self.history = QListWidget()
        self.tasks = QListWidget()
        history_page = QWidget()
        history_layout = QVBoxLayout(history_page)
        self.data_status = QLabel('正在读取本地数据…')
        self.data_status.setWordWrap(True)
        self.data_status.setStyleSheet('color:#64748b;')
        history_layout.addWidget(self.data_status)
        history_layout.addWidget(self.history)
        history_buttons = QHBoxLayout()
        self.delete_history = QPushButton('删除选中记录')
        self.delete_history.clicked.connect(self.delete_selected_history)
        self.purge_history = QPushButton('清理过期记录')
        self.purge_history.clicked.connect(lambda: self.send('purge'))
        history_buttons.addWidget(self.delete_history)
        history_buttons.addWidget(self.purge_history)
        history_layout.addLayout(history_buttons)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlaceholderText('结束后显示摘要，也可以点击历史记录查看。')
        self.details.setMinimumHeight(150)
        history_layout.addWidget(self.details)
        tabs.addTab(history_page, '历史')
        advanced_page = QWidget()
        advanced_layout = QVBoxLayout(advanced_page)
        self.today_summary = QLabel('今日：正在读取…')
        self.today_summary.setWordWrap(True)
        self.today_summary.setStyleSheet('color:#475569;')
        advanced_layout.addWidget(self.today_summary)
        advanced_tabs = self.advanced_tabs = QTabWidget()
        recurring_page = QWidget()
        recurring_layout = QVBoxLayout(recurring_page)
        plan_form = QFormLayout()
        self.plan_title = QLineEdit('专注')
        self.plan_time = QTimeEdit(QTime(9, 0))
        self.plan_time.setDisplayFormat('HH:mm')
        self.plan_recurrence = QComboBox()
        self.plan_recurrence.addItem('每天', 'daily')
        self.plan_recurrence.addItem('工作日', 'weekdays')
        self.plan_recurrence.addItem('每周选定星期', 'weekly')
        self.plan_minutes = QSpinBox()
        self.plan_minutes.setRange(1, 1440)
        self.plan_minutes.setValue(25)
        self.plan_minutes.setSuffix(' 分钟')
        self.plan_timezone = get_localzone_name()
        self.plan_timezone_label = QLabel(self.plan_timezone)
        plan_form.addRow('名称', self.plan_title)
        plan_form.addRow('时间', self.plan_time)
        plan_form.addRow('重复', self.plan_recurrence)
        weekday_box = QWidget()
        weekday_layout = QHBoxLayout(weekday_box)
        weekday_layout.setContentsMargins(0, 0, 0, 0)
        weekday_layout.setSpacing(5)
        self.plan_weekdays = []
        for label in ('一', '二', '三', '四', '五', '六', '日'):
            checkbox = QCheckBox(label)
            self.plan_weekdays.append(checkbox)
            weekday_layout.addWidget(checkbox)
        plan_form.addRow('星期', weekday_box)
        plan_form.addRow('时长', self.plan_minutes)
        break_box = QWidget()
        break_layout = QHBoxLayout(break_box)
        break_layout.setContentsMargins(0, 0, 0, 0)
        self.plan_break_enabled = QCheckBox('每')
        self.plan_break_every = QSpinBox()
        self.plan_break_every.setRange(1, 1440)
        self.plan_break_every.setValue(50)
        self.plan_break_every.setSuffix(' 分钟')
        self.plan_break_minutes = QSpinBox()
        self.plan_break_minutes.setRange(1, 1439)
        self.plan_break_minutes.setValue(10)
        self.plan_break_minutes.setSuffix(' 分钟')
        break_layout.addWidget(self.plan_break_enabled)
        break_layout.addWidget(self.plan_break_every)
        break_layout.addWidget(QLabel('休息'))
        break_layout.addWidget(self.plan_break_minutes)
        plan_form.addRow('计划休息', break_box)
        self.plan_offscreen = QCheckBox('允许离屏查看资料（本次不计算离屏扣分）')
        plan_form.addRow('', self.plan_offscreen)
        plan_form.addRow('时区', self.plan_timezone_label)
        recurring_layout.addLayout(plan_form)
        self.plans = QListWidget()
        recurring_layout.addWidget(self.plans, 1)
        plan_buttons = QHBoxLayout()
        self.new_plan = QPushButton('新建')
        self.save_plan = QPushButton('保存')
        self.toggle_plan = QPushButton('停用')
        self.delete_plan = QPushButton('删除')
        for button in (self.new_plan, self.save_plan, self.toggle_plan, self.delete_plan):
            plan_buttons.addWidget(button)
        recurring_layout.addLayout(plan_buttons)
        advanced_tabs.addTab(recurring_page, '重复计划')

        task_page = QWidget()
        task_layout = QVBoxLayout(task_page)
        current_controls = QHBoxLayout()
        current_controls.addWidget(QLabel('当前任务延长'))
        self.extend_minutes = QSpinBox()
        self.extend_minutes.setRange(1, 240)
        self.extend_minutes.setValue(25)
        self.extend_session = QPushButton('延长本次')
        current_controls.addWidget(self.extend_minutes)
        self.extend_unit = QLabel('分钟')
        current_controls.addWidget(self.extend_unit)
        current_controls.addWidget(self.extend_session)
        task_layout.addLayout(current_controls)
        self.extend_feedback = QLabel('')
        self.extend_feedback.setWordWrap(True)
        task_layout.addWidget(self.extend_feedback)
        self.extend_guard_active = False
        self.extend_guard_timer = QTimer(self)
        self.extend_guard_timer.setSingleShot(True)
        self.extend_guard_timer.timeout.connect(self.release_extend_guard)
        task_layout.addWidget(self.tasks)
        task_buttons = QHBoxLayout()
        self.retry_task = QPushButton('重试选中任务')
        self.retry_task.clicked.connect(self.retry_selected)
        self.cancel_task = QPushButton('取消选中任务')
        self.cancel_task.clicked.connect(self.cancel_selected)
        task_buttons.addWidget(self.retry_task)
        task_buttons.addWidget(self.cancel_task)
        task_layout.addLayout(task_buttons)
        advanced_tabs.addTab(task_page, '执行记录')
        advanced_layout.addWidget(advanced_tabs)
        tabs.addTab(advanced_page, '高级计划')
        self.history.itemClicked.connect(lambda item: self.worker.submit('show', item.data(Qt.ItemDataRole.UserRole)))
        self.plans.itemClicked.connect(self.load_plan)
        self.plan_recurrence.currentIndexChanged.connect(self.update_plan_fields)
        self.plan_break_enabled.toggled.connect(self.update_plan_fields)
        self.new_plan.clicked.connect(self.reset_plan_form)
        self.save_plan.clicked.connect(self.save_plan_clicked)
        self.toggle_plan.clicked.connect(self.toggle_plan_clicked)
        self.delete_plan.clicked.connect(self.delete_plan_clicked)
        self.extend_session.clicked.connect(
            lambda: self.send('extend', self.extend_minutes.value()))
        self.editing_plan_id = None
        self.plan_rows = {}
        self.event_images = {}
        self.preview_event = None
        self.preview_index = 0
        self.focus_event_signature = None
        self.preview_previous_width = None
        self.preview_panel = QWidget()
        self.preview_panel.setMinimumWidth(420)
        self.preview_panel.setMaximumWidth(520)
        self.preview_panel.setStyleSheet('background:#ffffff; border-left:1px solid #cbd5e1;')
        preview_layout = QVBoxLayout(self.preview_panel)
        preview_layout.setContentsMargins(16, 18, 16, 18)
        preview_header = QHBoxLayout()
        self.preview_title = QLabel('事件图片')
        self.preview_title.setWordWrap(True)
        self.preview_title.setStyleSheet('font-size:15px; font-weight:600; border:none;')
        self.close_preview = QPushButton('关闭')
        self.close_preview.clicked.connect(self.close_event_preview)
        preview_header.addWidget(self.preview_title, 1)
        preview_header.addWidget(self.close_preview)
        preview_layout.addLayout(preview_header)
        self.preview_image = QLabel('')
        self.preview_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_image.setMinimumSize(380, 480)
        self.preview_image.setStyleSheet('background:#f1f5f9; border:1px solid #cbd5e1;')
        preview_layout.addWidget(self.preview_image, 1)
        preview_controls = QHBoxLayout()
        self.preview_previous = QPushButton('上一张')
        self.preview_previous.clicked.connect(lambda: self.change_preview_image(-1))
        self.preview_counter = QLabel('')
        self.preview_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_next = QPushButton('下一张')
        self.preview_next.clicked.connect(lambda: self.change_preview_image(1))
        preview_controls.addWidget(self.preview_previous)
        preview_controls.addWidget(self.preview_counter, 1)
        preview_controls.addWidget(self.preview_next)
        preview_layout.addLayout(preview_controls)
        self.preview_meta = QLabel('')
        self.preview_meta.setWordWrap(True)
        self.preview_meta.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.preview_meta.setStyleSheet('color:#475569; border:none;')
        preview_layout.addWidget(self.preview_meta)
        self.preview_panel.hide()
        root_layout.addWidget(self.preview_panel)
        self.setCentralWidget(root)
        self.setStyleSheet('QMainWindow {background:#f8fafc;} QWidget {font-family:"Microsoft YaHei UI"; font-size:13px;} '
                           'QTabBar#mainPageTabs::tab {padding:9px 22px; font-size:14px;} '
                           'QLineEdit#focusTaskTitle,QSpinBox#focusDuration,QDateTimeEdit#focusStartTime {font-size:15px; padding:2px 8px;} '
                           'QLineEdit,QSpinBox,QDateTimeEdit,QTimeEdit,QComboBox,QTextEdit,QTextBrowser,QListWidget {background:white; border:1px solid #cbd5e1; border-radius:5px; padding:5px;} '
                           'QPushButton {padding:7px 12px; border:1px solid #cbd5e1; border-radius:5px;} '
                           'QPushButton:disabled {color:#94a3b8; background:#e2e8f0;}')
        self.move(available.x() + (available.width() - self.width()) // 2,
                  available.y() + (available.height() - self.height()) // 2)
        self.worker.snapshot.connect(self.update_view)
        self.worker.notice.connect(self.show_notice)
        self.worker.reminder.connect(self.show_reminder)
        self.worker.task_notification.connect(self.show_task_notification)
        self.worker.detail.connect(self.details.setPlainText)
        self.worker.busy.connect(self.set_busy)
        self.worker.finished.connect(self.worker_finished)
        self.update_buttons()
        self.worker.start()
        self.reset_plan_form()

    def send(self, command, *values):
        if self.blocked or self.closing:
            return False
        if not self.worker.submit(command, *values):
            return False
        self.pending_command = command
        self.set_busy(True)
        status = {
            'start': '启动中 · 正在准备模型和摄像头…',
            'resume': '继续中 · 正在重新打开摄像头…',
            'task': '正在安排任务…',
            'pause': '正在暂停…',
            'end': '正在结束并保存…',
            'delete-session': '正在删除记录…',
            'purge': '正在清理过期记录…',
            'plan-save': '正在保存重复计划…',
            'plan-toggle': '正在更新重复计划…',
            'plan-delete': '正在删除重复计划…',
            'extend': '正在延长当前会话…',
        }.get(command)
        if status:
            self.status.setText(status)
        return True

    @Slot(str)
    def show_notice(self, message, reminder_rule_id=None):
        self.message.setText(message)
        extend_success = message.startswith('当前会话已延长')
        if extend_success or self.pending_command == 'extend':
            succeeded = extend_success
            self.extend_feedback.setText(message)
            self.extend_feedback.setStyleSheet(
                'color:#047857; font-weight:600;' if succeeded
                else 'color:#b91c1c; font-weight:600;')
            if succeeded:
                self.extend_guard_active = True
                self.extend_guard_timer.start(1500)
            self.update_buttons()
        if message.startswith('提醒：'):
            rule_id = reminder_rule_id or 'phone_held'
            health = rule_id in HEALTH_REMINDER_RULES
            self.message.setStyleSheet(
                'color:#0f766e; font-weight:500;' if health
                else 'color:#b91c1c; font-weight:600;')
            self.alert_banner.setStyleSheet(
                HEALTH_REMINDER_STYLE if health else HARD_REMINDER_STYLE)
            self.alert_banner.setText(message + '（已记录）')
            self.alert_banner.show()
            play_reminder_sound(rule_id)
            if health:
                self.health_banner_timer.start(HEALTH_REMINDER_BANNER_MS)
            else:
                self.health_banner_timer.stop()
                QApplication.alert(self, 3000)
        else:
            self.message.setStyleSheet('')

    @Slot(str, str, str)
    def show_reminder(self, action_id, rule_id, message):
        if action_id in self.presented_reminders:
            return
        # IDs originate from committed actions. Reserve before any side effect.
        self.presented_reminders.add(action_id)
        try:
            self.show_notice(message, rule_id)
        except Exception as exc:
            self.message.setText(f'{message}（呈现失败：{exc}）')
            self.message.setStyleSheet('color:#b91c1c; font-weight:600;')
            self.worker.submit('notification', action_id, False, str(exc))
        else:
            self.worker.submit('notification', action_id, True, None)

    @Slot(bool)
    def set_task_notifications(self, enabled):
        self.desktop_settings.setValue('task_notifications', enabled)
        self.desktop_settings.sync()
        if self.desktop_settings.status() != QSettings.Status.NoError:
            self.notification_status.setText('通知设置保存失败；本次运行有效。')
        else:
            self.notification_status.setText(
                '任务通知已开启；Windows 可能阻止显示。' if enabled
                else '任务系统通知已关闭。')

    @Slot(str, str)
    def show_task_notification(self, identifier, message):
        if identifier in self.presented_task_notifications:
            return
        self.presented_task_notifications.add(identifier)
        if self.task_notifications.isChecked() and not self.closing:
            self.notification_status.setText(self.tray.notify(message))

    @Slot()
    def start_clicked(self):
        self.health_banner_timer.stop()
        self.alert_banner.hide()
        self.alert_banner.clear()
        self.extend_feedback.clear()
        if self.scheduled.isChecked():
            when = datetime.fromtimestamp(self.when.dateTime().toSecsSinceEpoch()).astimezone()
            self.send('task', self.title.text(), when, self.minutes.value())
        else:
            self.send('start', self.title.text(), self.minutes.value())

    @Slot()
    def cancel_selected(self):
        item = self.tasks.currentItem()
        if item:
            self.worker.submit('cancel', item.data(Qt.ItemDataRole.UserRole))

    @Slot()
    def retry_selected(self):
        item = self.tasks.currentItem()
        if item:
            self.worker.submit('retry', item.data(Qt.ItemDataRole.UserRole))

    @Slot()
    def reset_plan_form(self):
        self.editing_plan_id = None
        self.plans.clearSelection()
        self.plan_title.setText('专注')
        self.plan_time.setTime(QTime(9, 0))
        self.plan_recurrence.setCurrentIndex(0)
        for checkbox in self.plan_weekdays:
            checkbox.setChecked(False)
        self.plan_minutes.setValue(25)
        self.plan_break_enabled.setChecked(False)
        self.plan_break_every.setValue(50)
        self.plan_break_minutes.setValue(10)
        self.plan_offscreen.setChecked(False)
        self.plan_timezone = get_localzone_name()
        self.plan_timezone_label.setText(self.plan_timezone)
        self.update_plan_fields()
        self.update_buttons()

    @Slot(QListWidgetItem)
    def load_plan(self, item):
        identifier = item.data(Qt.ItemDataRole.UserRole)
        plan = self.plan_rows.get(identifier)
        if not plan:
            return
        self.editing_plan_id = identifier
        self.plan_title.setText(plan['title'])
        hour, minute = (int(value) for value in plan['local_time'].split(':'))
        self.plan_time.setTime(QTime(hour, minute))
        self.plan_recurrence.setCurrentIndex(
            self.plan_recurrence.findData(plan['recurrence']))
        for day, checkbox in enumerate(self.plan_weekdays):
            checkbox.setChecked(day in plan['weekdays'])
        self.plan_minutes.setValue(round(plan['minutes']))
        has_break = plan['break_every_minutes'] is not None
        self.plan_break_enabled.setChecked(has_break)
        if has_break:
            self.plan_break_every.setValue(round(plan['break_every_minutes']))
            self.plan_break_minutes.setValue(round(plan['break_minutes']))
        self.plan_offscreen.setChecked(plan['allow_offscreen'])
        self.plan_timezone = plan['timezone']
        self.plan_timezone_label.setText(self.plan_timezone)
        self.update_plan_fields()
        self.update_buttons()

    @Slot()
    def update_plan_fields(self):
        weekly = self.plan_recurrence.currentData() == 'weekly'
        for checkbox in self.plan_weekdays:
            checkbox.setEnabled(weekly and not self.blocked and not self.closing)
        breaks = self.plan_break_enabled.isChecked()
        self.plan_break_every.setEnabled(breaks and not self.blocked and not self.closing)
        self.plan_break_minutes.setEnabled(breaks and not self.blocked and not self.closing)

    @Slot()
    def save_plan_clicked(self):
        weekdays = tuple(day for day, checkbox in enumerate(self.plan_weekdays)
                         if checkbox.isChecked())
        every = self.plan_break_every.value() if self.plan_break_enabled.isChecked() else None
        rest = self.plan_break_minutes.value() if self.plan_break_enabled.isChecked() else None
        self.send(
            'plan-save', self.editing_plan_id, self.plan_title.text(),
            self.plan_time.time().toString('HH:mm'), self.plan_timezone,
            self.plan_recurrence.currentData(), self.plan_minutes.value(), weekdays,
            every, rest, self.plan_offscreen.isChecked(),
        )

    @Slot()
    def toggle_plan_clicked(self):
        plan = self.plan_rows.get(self.editing_plan_id)
        if plan:
            self.send('plan-toggle', self.editing_plan_id, not plan['enabled'])

    @Slot()
    def delete_plan_clicked(self):
        if self.editing_plan_id:
            self.send('plan-delete', self.editing_plan_id)

    @Slot()
    def delete_selected_history(self):
        item = self.history.currentItem()
        if item:
            self.send('delete-session', item.data(Qt.ItemDataRole.UserRole))

    @Slot(bool)
    def images_changed(self, enabled):
        self.worker.submit('images', enabled)
        self.update_buttons()

    @Slot(object)
    def open_event_image(self, link):
        href = link.toString() if hasattr(link, 'toString') else str(link)
        prefix = 'evidence:'
        if not href.startswith(prefix):
            return
        event_id = href[len(prefix):]
        item = self.event_images.get(event_id)
        if not item:
            self.show_notice('事件图片链接已失效')
            return
        self.preview_event = item
        self.preview_index = 0
        self.show_preview_image()

    def show_preview_image(self):
        item = self.preview_event
        if not item:
            return
        images = item.get('images') or []
        if not images and item.get('image_path'):
            images = [{'image_path': item['image_path'],
                       'sha256': item.get('image_sha256'), 'kind': 'legacy',
                       'offset': item.get('offset'), 'signals': item.get('signals', [])}]
        if not images:
            self.show_notice('事件图片链接已失效')
            return
        self.preview_index = max(0, min(self.preview_index, len(images)-1))
        record = images[self.preview_index]
        pixmap = QPixmap(record['image_path'])
        if pixmap.isNull():
            self.show_notice('事件图片无法读取')
            return
        if self.preview_panel.isHidden():
            self.preview_previous_width = self.width()
            self.preview_panel.show()
            self.resize(max(1080, self.width()+440), self.height())
        scaled = pixmap.scaled(
            480, 650, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_image.setPixmap(scaled)
        self.preview_counter.setText(f'{self.preview_index+1} / {len(images)}')
        self.preview_previous.setEnabled(self.preview_index > 0)
        self.preview_next.setEnabled(self.preview_index+1 < len(images))
        self.preview_title.setText(
            f"{NAMES.get(item['rule_id'], item['rule_id'])} · {item['offset']:.1f} 秒")
        dimensions = f'{pixmap.width()}×{pixmap.height()}'
        digest = record.get('sha256') or '—'
        kind = {'start': '开始', 'sample': '过程', 'end': '结束',
                'event': '事件', 'legacy': '历史证据'}.get(record.get('kind'), '事件')
        offset = record.get('offset')
        offset_text = '—' if offset is None else f'{float(offset):.1f} 秒'
        signal_text = ' + '.join(
            NAMES.get(signal, signal) for signal in record.get('signals', [])) or '—'
        self.preview_meta.setText(
            f"{kind} · 会话内 {offset_text} · {signal_text}\n"
            f"事件 {item['event_id']}\n原图 {dimensions}\nSHA-256 {digest}")

    def change_preview_image(self, step):
        if not self.preview_event:
            return
        self.preview_index += step
        self.show_preview_image()

    @Slot()
    def close_event_preview(self):
        if self.preview_panel.isHidden():
            return
        self.preview_panel.hide()
        self.preview_image.clear()
        self.preview_counter.clear()
        self.preview_meta.clear()
        self.preview_event = None
        self.preview_index = 0
        if self.preview_previous_width:
            self.resize(self.preview_previous_width, self.height())
        self.preview_previous_width = None

    def update_focus_events(self, result, events, active=False):
        session_id = result.get('id') if result else None
        previous_session_id = (self.focus_event_signature[0]
                               if self.focus_event_signature else None)
        scroll_bar = self.focus_events.verticalScrollBar()
        scroll_position = scroll_bar.value()
        signature = (
            session_id,
            active,
            tuple((item['event_id'], item['rule_id'], item['offset'],
                   item['resolved_offset'], item['points'], item['delivery_state'],
                   item['image_path'], item['image_status'],
                   tuple((image.get('id'), image.get('sha256'))
                         for image in item.get('images', [])),
                   tuple(item.get('signals', [])), item.get('violation_seconds'))
                  for item in events),
        )
        if signature == self.focus_event_signature:
            return
        self.focus_event_signature = signature
        self.event_images = {
            item['event_id']: item for item in events if item.get('image_path')
        }
        if not result:
            self.focus_events.setHtml('<span style="color:#64748b">本轮尚未开始</span>')
            self.close_event_preview()
            return
        heading = ('本轮' if active else '最近一轮') + '：' + html.escape(result['title'])
        if not active:
            score = result.get('score')
            score_text = f'{score:g} 分' if score is not None else '证据不足'
            coverage = result.get('coverage')
            coverage_text = f'{coverage:.0%}' if coverage is not None else '—'
            if result.get('score_version') == 'time_ratio_v2':
                heading += (
                    f' · 总分：{score_text}'
                    f' · 违规：{result.get("violation_seconds", 0):.1f}秒'
                    f' · 覆盖：{coverage_text}')
            else:
                heading += (
                    f' · 总分：{score_text}'
                    f' · 旧规则扣分：{result.get("deducted", 0):g}'
                    f' · 覆盖：{coverage_text}')
        lines = [f'<p><b>{heading}</b></p>']
        if not events:
            lines.append('<p style="color:#64748b">尚无成立事件</p>')
        for index, item in enumerate(events, 1):
            if item.get('test_candidate'):
                name = html.escape(NAMES['focus_candidate'])
            elif item.get('signals'):
                name = html.escape(' + '.join(
                    NAMES.get(signal, signal) for signal in item['signals']))
            else:
                name = html.escape(NAMES.get(item['rule_id'], item['rule_id']))
            minute, second = divmod(item['offset'], 60)
            details = [f'{int(minute):02d}:{second:04.1f}', name]
            if item.get('violation_seconds') is not None:
                details.append(f"违规 {item['violation_seconds']:.1f} 秒")
            elif item.get('test_candidate'):
                details.append('测试取证')
            elif item['points']:
                details.append(f"扣 {item['points']:g} 分")
            elif item['delivery_state']:
                details.append('提醒')
            else:
                details.append(
                    '已记录（姿态提醒冷却中）'
                    if item['rule_id'] == 'posture_deviation'
                    else '已记录（健康提醒冷却中）')
            if item['resolved_offset'] is not None:
                duration = item['resolved_offset'] - item['offset']
                details.append(f'{duration:.1f}秒后恢复')
            label = ' · '.join(html.escape(value) for value in details)
            if item.get('image_path'):
                image = (f'<a href="evidence:{item["event_id"]}">查看'
                         f' {len(item.get("images") or [item])}张</a>')
            else:
                image = f'<span style="color:#94a3b8">{html.escape(item["image_status"])}</span>'
            lines.append(f'<p>{index}. {label}　{image}</p>')
        self.focus_events.setHtml(''.join(lines))
        if session_id == previous_session_id:
            # Evidence samples and live durations rebuild the document. Keep
            # the user's reading position instead of QTextBrowser's default top.
            scroll_bar.setValue(scroll_position)
        if not self.event_images:
            self.close_event_preview()

    @Slot(bool)
    def set_busy(self, busy):
        if busy and self.pending_command is None:
            self.pending_command = ('resume' if self.state in ('paused', 'resting')
                                    else 'start')
        self.blocked = busy
        if not busy:
            self.pending_command = None
        self.update_buttons()

    @Slot()
    def release_extend_guard(self):
        self.extend_guard_active = False
        self.update_buttons()

    def update_buttons(self):
        available = not self.blocked and not self.closing
        pending = self.pending_command
        if pending == 'start':
            self.start.setText('启动中…')
        elif pending == 'task':
            self.start.setText('安排中…')
        else:
            self.start.setText('安排任务' if self.scheduled.isChecked() else '开始')
        if pending == 'resume':
            self.pause.setText('继续中…')
        elif pending == 'pause':
            self.pause.setText('暂停中…')
        else:
            self.pause.setText('继续' if self.state == 'paused' else '暂停')
        self.end.setText('结束中…' if pending == 'end' else '结束')
        self.start.setEnabled(available and self.state == 'idle')
        self.pause.setEnabled(available and self.state in ('monitoring', 'paused'))
        self.end.setEnabled(available and self.state in ('monitoring', 'paused', 'resting'))
        self.exit_button.setEnabled(not self.closing)
        self.tray.update(
            '正在退出' if self.closing else STATES.get(self.state, self.state),
            paused=self.state == 'paused', can_pause=self.pause.isEnabled(),
            can_end=self.end.isEnabled(), closing=self.closing)
        for button in (self.start, self.pause, self.end):
            button.setStyleSheet(
                ACTION_BUTTON_ENABLED_STYLE if button.isEnabled()
                else ACTION_BUTTON_DISABLED_STYLE)
        self.delete_history.setEnabled(available)
        self.purge_history.setEnabled(available)
        editable = available and self.state == 'idle'
        self.title.setEnabled(editable)
        self.minutes.setEnabled(editable)
        self.scheduled.setEnabled(editable)
        self.when.setEnabled(editable and self.scheduled.isChecked())
        self.images.setEnabled(available)
        self.test_evidence.setEnabled(available and self.images.isChecked())
        for widget in (self.plan_title, self.plan_time, self.plan_recurrence,
                       self.plan_minutes, self.plan_break_enabled, self.plan_offscreen):
            widget.setEnabled(available)
        self.new_plan.setEnabled(available)
        self.save_plan.setEnabled(available)
        selected_plan = self.plan_rows.get(self.editing_plan_id)
        self.toggle_plan.setEnabled(available and selected_plan is not None)
        self.delete_plan.setEnabled(available and selected_plan is not None)
        self.toggle_plan.setText(
            '停用' if selected_plan is None or selected_plan['enabled'] else '启用')
        self.retry_task.setEnabled(available)
        self.cancel_task.setEnabled(available)
        self.extend_session.setText('延长中…' if pending == 'extend' else '延长本次')
        can_extend = (available and not self.extend_guard_active
                      and self.state in ('monitoring', 'paused', 'resting'))
        self.extend_minutes.setEnabled(can_extend)
        self.extend_session.setEnabled(can_extend)
        self.update_plan_fields()

    @staticmethod
    def fill_list(widget, rows):
        current = widget.currentItem()
        selected = current.data(Qt.ItemDataRole.UserRole) if current else None
        signature = [(r[0], r[1]) for r in rows]
        if getattr(widget, '_signature', None) == signature:
            return
        widget._signature = signature
        widget.clear()
        for identifier, text in rows:
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, identifier)
            widget.addItem(item)
            if identifier == selected:
                widget.setCurrentItem(item)

    @Slot(dict)
    def update_view(self, snapshot):
        self.state = snapshot['state']
        self.update_buttons()
        current = snapshot['current']
        if current:
            extended = current.get('extended_minutes', 0)
            if extended and not self.extend_feedback.text():
                self.extend_feedback.setText(f'本次累计延长 {extended:g} 分钟')
                self.extend_feedback.setStyleSheet('color:#047857; font-weight:600;')
            resting = self.state == 'resting'
            seconds = int(snapshot['break_remaining'] if resting else snapshot['remaining'])
            score = f"{current['score']:g} 分" if current['score'] is not None else '证据不足'
            timer_name = '休息剩余' if resting else '剩余'
            text = (f"{STATES[self.state]} · {current['title']}\n"
                    f"{timer_name} {seconds//60:02}:{seconds%60:02} · {score}")
            if self.worker.args.demo or self.worker.args.profile:
                attention = snapshot.get('signals', {}).get('attention_away', 'unknown')
                label = {'compliant': '屏内', 'violation': '屏外候选',
                         'unknown': '不可判断', 'exempt': '已暂停'}.get(attention, attention)
                text += f"\n头眼：{label}"
            text += ('\n其他事件：' + live_other_event_text(
                snapshot.get('signals', {}), self.state))
            self.status.setText(text)
        else:
            self.status.setText('等待开始')
            self.extend_feedback.clear()
        self.update_focus_events(
            snapshot.get('focus_session'), snapshot.get('focus_events', []),
            active=current is not None,
        )
        self.images.blockSignals(True)
        self.images.setChecked(snapshot['images'])
        self.images.blockSignals(False)
        self.test_evidence.blockSignals(True)
        self.test_evidence.setChecked(snapshot.get('test_evidence', False))
        self.test_evidence.blockSignals(False)
        self.test_evidence.setEnabled(self.pending_command is None and self.images.isChecked())
        self.data_status.setText(data_status_text(snapshot['data_status']))
        today = snapshot.get('today') or {}
        observed_minutes = today.get('observed_seconds', 0) / 60
        break_minutes = today.get('break_seconds', 0) / 60
        coverage = ('—' if today.get('coverage') is None
                    else f"{today['coverage']:.0%}")
        self.today_summary.setText(
            f"今日：计划 {today.get('planned', 0)} · 完成 {today.get('completed', 0)} · "
            f"观察 {observed_minutes:.0f} 分钟 · 休息 {break_minutes:.0f} 分钟 · "
            f"违规 {today.get('violation_seconds', 0):.0f} 秒 · "
            f"得分 {today.get('score') if today.get('score') is not None else '—'} · "
            f"覆盖 {coverage}" +
            (f" · 旧规则扣分 {today.get('legacy_deducted', 0):g}"
             if today.get('legacy_deducted') else ''))
        if hasattr(self, 'demo_signal'):
            self.demo_signal.setEnabled(self.state == 'monitoring')
            self.demo_signal.blockSignals(True)
            self.demo_signal.setCurrentIndex(self.demo_signal.findData(snapshot['demo_signal']))
            self.demo_signal.blockSignals(False)
        def local(stamp):
            return datetime.fromisoformat(stamp).astimezone().strftime('%m-%d %H:%M')
        self.fill_list(self.history, [(r['id'], f"{local(r['started'])}  {r['title']} · {STATES.get(r['state'],r['state'])}" +
                                      {'demo': ' · 模拟', 'real': '', 'unknown': ' · 身份异常'}[r['mode']])
                                     for r in snapshot['history']])
        self.fill_list(self.tasks, [(r['id'], f"{local(r['starts'])}  {r['title']} · {STATES.get(r['state'],r['state'])}" +
                                    (f"\n{user_error_message(r['error'])[:120]}"
                                     if r.get('error') else ''))
                                   for r in snapshot['tasks'] if r['state'] in ('pending', 'conflict', 'failed', 'missed')])
        self.plan_rows = {plan['id']: plan for plan in snapshot.get('plans', [])}
        recurrence_names = {'daily': '每天', 'weekdays': '工作日', 'weekly': '每周'}
        self.fill_list(self.plans, [
            (plan['id'],
             f"{plan['title']} · {recurrence_names[plan['recurrence']]} {plan['local_time']}"
             f" · {'启用' if plan['enabled'] else '停用'}")
            for plan in snapshot.get('plans', [])
        ])
        if self.editing_plan_id and self.editing_plan_id not in self.plan_rows:
            self.reset_plan_form()
        else:
            self.update_buttons()

    @Slot()
    def worker_finished(self):
        self.blocked = True
        self.update_buttons()
        if self.closing:
            self.close()
        else:
            self.status.setText('运行已停止，请关闭后重新打开')
            self.tray.close()
            self.tray_watch.stop()
            self.lifecycle_hint.setText('运行已停止；关闭窗口即可退出程序。')
            self.restore_window()

    @Slot()
    def restore_window(self):
        self.hidden_to_tray = False
        self.setWindowState(self.windowState() & ~Qt.WindowState.WindowMinimized)
        self.show()
        self.raise_()
        self.activateWindow()

    def hideEvent(self, event):
        if self.worker.isRunning() and not self.closing:
            self.worker.submit('visibility', False)
        super().hideEvent(event)

    def showEvent(self, event):
        if self.worker.isRunning() and not self.closing:
            self.worker.submit('visibility', True)
        super().showEvent(event)

    @Slot()
    def check_tray(self):
        if self.closing:
            return
        available = self.tray.available
        self.lifecycle_hint.setText(
            '关闭窗口后仍在托盘运行；停止观察请暂停，完全退出请点“退出程序”。'
            if available else '系统托盘不可用；关闭窗口将结束会话并退出程序。')
        if self.hidden_to_tray and not available:
            self.restore_window()

    def begin_exit(self):
        if self.closing:
            return
        self.closing = True
        self.blocked = True
        self.tray_watch.stop()
        self.update_buttons()
        self.lifecycle_hint.setText('正在退出，请等待摄像头和视觉模型释放完成。')
        self.message.setText('正在结束会话并释放摄像头和视觉模型…')
        self.worker.requestInterruption()
        self.restore_window()

    @Slot()
    def request_exit(self):
        self.begin_exit()
        self.close()

    def closeEvent(self, event):
        if self.worker.isRunning():
            if not self.closing and self.tray.available:
                self.hidden_to_tray = True
                self.hide()
            else:
                self.begin_exit()
            event.ignore()
        else:
            self.closing = True
            self.tray_watch.stop()
            self.tray.close()
            event.accept()
            if not self.exit_completed:
                self.exit_completed = True
                self.exit_ready.emit()


def launch(args):
    app = QApplication.instance() or QApplication([])
    app.setStyle('Fusion')
    app.setQuitOnLastWindowClosed(False)
    if SingleInstanceWake.wake_existing(args.data_dir):
        return 0
    wake = SingleInstanceWake(args.data_dir, app)
    wake.start()
    window = ObserverWindow(args)
    power_filter = None
    if os.name == 'nt' and app.platformName() == 'windows':
        power_filter = WindowsPowerEventFilter(window.worker.note_power_transition)
        app.installNativeEventFilter(power_filter)
    wake.activation_requested.connect(window.restore_window)
    window.exit_ready.connect(app.quit)
    window.exit_ready.connect(wake.close)
    app.commitDataRequest.connect(window.request_exit)
    window.show()
    try:
        return app.exec()
    finally:
        if power_filter is not None:
            app.removeNativeEventFilter(power_filter)
        # Also drain the owner if Qt's loop ends due to OS/session shutdown.
        window.worker.requestInterruption()
        window.worker.wait()
        window.tray.close()
