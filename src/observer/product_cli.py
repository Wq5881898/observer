"""Minimal interactive console; the same SessionService can serve a future UI."""
from __future__ import annotations

import argparse
import hashlib
from importlib import resources as package_resources
import json
import math
import queue
import shlex
import sqlite3
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from observer.focus import FOCUS_RULE_ID, FOCUS_SIGNALS
from observer.observations import Observation, SignalState, SignalValue
from observer.config import load_config
from observer.product import NAMES, ProductStore, RuntimeLock, SessionService, utc_now, utc_text
from observer.resources import SessionResourceTracker
from observer.rules import ActionKind, RulePhase
from observer.deployment import default_data_dir, default_models_dir, default_profile, source_identity

FROZEN_MODEL = Path('models/yolo/observer-phone-s304-e2b.onnx')
FROZEN_SHA = 'acdfa05d0c1cc4d626f84fd010f61cd17d62636b95d9f71fe20f22ad281d23f5'
FROZEN_MODEL_SIZE = 10_565_246
FROZEN_MANIFEST_SHA = 'b049addf45643a8b2feff4281be0a1aede74ab45c7d49aa2f95a930572c9fb59'
FROZEN_CONFIG_SHA = '67a2d6a2890d8495147ab00390c02e7b329eb608829f7d3c796c214f57b41e06'
FROZEN_PHONE_SCORE_THRESHOLD = .25
FROZEN_PHONE_RETAIN_SECONDS = 1.5
FROZEN_PHONE_CENTER_DISTANCE = .18
FROZEN_PHONE_CONFIRMATION_HITS = 2
# Enabled only by a build that also pins and delivers the trained verifier.
FROZEN_PHONE_VERIFIER_SHA: str | None = '73edae8868ebf8788fa235ce3bab9e12a97952ec8e22a2874486e56eec763469'
DEFAULT_PROFILE = Path('data/calibration/default.json')
STALE_FRAME_SECONDS = 5.0
FRESH_FRAME_SECONDS = .75
CAMERA_FRAME_WAIT_SECONDS = 5.0
LONG_RESOURCE_GATE_SECONDS = 3600.0
INITIAL_VISION_WARMUP_SECONDS = 3.0


class SourceStalled(RuntimeError):
    """A running source stopped producing fresh frames and may be reconnected."""


def wait_for_camera_frame(camera, *, timeout_seconds=None, cancelled=None):
    """Require an actual fresh frame, not merely an opened camera handle."""
    if timeout_seconds is None:
        timeout_seconds = CAMERA_FRAME_WAIT_SECONDS
    deadline = time.monotonic() + timeout_seconds
    while True:
        if cancelled and cancelled():
            raise SourceStalled('摄像头准备已取消')
        now = time.monotonic()
        frame = camera.buffer.latest()
        if (camera.stats.opened and frame is not None
                and 0 <= now - frame.captured_monotonic <= FRESH_FRAME_SECONDS):
            return frame
        if now >= deadline:
            raise SourceStalled(camera.stats.last_error or '摄像头已打开，但没有新鲜画面')
        time.sleep(min(.05, deadline - now))


def capture_event_evidence(store, session_id: str, frame, action) -> bool:
    """Persist one opt-in frame for every event when it first becomes active."""
    capture_kind = (ActionKind.ALERT if action.rule_id == FOCUS_RULE_ID
                    else ActionKind.ACTIVATED)
    if action.kind is not capture_kind or not store.setting('images'):
        return False
    store.capture(session_id, action.event_id, frame)
    return True


class FocusEvidenceRecorder:
    """Capture one focus episode sparsely in production or every two seconds in tests."""
    interval_seconds = 2.0

    def __init__(self):
        self.reset()

    def reset(self):
        self.group_id = None
        self.event_id = None
        self.confirmed = False
        self.test_mode = False
        self.signal_starts = {}
        self.start_frame = None
        self.start_offset = None
        self.start_signals = ()
        self.next_capture_at = None

    @staticmethod
    def _copy_frame(frame):
        return frame.copy() if frame is not None and hasattr(frame, 'copy') else frame

    @staticmethod
    def _violation_signals(observation, scoring_signals):
        return tuple(
            signal for signal in FOCUS_SIGNALS
            if signal in scoring_signals
            and observation.signal(signal).state is SignalState.VIOLATION
        )

    def _capture(self, store, service, frame, at, kind, signals):
        event_id = self.event_id or self.group_id
        return store.capture(
            service.identifier, event_id, frame,
            rule_id=FOCUS_RULE_ID if self.confirmed else 'focus_candidate',
            kind=kind, offset=max(0., at-service.started), signals=signals,
            capture_id=str(uuid4()),
        )

    def update(self, store, service, frame, observation, actions) -> bool:
        if not service.identifier or service.focus is None or not store.setting('images'):
            self.reset()
            return False
        scoring_signals = service.focus.scoring_signals
        phases = service.engine.phases
        active = any(
            phases.get(signal, RulePhase.CLEAR) is not RulePhase.CLEAR
            for signal in scoring_signals
        )
        at = observation.monotonic_seconds
        signals = self._violation_signals(observation, scoring_signals)
        offset = max(0., at-service.started)
        for signal in scoring_signals:
            phase = phases.get(signal, RulePhase.CLEAR)
            if phase is RulePhase.CLEAR:
                self.signal_starts.pop(signal, None)
            elif signal not in self.signal_starts:
                self.signal_starts[signal] = (
                    self._copy_frame(frame), offset, signals)
        captured = False
        if active and self.group_id is None:
            self.group_id = str(uuid4())
            self.test_mode = bool(store.setting('test_evidence'))
            self.start_frame = self._copy_frame(frame)
            self.start_offset = offset
            self.start_signals = signals
            self.next_capture_at = at + self.interval_seconds
            if self.test_mode:
                captured |= self._capture(
                    store, service, self.start_frame, at, 'start', signals)

        activated = next((action for action in actions
                          if action.rule_id == FOCUS_RULE_ID
                          and action.kind is ActionKind.ACTIVATED), None)
        if activated is not None:
            if self.group_id is None:
                self.group_id = str(uuid4())
                self.test_mode = bool(store.setting('test_evidence'))
                self.start_frame = self._copy_frame(frame)
                self.start_offset = max(0., activated.event_started_monotonic-service.started)
                self.start_signals = signals
                self.next_capture_at = at + self.interval_seconds
            previous = self.group_id
            self.event_id = activated.event_id
            self.confirmed = True
            if self.test_mode:
                store.reassign_evidence_event(
                    service.identifier, previous, self.event_id, FOCUS_RULE_ID)
            else:
                confirmed_signals = (
                    service.focus.current.signals
                    if service.focus is not None and service.focus.current is not None
                    else set(signals)
                )
                confirmed_starts = [
                    self.signal_starts[signal]
                    for signal in confirmed_signals
                    if signal in self.signal_starts
                ]
                if confirmed_starts:
                    (self.start_frame, self.start_offset,
                     self.start_signals) = min(
                        confirmed_starts, key=lambda item: item[1])
                captured |= store.capture(
                    service.identifier, self.event_id, self.start_frame,
                    rule_id=FOCUS_RULE_ID, kind='start', offset=self.start_offset,
                    signals=self.start_signals, capture_id=str(uuid4()),
                )

        if (active and self.test_mode and self.group_id is not None
                and self.next_capture_at is not None and at >= self.next_capture_at):
            captured |= self._capture(store, service, frame, at, 'sample', signals)
            self.next_capture_at = at + self.interval_seconds

        if not active and self.group_id is not None:
            if self.test_mode or self.confirmed:
                captured |= self._capture(store, service, frame, at, 'end', signals)
            self.reset()
        return captured

    def finish(self, store, service, frame, at) -> bool:
        if self.group_id is None:
            return False
        captured = False
        if store.setting('images') and (self.test_mode or self.confirmed):
            captured = self._capture(store, service, frame, at, 'end', ())
        self.reset()
        return captured


def apply_default_profile(args, candidate: Path | None = None):
    """Use this machine's saved calibration unless the operator opts out."""
    if args.demo or args.profile or getattr(args, 'no_profile', False):
        return args.profile
    if candidate is None:
        candidate = default_profile()
    if candidate.is_file():
        args.profile = candidate
    return args.profile


def load_frozen_config(path: Path | None = None):
    if path is None:
        raw = package_resources.files('observer').joinpath('default_config.json').read_bytes()
    else:
        raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != FROZEN_CONFIG_SHA:
        raise ValueError('默认配置与 C 冻结身份不一致；未打开摄像头')
    return load_config(path)


def verify_frozen_models(models_dir: Path):
    from observer.models import verify_model_directory, sha256_file
    model_path = models_dir / 'yolo' / FROZEN_MODEL.name
    if model_path.stat().st_size != FROZEN_MODEL_SIZE:
        raise ValueError('手机模型大小与 C 冻结身份不一致；未打开摄像头')
    if sha256_file(model_path).lower() != FROZEN_SHA:
        raise ValueError('手机模型与 C 冻结哈希不一致；未打开摄像头')
    manifest_path = models_dir / 'manifest.json'
    if sha256_file(manifest_path).lower() != FROZEN_MANIFEST_SHA:
        raise ValueError('模型清单与 C 冻结身份不一致；未打开摄像头')
    return verify_model_directory(models_dir), model_path, manifest_path


@dataclass(slots=True)
class PreparedCameraResources:
    models: object
    model_path: Path
    manifest_path: Path
    config: object
    profile: object
    probe: object | None
    profile_sha256: str | None
    manifest_sha256: str
    timings: dict
    background: bool = False

    def take_probe(self):
        probe, self.probe = self.probe, None
        if probe is None:
            raise RuntimeError('视觉模型已被使用')
        return probe

    def close(self):
        probe, self.probe = self.probe, None
        if probe is not None:
            probe.close()


def prepare_camera_resources(models_dir: Path, profile_path: Path | None,
                             progress=None, *, background=False):
    """Verify and load vision models without opening the camera."""
    progress = progress or (lambda _message: None)
    started = time.monotonic()
    from observer.models import sha256_file
    progress('启动中：正在核对冻结模型…')
    models, model_path, manifest_path = verify_frozen_models(models_dir)
    frozen_models_at = time.monotonic()
    config = load_frozen_config()
    config_at = time.monotonic()
    from observer.calibration import load_profile
    profile = load_profile(profile_path) if profile_path else None
    verified_at = time.monotonic()
    from observer.object_detection import YoloOnnxBackend
    from observer.vision import MediapipeVisionProbe
    runtime_imports_at = time.monotonic()
    progress('启动中：正在加载视觉模型…')
    backend = YoloOnnxBackend(
        model_path, score_threshold=FROZEN_PHONE_SCORE_THRESHOLD,
        single_class_name='cell_phone')
    yolo_ready_at = time.monotonic()
    try:
        if FROZEN_PHONE_VERIFIER_SHA is not None:
            from observer.phone_verifier import PhoneProposalVerifier, VerifiedPhoneBackend
            backend = VerifiedPhoneBackend(backend, PhoneProposalVerifier(
                models_dir / 'yolo' / 'phone-proposal-verifier.xml', FROZEN_PHONE_VERIFIER_SHA))
        probe = MediapipeVisionProbe(
            models, config.inference, profile,
            phone_score_threshold=FROZEN_PHONE_SCORE_THRESHOLD,
            phone_retain_seconds=FROZEN_PHONE_RETAIN_SECONDS,
            phone_center_distance=FROZEN_PHONE_CENTER_DISTANCE,
            object_backend=backend)
    except BaseException:
        backend.close()
        raise
    ready_at = time.monotonic()
    return PreparedCameraResources(
        models=models,
        model_path=model_path,
        manifest_path=manifest_path,
        config=config,
        profile=profile,
        probe=probe,
        profile_sha256=sha256_file(profile_path) if profile_path else None,
        manifest_sha256=sha256_file(manifest_path),
        timings={
            'verify': verified_at-started,
            'models': ready_at-verified_at,
            'total': ready_at-started,
            'details': {
                'frozen_models': frozen_models_at-started,
                'config': config_at-frozen_models_at,
                'profile': verified_at-config_at,
                'runtime_imports': runtime_imports_at-verified_at,
                'yolo_session': yolo_ready_at-runtime_imports_at,
                'mediapipe_probe': ready_at-yolo_ready_at,
            },
        },
        background=background,
    )


class BackgroundCameraPreloader:
    """Load one camera-free vision stack while the product window is idle."""

    def __init__(self, prepare):
        self._prepare = prepare
        self._lock = threading.Lock()
        self._event = None
        self._thread = None
        self._prepared = None
        self._error = None
        self._shutdown = False

    @property
    def ready(self):
        with self._lock:
            return self._prepared is not None

    def start(self, completed=None):
        with self._lock:
            if self._shutdown or self._prepared is not None or self._thread is not None:
                return False
            event = self._event = threading.Event()

            def load():
                prepared = error = None
                try:
                    prepared = self._prepare()
                except BaseException as exc:
                    error = exc
                with self._lock:
                    if self._shutdown and prepared is not None:
                        prepared.close()
                        prepared = None
                    self._prepared = prepared
                    self._error = error
                    event.set()
                if completed is not None and not self._shutdown:
                    completed(error)

            self._thread = threading.Thread(
                target=load, name='observer-model-prewarm', daemon=True)
            self._thread.start()
            return True

    def acquire(self, cancelled=None):
        self.start()
        with self._lock:
            event = self._event
        while event is not None and not event.wait(.05):
            if cancelled and cancelled():
                raise RuntimeError('视觉模型准备已取消')
        with self._lock:
            prepared, error = self._prepared, self._error
            self._prepared = self._error = None
            self._thread = self._event = None
        if error is not None:
            raise error
        if prepared is None:
            raise RuntimeError('视觉模型没有准备完成')
        return prepared

    def close(self, *, wait=False):
        with self._lock:
            self._shutdown = True
            prepared, self._prepared = self._prepared, None
            thread = self._thread
        try:
            if prepared is not None:
                prepared.close()
        finally:
            if wait and thread is not None and thread is not threading.current_thread():
                thread.join()


class DemoSource:
    identity = {'source': 'synthetic-demo', 'warning': '模拟数据，不代表真人检测结果'}
    frame = None

    def __init__(self):
        self.signal = 'normal'

    def read(self, at):
        if self.signal == 'unknown':
            return Observation(at, {name: SignalValue.unknown('模拟不可观测') for name in NAMES})
        return Observation(at, {name: SignalValue.violation() if self.signal == name
                                else SignalValue.compliant() for name in NAMES})

    def close(self):
        pass


class CameraSource:
    def __init__(self, models_dir: Path, profile_path: Path | None, progress=None,
                 setup_preview: bool = True, cancelled=None,
                 prepared: PreparedCameraResources | None = None,
                 startup_started: float | None = None):
        progress = progress or (lambda _message: None)
        startup_started = startup_started or time.monotonic()
        if prepared is None:
            prepared = prepare_camera_resources(models_dir, profile_path, progress)
        elif prepared.background:
            progress('启动中：视觉模型已预热，正在打开摄像头…')
        models_ready_at = time.monotonic()
        config, profile = prepared.config, prepared.profile
        model_path = prepared.model_path
        from observer.camera import CameraWorker
        self.camera = CameraWorker(config.camera)
        self.camera_config = config.camera
        self.probe = prepared.take_probe()
        self.frame = None
        self.stale_since = None
        self.automatic_reconnects = 0
        self.cancelled = cancelled
        attention_baseline = None
        try:
            progress('启动中：正在打开摄像头…')
            started = (self.camera.start(cancelled=cancelled) if cancelled else
                       self.camera.start())
            if not started:
                raise RuntimeError(self.camera.stats.last_error or '摄像头不可用')
            camera_ready_at = time.monotonic()
            self.sequence = 0
            from observer.alignment import AlignmentCancelled
            try:
                first = wait_for_camera_frame(self.camera, cancelled=cancelled)
            except SourceStalled as exc:
                if cancelled and cancelled():
                    raise AlignmentCancelled('session setup cancelled') from exc
                raise RuntimeError(str(exc)) from exc
            first_frame_at = time.monotonic()
            progress('启动中：正在完成首帧视觉初始化…')
            # Model construction does not allocate every native inference cache.
            # Warm all three frozen modules before the session and its resource
            # tracker begin, so calibrated and uncalibrated starts have the same
            # readiness and steady-memory semantics.
            self.probe.process_due(first.frame, first.captured_monotonic)
            first_inference_at = time.monotonic()
            inference_ready_at = first_inference_at
            if not profile:
                progress('启动中：正在稳定视觉运行环境（约3秒）…')
                warmup_deadline = time.monotonic() + INITIAL_VISION_WARMUP_SECONDS
                warmup_sequence = first.sequence
                last_fresh_at = first.captured_monotonic
                while time.monotonic() < warmup_deadline:
                    if cancelled and cancelled():
                        raise AlignmentCancelled('session setup cancelled')
                    warmup_frame = self.camera.buffer.latest(
                        after_sequence=warmup_sequence)
                    if warmup_frame is None:
                        if time.monotonic() - last_fresh_at > FRESH_FRAME_SECONDS:
                            raise RuntimeError(
                                self.camera.stats.last_error or '摄像头准备期间画面中断')
                        time.sleep(.01)
                        continue
                    warmup_sequence = warmup_frame.sequence
                    last_fresh_at = warmup_frame.captured_monotonic
                    self.probe.process_due(
                        warmup_frame.frame, warmup_frame.captured_monotonic)
                if time.monotonic() - last_fresh_at > FRESH_FRAME_SECONDS:
                    raise RuntimeError(
                        self.camera.stats.last_error or '摄像头准备期间画面中断')
                self.sequence = warmup_sequence
                inference_ready_at = time.monotonic()
            environment_at = inference_ready_at
            if profile:
                from observer.calibration_workflow import current_environment
                from observer.alignment import guide_attention_session_baseline
                profile.assert_environment(current_environment(config, first.frame.shape))
                environment_at = time.monotonic()
                progress('启动中：请看屏幕中央，正在建立本次头姿基线（约3秒）…')
                attention_baseline = guide_attention_session_baseline(
                    self.camera,
                    self.probe,
                    show_preview=setup_preview,
                    cancelled=cancelled,
                    progress=lambda held, required, reason: progress(
                        '启动中：请看屏幕中央，头姿基线 '
                        + (f'{min(held, required):.1f}/{required:.1f}秒'
                           if reason is None else '等待清晰正脸')
                    ),
                )
            ready_at = time.monotonic()
            self.identity = {'source': 'camera', 'phone_model': str(model_path.resolve()),
                             'phone_sha256': FROZEN_SHA, 'person': 'EfficientDet low-frequency',
                             'phone_verifier_sha256': FROZEN_PHONE_VERIFIER_SHA,
                             'phone_tracking': {
                                 'score_threshold': FROZEN_PHONE_SCORE_THRESHOLD,
                                 'retain_seconds': FROZEN_PHONE_RETAIN_SECONDS,
                                 'center_distance': FROZEN_PHONE_CENTER_DISTANCE,
                                 'confirmation_hits': FROZEN_PHONE_CONFIRMATION_HITS,
                             },
                             'profile_id': profile.profile_id if profile else None,
                             'profile_sha256': prepared.profile_sha256,
                             'attention_session_baseline': (
                                 asdict(attention_baseline) if attention_baseline else None
                             ),
                             'soft_signals': 'calibrated' if profile else 'unavailable',
                             'config': asdict(config),
                             'config_sha256': FROZEN_CONFIG_SHA,
                             'model_manifest_sha256': prepared.manifest_sha256,
                             'startup_seconds': {
                                 'verify': round(
                                     0 if prepared.background else prepared.timings['verify'], 3),
                                 'models': round(
                                     models_ready_at-startup_started if prepared.background
                                     else prepared.timings['models'], 3),
                                 'camera': round(camera_ready_at-models_ready_at, 3),
                                  'baseline': round(ready_at-camera_ready_at, 3),
                                  'total': round(ready_at-startup_started, 3),
                                  'details': {
                                      **{
                                          name: round(value, 3)
                                          for name, value in prepared.timings['details'].items()
                                      },
                                      'prewarmed': prepared.background,
                                      'prewarm_total': round(prepared.timings['total'], 3),
                                      'prewarm_wait': round(
                                          models_ready_at-startup_started, 3)
                                          if prepared.background else 0,
                                      'camera_open': round(camera_ready_at-models_ready_at, 3),
                                      'first_frame': round(first_frame_at-camera_ready_at, 3),
                                      'first_inference': round(
                                          first_inference_at-first_frame_at, 3),
                                      'inference_warmup': round(
                                          inference_ready_at-first_inference_at, 3),
                                      'environment': round(
                                          environment_at-inference_ready_at, 3),
                                      'baseline_stability': round(ready_at-environment_at, 3),
                                  },
                              }}
        except BaseException:
            self.camera.stop()
            if self.probe:
                self.probe.close()
            raise

    def read(self, at):
        latest = self.camera.buffer.latest()
        if latest is None or at-latest.captured_monotonic > FRESH_FRAME_SECONDS:
            self.frame = None
            if self.stale_since is None:
                self.stale_since = at
            elif at-self.stale_since >= STALE_FRAME_SECONDS:
                detail = self.camera.stats.last_error or '连续没有新鲜摄像头画面'
                raise SourceStalled(detail)
            return Observation(at, {name: SignalValue.unknown('摄像头帧不可用或已过期') for name in NAMES})
        self.stale_since = None
        self.frame = latest.frame
        if latest.sequence != self.sequence:
            self.sequence = latest.sequence
            self.probe.process_due(latest.frame, at)
        return self.probe.observation(at)

    def reconnect(self):
        if self.automatic_reconnects >= 1:
            raise SourceStalled('本次会话的自动重连次数已用完')
        from observer.camera import CameraWorker
        self.automatic_reconnects += 1
        self.camera.stop()
        self.camera = CameraWorker(self.camera_config)
        self.frame = None
        self.stale_since = None
        self.sequence = 0
        cancelled = getattr(self, 'cancelled', None)
        started = (self.camera.start(cancelled=cancelled) if cancelled else
                   self.camera.start())
        if not started:
            raise SourceStalled(self.camera.stats.last_error or '摄像头自动重连失败')
        try:
            wait_for_camera_frame(self.camera, cancelled=cancelled)
        except SourceStalled:
            self.camera.stop()
            raise

    def close(self):
        self.camera.stop()
        if self.probe:
            self.probe.close()


def show_summary(result):
    score = (f"{result['score']:.1f}" if result['score'] is not None
             and result.get('score_version') == 'time_ratio_v2'
             else f"{result['score']:g}" if result['score'] is not None
             else '证据不足')
    mode = {'demo': '模拟', 'real': '真实', 'unknown': '身份异常'}[result['mode']]
    suffix = (f"违规 {result.get('violation_seconds', 0):.1f}s"
              if result.get('score_version') == 'time_ratio_v2'
              else f"旧规则已扣 {result.get('deducted', 0):g}")
    print(f"[{mode}] {result['title']} | {result['state']} | 分数 {score} | "
          f"有效覆盖 {result['coverage']:.0%} | {suffix}")
    if result.get('data_warnings'):
        print('  数据警告：' + '；'.join(result['data_warnings']))
    if result.get('error'):
        print('  中断原因：'+result['error'])
    incidents = {item['id']: item for item in result.get('focus_incidents', [])}
    for action in result['actions']:
        if action['kind'] in ('alert', 'penalty'):
            incident = incidents.get(action['event_id'])
            name = (' + '.join(NAMES.get(signal, signal)
                               for signal in incident['signals'])
                    if incident else NAMES.get(action['rule_id'], action['rule_id']))
            print(f"  {action['offset']:.1f}s {name} "
                  f"{'扣 '+str(action['points'])+' 分' if action['kind']=='penalty' else '提醒'}")
    alerts = [action for action in result['actions'] if action['kind'] == 'alert']
    if alerts:
        delivered = sum(action.get('delivery_state') == 'delivered' for action in alerts)
        unresolved = len(alerts)-delivered
        print(f'  提醒交付：{delivered}次已确认，{unresolved}次未确认或历史状态未知')
    for signal, totals in result['totals'].items():
        print(f"  {NAMES.get(signal,signal)}：观测 {totals['eligible']:.1f}s，"
              f"有效 {totals['valid']:.1f}s，异常 {totals['violation']:.1f}s")
    resources = result.get('resource_summary')
    if resources:
        if resources['sample_count']:
            print(f"  资源：CPU平均 {resources['process_cpu_average_percent']:.1f}%，"
                  f"P95 {resources['process_cpu_p95_percent']:.1f}%，"
                  f"内存峰值 {resources['process_memory_peak_mb']:.0f} MB，"
                  f"样本 {resources['sample_count']}")
        elif resources.get('sampling_error'):
            print('  资源采样不可用：'+resources['sampling_error'])


def register_parser(subparsers):
    parser = subparsers.add_parser('product', help='极简单次任务、会话和历史记录')
    parser.add_argument('--data-dir', type=Path, default=default_data_dir())
    commands = parser.add_subparsers(dest='product_command', required=True)
    run = commands.add_parser('run', help='打开控制台运行窗口')
    run.add_argument('--demo', action='store_true', help='仅使用模拟信号，不打开摄像头')
    run.add_argument('--models-dir', type=Path, default=default_models_dir())
    run.add_argument('--profile', type=Path, help='指定个人校准；默认采用当前部署的用户校准路径')
    run.add_argument('--no-profile', action='store_true', help='忽略默认校准，仅启用离席和可见手机')
    gui = commands.add_parser('gui', help='打开极简桌面窗口')
    gui.add_argument('--demo', action='store_true', help='模拟模式，不打开摄像头')
    gui.add_argument('--models-dir', type=Path, default=default_models_dir())
    gui.add_argument('--profile', type=Path)
    gui.add_argument('--no-profile', action='store_true', help='忽略默认校准，仅启用离席和可见手机')
    check = commands.add_parser('check', help='在临时目录执行产品负载检查')
    check.add_argument('--seconds', type=float, default=15,
                       help='检查时长，0.1至3600秒；3600秒启用D5长期资源门')
    check.add_argument('--demo', action='store_true', help='模拟模式，不打开摄像头')
    check.add_argument('--models-dir', type=Path, default=default_models_dir())
    check.add_argument('--profile', type=Path)
    check.add_argument('--no-profile', action='store_true', help='忽略默认校准，仅启用硬信号')
    commands.add_parser('history', help='最近100次会话')
    commands.add_parser('data', help='查看本地产品记录与事件图片占用')
    show = commands.add_parser('show', help='查看会话摘要')
    show.add_argument('id')
    task = commands.add_parser('task', help='增加一次任务（程序运行时到点执行）')
    task.add_argument('title')
    task.add_argument('--at', required=True, help='带时区的ISO时间，例如2026-09-15T09:00:00-06:00')
    task.add_argument('--minutes', required=True, type=float)
    commands.add_parser('tasks', help='查看单次任务')
    cancel = commands.add_parser('cancel', help='取消待执行任务')
    cancel.add_argument('id')
    retry = commands.add_parser('retry', help='立即重试冲突、失败或错过的任务')
    retry.add_argument('id')
    delete = commands.add_parser('delete', help='删除指定已结束会话及其图像')
    delete.add_argument('id')
    commands.add_parser('purge', help='按默认30天保留期清理已结束会话')
    commands.add_parser('delete-all', help='删除产品会话、证据和任务，不触碰C数据')
    images = commands.add_parser('images', help='事件截图默认关闭')
    images.add_argument('mode', choices=('on', 'off'))


HELP = ('命令：start [分钟] [任务名] | pause | resume | extend [分钟] | end | status | history | quit\n'
        '一次任务可在另一个终端用 product task 添加。暂停会冻结剩余时间。')


def resource_gate_problems(result: dict, requested_seconds: float) -> list[str] | None:
    """Return D5 gate failures, or None when this is only a diagnostic short check."""
    if requested_seconds < LONG_RESOURCE_GATE_SECONDS:
        return None

    problems = []
    if result.get('state') != 'ended':
        problems.append('会话未正常结束')
    if result.get('coverage', 0.0) < .8:
        problems.append('硬信号有效覆盖低于80%')

    resources = result.get('resource_summary')
    if not resources:
        problems.append('缺少资源汇总')
        return problems
    if resources.get('sampling_error'):
        problems.append('资源采样失败：' + str(resources['sampling_error']))
    # The normal five-second cadence yields about 720 samples. Requiring half
    # catches a stalled sampler without rejecting harmless scheduling jitter.
    minimum_samples = max(2, int(requested_seconds / 10))
    if resources.get('sample_count', 0) < minimum_samples:
        problems.append(f'资源样本不足（少于{minimum_samples}）')
    if resources.get('process_memory_growth_mb_per_hour') is None:
        problems.append('缺少每小时内存增长值')
    violations = tuple(resources.get('violations') or ())
    names = {
        'average CPU budget exceeded': 'CPU平均值超过15%',
        'p95 CPU budget exceeded': 'CPU P95超过25%',
        'steady memory budget exceeded': '内存峰值超过500 MB',
        'memory growth budget exceeded': '每小时内存增长超过50 MB',
    }
    problems.extend(names.get(item, str(item)) for item in violations)
    if not resources.get('within_budget', False) and not violations:
        problems.append('资源预算未通过')
    return problems


def run_load_check(args):
    seconds = args.seconds
    if not math.isfinite(seconds) or not .1 <= seconds <= LONG_RESOURCE_GATE_SECONDS:
        raise ValueError('产品负载检查须为0.1至3600秒')
    long_gate = seconds >= LONG_RESOURCE_GATE_SECONDS
    if long_gate and args.demo:
        raise ValueError('60分钟D5资源门必须使用真实摄像头；--demo只用于短检')
    source = None
    with tempfile.TemporaryDirectory(prefix='observer-product-check-') as directory:
        store = ProductStore(Path(directory))
        service = SessionService(store)
        tracker = SessionResourceTracker(load_config().resources)
        try:
            source = DemoSource() if args.demo else CameraSource(args.models_dir, args.profile)
            identity = dict(source.identity)
            identity['check'] = 'd5-resource-gate' if long_gate else 'short-product-load'
            if long_gate:
                print('开始60分钟D5资源门；使用临时数据，不验证提醒交付。', flush=True)
            else:
                print(f'开始 {seconds:g} 秒短检；只验证观察与资源链路，不验证提醒交付。',
                      flush=True)
            started = time.monotonic()
            service.start('短时负载检查', seconds/60, started, utc_now(), identity)
            tracker.begin(started)
            deadline = started + seconds
            while time.monotonic() < deadline:
                at = time.monotonic()
                tracker.sample(at)
                service.process(source.read(at), utc_now())
                time.sleep(.05)
            ended = time.monotonic()
            result = service.end(
                ended, utc_now(), resource_summary=tracker.finish(ended))
            show_summary(result)
            problems = resource_gate_problems(result, seconds)
            if problems:
                print('60分钟D5资源门未通过：' + '；'.join(problems) + '；临时数据已清理。')
                return 1
            if problems == []:
                resources = result['resource_summary']
                print('60分钟D5资源门通过：硬信号覆盖和资源预算均满足；'
                      f"每小时内存增长 {resources['process_memory_growth_mb_per_hour']:.1f} MB；"
                      '临时数据已清理。')
                return 0
            print('短时检查完成；临时数据已清理。短检不验证提醒交付，也不替代60分钟资源门。')
            return 0
        finally:
            try:
                if source:
                    source.close()
            finally:
                try:
                    if service.identifier:
                        ended = time.monotonic()
                        service.end(ended, utc_now(), 'interrupted', '短时检查异常结束',
                                    tracker.finish(ended))
                finally:
                    store.close()


def run_console(args, store, lock):
    source = None
    service = SessionService(store)
    evidence_recorder = FocusEvidenceRecorder()
    resource_tracker = SessionResourceTracker(load_config().resources)
    failure_detail = None
    try:
        store.recover_evidence()
        store.interrupt_abandoned()
        startup_now = utc_now()
        store.materialize_plans(startup_now)
        store.expire_pending_tasks(startup_now)
        store.purge(startup_now)
        inputs = queue.Queue(maxsize=32)

        def read_commands():
            try:
                while True:
                    line = input()
                    inputs.put(line)
                    if line.strip() == 'quit':
                        return
            except EOFError:
                inputs.put('quit')

        threading.Thread(target=read_commands, daemon=True).start()
        print(HELP, flush=True)
        print('模拟模式：signal normal/phone_held/away/unknown/posture_deviation' if args.demo
              else '真实模式：开始或任务到点才打开摄像头。无校准时仅使用硬信号。', flush=True)
        next_purge = time.monotonic() + 3600
        next_plan_refresh = time.monotonic() + 60

        def open_source():
            return DemoSource() if args.demo else CameraSource(args.models_dir, args.profile)

        def begin(title, minutes, task_id=None, break_every_minutes=None,
                  break_minutes=None, allow_offscreen=False):
            nonlocal source
            if service.identifier:
                raise ValueError('已有会话')
            evidence_recorder.reset()
            from observer.product import minutes_valid
            minutes_valid(minutes)
            source = open_source()
            try:
                identity = dict(source.identity)
                identity.update(source_identity())
                identifier = service.start(
                    title, minutes, time.monotonic(), utc_now(), identity, task_id,
                    break_every_minutes, break_minutes, allow_offscreen)
                resource_tracker.begin(time.monotonic())
            except BaseException:
                source.close()
                source = None
                raise
            print(f'已开始：{title} | {identifier}', flush=True)

        while True:
            at, now = time.monotonic(), utc_now()
            if at >= next_plan_refresh:
                store.materialize_plans(now)
                next_plan_refresh = at + 60
            if at >= next_purge:
                store.purge(now)
                next_purge = at + 3600
            if (service.identifier and service.state != 'paused'
                    and at >= service.deadline):
                if source:
                    evidence_recorder.finish(store, service, source.frame, at)
                show_summary(service.end(
                    at, now, resource_summary=resource_tracker.finish(at)))
                if source:
                    source.close()
                    source = None
            elif service.break_due():
                if source:
                    evidence_recorder.finish(store, service, source.frame, at)
                service.start_break(at, now)
                if source:
                    source.close()
                    source = None
                print(f'计划休息 {service.break_duration_seconds/60:g} 分钟；摄像头已关闭',
                      flush=True)
            elif service.state == 'resting' and at >= service.break_deadline:
                try:
                    source = open_source()
                    service.finish_break(time.monotonic(), utc_now())
                except Exception as exc:
                    ended_at = time.monotonic()
                    show_summary(service.end(
                        ended_at, utc_now(), 'interrupted', f'计划休息后恢复失败：{exc}',
                        resource_tracker.finish(ended_at)))
                    if source:
                        source.close()
                        source = None
                    print(f'会话已中断：计划休息后恢复失败：{exc}', flush=True)
                else:
                    print('计划休息结束，观察已继续', flush=True)
            task = service.due(now)
            if task:
                try:
                    begin(task['title'], task['minutes'], task['id'],
                          task['break_every_minutes'], task['break_minutes'],
                          task['allow_offscreen'])
                except (ValueError, OSError, RuntimeError) as exc:
                    store.fail_task(task['id'], str(exc))
                    print(f'任务未能启动，不计分：{exc}', flush=True)
            try:
                command = inputs.get_nowait()
            except queue.Empty:
                command = ''
            try:
                words = shlex.split(command)
                name = words[0] if words else ''
                if name == 'quit':
                    break
                if name == 'start':
                    begin(' '.join(words[2:]) or '专注', float(words[1]) if len(words)>1 else 25)
                elif name == 'pause':
                    if source:
                        evidence_recorder.finish(
                            store, service, source.frame, time.monotonic())
                    service.pause(time.monotonic(), utc_now())
                    source.close()
                    source = None
                    print('已暂停；倒计时已冻结，摄像头已关闭', flush=True)
                elif name == 'resume':
                    if service.state != 'paused':
                        raise ValueError('会话没有暂停')
                    candidate = open_source()
                    try:
                        service.resume(time.monotonic(), utc_now())
                    except BaseException:
                        candidate.close()
                        raise
                    source = candidate
                    print('已继续', flush=True)
                elif name == 'end':
                    ended_at = time.monotonic()
                    if source:
                        evidence_recorder.finish(store, service, source.frame, ended_at)
                    show_summary(service.end(
                        ended_at, utc_now(), resource_summary=resource_tracker.finish(ended_at)))
                    if source:
                        source.close()
                        source = None
                elif name == 'extend':
                    service.extend(float(words[1]) if len(words) > 1 else 25, utc_now())
                    print('当前会话已延长', flush=True)
                elif name == 'status':
                    if service.identifier:
                        show_summary(store.summary(service.identifier))
                    else:
                        print('等待开始/计划任务', flush=True)
                elif name == 'history':
                    print_history(store)
                elif name == 'signal' and args.demo:
                    if not isinstance(source, DemoSource) or len(words)!=2 or words[1] not in (*NAMES, 'normal', 'unknown'):
                        raise ValueError('请先 start，再输入支持的模拟信号')
                    source.signal = words[1]
                    print('模拟信号：'+source.signal, flush=True)
                elif name:
                    print(HELP, flush=True)
            except (ValueError, OSError, RuntimeError) as exc:
                print(f'未执行：{exc}', flush=True)
            if service.state == 'monitoring' and source:
                resource_tracker.sample(at)
                try:
                    observation = source.read(time.monotonic())
                except SourceStalled as exc:
                    evidence_recorder.finish(store, service, source.frame, at)
                    service.source_gap(time.monotonic(), utc_now(), str(exc))
                    try:
                        source.reconnect()
                    except Exception as reconnect_exc:
                        detail = f'{exc}；自动重连失败：{reconnect_exc}'
                        ended_at = time.monotonic()
                        show_summary(service.end(
                            ended_at, utc_now(), 'interrupted', detail,
                            resource_tracker.finish(ended_at)))
                        source.close()
                        source = None
                        print(f'会话已中断：{detail}。运行窗口仍可继续使用。', flush=True)
                    else:
                        print('摄像头已自动恢复，会话继续。', flush=True)
                except Exception as exc:
                    ended_at = time.monotonic()
                    show_summary(service.end(
                        ended_at, utc_now(), 'interrupted', str(exc),
                        resource_tracker.finish(ended_at)))
                    source.close()
                    source = None
                    print(f'会话已中断：{exc}。运行窗口仍可继续使用。', flush=True)
                else:
                    actions = service.process(observation, utc_now())
                    for action in actions:
                        if action.kind is ActionKind.ALERT:
                            reminder_name = (action.reason if
                                             action.rule_id == FOCUS_RULE_ID
                                             and action.reason else
                                             NAMES.get(action.rule_id, action.rule_id))
                            print('提醒：'+reminder_name, flush=True)
                            try:
                                store.mark_notification(service.action_identifier(action), True)
                            except sqlite3.Error as exc:
                                print(f'提醒已显示，但交付状态保存失败：{exc}', flush=True)
                        elif action.kind is ActionKind.PENALTY:
                            print('扣分已记录：'+NAMES.get(action.rule_id, action.rule_id), flush=True)
                        if action.rule_id != FOCUS_RULE_ID:
                            capture_event_evidence(
                                store, service.identifier, source.frame, action
                            )
                    evidence_recorder.update(
                        store, service, source.frame, observation, actions)
            time.sleep(.05)
    except Exception as exc:
        failure_detail = str(exc).strip()[:500] or type(exc).__name__
        raise
    finally:
        try:
            if source:
                if service.identifier:
                    evidence_recorder.finish(
                        store, service, source.frame, time.monotonic())
                source.close()
        finally:
            try:
                if service.identifier:
                    ended_at = time.monotonic()
                    service.end(ended_at, utc_now(), 'interrupted', failure_detail,
                                resource_summary=resource_tracker.finish(ended_at))
            finally:
                lock.close()
    return 0


def print_history(store):
    rows = store.history()
    for row in rows:
        mode = {'demo': '模拟', 'real': '真实', 'unknown': '身份异常'}[row['mode']]
        print(f"{row['id']} | [{mode}] {row['title']} | {row['state']} | {row['started']}")
    if not rows:
        print('暂无会话')


def open_locked_store(root: Path):
    """Claim runtime ownership before opening or migrating the product database."""
    resolved = Path(root).resolve()
    lock = RuntimeLock(resolved)
    try:
        store = ProductStore(resolved)
    except BaseException:
        lock.close()
        raise
    return store, lock


def dispatch(args):
    store = None
    try:
        if args.product_command in ('run', 'gui', 'check'):
            apply_default_profile(args)
        if args.product_command == 'gui':
            try:
                from observer.desktop import launch
            except ModuleNotFoundError as exc:
                if exc.name and exc.name.startswith('PySide6'):
                    raise RuntimeError('请先安装桌面依赖：uv sync --extra desktop --extra yolo') from exc
                raise
            return launch(args)
        if args.product_command == 'check':
            return run_load_check(args)
        command = args.product_command
        if command == 'run':
            store, lock = open_locked_store(args.data_dir)
            return run_console(args, store, lock)
        if command == 'delete-all':
            store, lock = open_locked_store(args.data_dir)
            try:
                store.interrupt_abandoned()
                store.delete_all()
                print('产品会话、证据和任务已删除；设置保留')
            finally:
                lock.close()
            return 0
        store = ProductStore(args.data_dir)
        if command == 'task':
            when = datetime.fromisoformat(args.at)
            utc_text(when)
            if when <= utc_now():
                raise ValueError('请选择未来的开始时间')
            print(store.add_task(args.title, when, args.minutes))
        elif command == 'tasks':
            for task in store.tasks():
                detail = f" | {task['error']}" if task.get('error') else ''
                print(f"{task['id']} | {task['title']} | {task['starts']} | {task['state']}{detail}")
        elif command == 'cancel':
            store.cancel_task(args.id)
        elif command == 'retry':
            store.retry_task(args.id, utc_now())
        elif command == 'history':
            print_history(store)
        elif command == 'data':
            status = store.data_status()
            image_mode = '开启' if status['images_enabled'] else '关闭'
            print(f"会话 {status['sessions']} | 任务 {status['tasks']} | "
                  f"重复计划 {status['plans']} | "
                  f"图片 {status['images']} ({status['image_bytes']} bytes) | "
                  f"截图 {image_mode} | 截图保留 {status['retention_days']} 天，记录长期保留 | "
                  f"证据异常 {status['evidence_errors']}")
        elif command == 'show':
            show_summary(store.summary(args.id))
        elif command == 'images':
            store.set_images(args.mode == 'on')
            print('事件截图：'+args.mode)
        elif command == 'delete':
            store.delete_session(args.id)
        elif command == 'purge':
            print(f'已清理 {store.purge(utc_now())} 张过期截图；任务和文字日志已保留')
        return 0
    except (ValueError, OSError, RuntimeError, sqlite3.Error) as exc:
        print(f'Observer：{exc}')
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        if store:
            store.close()
