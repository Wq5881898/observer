"""Minimal interactive console; the same SessionService can serve a future UI."""
from __future__ import annotations

import argparse
import hashlib
import json
import queue
import shlex
import sqlite3
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from observer.observations import Observation, SignalValue
from observer.product import NAMES, ProductStore, RuntimeLock, SessionService, utc_now, utc_text
from observer.rules import ActionKind

FROZEN_MODEL = Path('models/yolo/observer-phone-s304-e2b.onnx')
FROZEN_SHA = 'c4a8a791e227780b334e075d671d4347ed9d5fee7474d6f017659366709c486f'
DEFAULT_PROFILE = Path('data/calibration/default.json')
STALE_FRAME_SECONDS = 5.0


class SourceStalled(RuntimeError):
    """A running source stopped producing fresh frames and may be reconnected."""


def apply_default_profile(args, candidate: Path = DEFAULT_PROFILE):
    """Use this machine's saved calibration unless the operator opts out."""
    if args.demo or args.profile or getattr(args, 'no_profile', False):
        return args.profile
    if candidate.is_file():
        args.profile = candidate
    return args.profile


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
    def __init__(self, models_dir: Path, profile_path: Path | None):
        from observer.models import verify_model_directory, sha256_file
        from observer.config import load_config
        model_path = models_dir / 'yolo' / FROZEN_MODEL.name
        if sha256_file(model_path).lower() != FROZEN_SHA:
            raise ValueError('手机模型与 C 冻结哈希不一致；未打开摄像头')
        models = verify_model_directory(models_dir)
        config = load_config()
        from observer.calibration import load_profile
        profile = load_profile(profile_path) if profile_path else None
        from observer.object_detection import YoloOnnxBackend
        from observer.camera import CameraWorker
        from observer.vision import MediapipeVisionProbe
        self.camera = CameraWorker(config.camera)
        self.camera_config = config.camera
        self.probe = None
        self.frame = None
        self.stale_since = None
        self.automatic_reconnects = 0
        attention_baseline = None
        backend = YoloOnnxBackend(model_path, score_threshold=.25, single_class_name='cell_phone')
        try:
            self.probe = MediapipeVisionProbe(models, config.inference, profile,
                                             object_backend=backend)
            if not self.camera.start():
                raise RuntimeError(self.camera.stats.last_error or '摄像头不可用')
            self.sequence = 0
            if profile:
                from observer.calibration_workflow import current_environment
                deadline = time.monotonic() + 5
                first = self.camera.buffer.latest()
                while first is None and time.monotonic() < deadline:
                    time.sleep(.05)
                    first = self.camera.buffer.latest()
                if first is None:
                    raise RuntimeError('未取得摄像头帧')
                profile.assert_environment(current_environment(config, first.frame.shape))
                from observer.alignment import guide_attention_session_baseline
                attention_baseline = guide_attention_session_baseline(self.camera, self.probe)
            self.identity = {'source': 'camera', 'phone_model': str(model_path.resolve()),
                             'phone_sha256': FROZEN_SHA, 'person': 'EfficientDet low-frequency',
                             'profile_id': profile.profile_id if profile else None,
                             'profile_sha256': sha256_file(profile_path) if profile_path else None,
                             'attention_session_baseline': (
                                 asdict(attention_baseline) if attention_baseline else None
                             ),
                             'soft_signals': 'calibrated' if profile else 'unavailable',
                             'config': asdict(config),
                             'model_manifest_sha256': sha256_file(models_dir / 'manifest.json')}
        except BaseException:
            self.camera.stop()
            if self.probe:
                self.probe.close()
            else:
                backend.close()
            raise

    def read(self, at):
        latest = self.camera.buffer.latest()
        if latest is None or at-latest.captured_monotonic > .75:
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
        if not self.camera.start():
            raise SourceStalled(self.camera.stats.last_error or '摄像头自动重连失败')

    def close(self):
        self.camera.stop()
        if self.probe:
            self.probe.close()


def show_summary(result):
    score = f"{result['score']:g}" if result['score'] is not None else '证据不足'
    mode = '模拟' if json.loads(result['identity']).get('source') == 'synthetic-demo' else '真实'
    print(f"[{mode}] {result['title']} | {result['state']} | 分数 {score} | "
          f"有效覆盖 {result['coverage']:.0%} | 已扣 {result['deducted']:g}")
    if result.get('error'):
        print('  中断原因：'+result['error'])
    for action in result['actions']:
        if action['kind'] in ('alert', 'penalty'):
            print(f"  {action['offset']:.1f}s {NAMES.get(action['rule_id'], action['rule_id'])} "
                  f"{'扣 '+str(action['points'])+' 分' if action['kind']=='penalty' else '提醒'}")
    for signal, totals in result['totals'].items():
        print(f"  {NAMES.get(signal,signal)}：观测 {totals['eligible']:.1f}s，"
              f"有效 {totals['valid']:.1f}s，异常 {totals['violation']:.1f}s")


def register_parser(subparsers):
    parser = subparsers.add_parser('product', help='极简单次任务、会话和历史记录')
    parser.add_argument('--data-dir', type=Path, default=Path('data/product'))
    commands = parser.add_subparsers(dest='product_command', required=True)
    run = commands.add_parser('run', help='打开控制台运行窗口')
    run.add_argument('--demo', action='store_true', help='仅使用模拟信号，不打开摄像头')
    run.add_argument('--models-dir', type=Path, default=Path('models'))
    run.add_argument('--profile', type=Path, help='指定个人校准；默认自动采用 data/calibration/default.json')
    run.add_argument('--no-profile', action='store_true', help='忽略默认校准，仅启用离席和可见手机')
    gui = commands.add_parser('gui', help='打开极简桌面窗口')
    gui.add_argument('--demo', action='store_true', help='模拟模式，不打开摄像头')
    gui.add_argument('--models-dir', type=Path, default=Path('models'))
    gui.add_argument('--profile', type=Path)
    gui.add_argument('--no-profile', action='store_true', help='忽略默认校准，仅启用离席和可见手机')
    commands.add_parser('history', help='最近100次会话')
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


HELP = ('命令：start [分钟] [任务名] | pause | resume | end | status | history | quit\n'
        '一次任务可在另一个终端用 product task 添加。暂停不会延后计划结束时间。')


def run_console(args, store):
    lock = RuntimeLock(store.root)
    source = None
    service = SessionService(store)
    try:
        store.interrupt_abandoned()
        store.expire_pending_tasks(utc_now())
        store.purge(utc_now())
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

        def open_source():
            return DemoSource() if args.demo else CameraSource(args.models_dir, args.profile)

        def begin(title, minutes, task_id=None):
            nonlocal source
            if service.identifier:
                raise ValueError('已有会话')
            from observer.product import minutes_valid
            minutes_valid(minutes)
            source = open_source()
            try:
                identity = dict(source.identity)
                files = sorted(Path(__file__).parent.glob('*.py'))
                digest = hashlib.sha256()
                for path in files:
                    digest.update(path.name.encode())
                    digest.update(path.read_bytes())
                identity['source_tree_sha256'] = digest.hexdigest()
                identifier = service.start(title, minutes, time.monotonic(), utc_now(), identity, task_id)
            except BaseException:
                source.close()
                source = None
                raise
            print(f'已开始：{title} | {identifier}', flush=True)

        while True:
            at, now = time.monotonic(), utc_now()
            if at >= next_purge:
                store.purge(now)
                next_purge = at + 3600
            if service.identifier and at >= service.deadline:
                show_summary(service.end(at, now))
                if source:
                    source.close()
                    source = None
            task = service.due(now)
            if task:
                try:
                    begin(task['title'], task['minutes'], task['id'])
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
                    service.pause(time.monotonic(), utc_now())
                    source.close()
                    source = None
                    print('已暂停；摄像头已关闭', flush=True)
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
                    show_summary(service.end(time.monotonic(), utc_now()))
                    if source:
                        source.close()
                        source = None
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
                try:
                    observation = source.read(time.monotonic())
                except SourceStalled as exc:
                    service.source_gap(time.monotonic(), utc_now(), str(exc))
                    try:
                        source.reconnect()
                    except Exception as reconnect_exc:
                        detail = f'{exc}；自动重连失败：{reconnect_exc}'
                        show_summary(service.end(time.monotonic(), utc_now(), 'interrupted', detail))
                        source.close()
                        source = None
                        print(f'会话已中断：{detail}。运行窗口仍可继续使用。', flush=True)
                    else:
                        print('摄像头已自动恢复，会话继续。', flush=True)
                except Exception as exc:
                    show_summary(service.end(time.monotonic(), utc_now(), 'interrupted', str(exc)))
                    source.close()
                    source = None
                    print(f'会话已中断：{exc}。运行窗口仍可继续使用。', flush=True)
                else:
                    actions = service.process(observation, utc_now())
                    for action in actions:
                        if action.kind is ActionKind.ALERT:
                            print('提醒：'+NAMES.get(action.rule_id, action.rule_id), flush=True)
                        elif action.kind is ActionKind.PENALTY:
                            print('扣分已记录：'+NAMES.get(action.rule_id, action.rule_id), flush=True)
                        elif action.kind is ActionKind.ACTIVATED and action.rule_id in ('away', 'phone_held'):
                            store.capture(service.identifier, action.event_id, source.frame)
            time.sleep(.05)
    finally:
        try:
            if source:
                source.close()
        finally:
            try:
                if service.identifier:
                    service.end(time.monotonic(), utc_now(), 'interrupted')
            finally:
                lock.close()
    return 0


def print_history(store):
    rows = store.history()
    for row in rows:
        mode = '模拟' if json.loads(row['identity']).get('source') == 'synthetic-demo' else '真实'
        print(f"{row['id']} | [{mode}] {row['title']} | {row['state']} | {row['started']}")
    if not rows:
        print('暂无会话')


def dispatch(args):
    store = None
    try:
        if args.product_command in ('run', 'gui'):
            apply_default_profile(args)
        if args.product_command == 'gui':
            try:
                from observer.desktop import launch
            except ModuleNotFoundError as exc:
                if exc.name and exc.name.startswith('PySide6'):
                    raise RuntimeError('请先安装桌面依赖：uv sync --extra desktop --extra yolo') from exc
                raise
            return launch(args)
        store = ProductStore(args.data_dir)
        command = args.product_command
        if command == 'run':
            return run_console(args, store)
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
        elif command == 'show':
            show_summary(store.summary(args.id))
        elif command == 'images':
            store.set_images(args.mode == 'on')
            print('事件截图：'+args.mode)
        elif command == 'delete':
            store.delete_session(args.id)
        elif command == 'purge':
            print(f'已清理 {store.purge(utc_now())} 个过期会话')
        elif command == 'delete-all':
            lock = RuntimeLock(store.root)
            try:
                store.interrupt_abandoned()
                ids = [r[0] for r in store.db.execute('SELECT id FROM sessions')]
                for identifier in ids:
                    store.delete_session(identifier)
                with store.db:
                    store.db.execute('DELETE FROM tasks')
                print('产品会话、证据和任务已删除；设置保留')
            finally:
                lock.close()
        return 0
    except (ValueError, OSError, RuntimeError, sqlite3.Error) as exc:
        print(f'Observer：{exc}')
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        if store:
            store.close()
