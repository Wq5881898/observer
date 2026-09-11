"""One-window Qt frontend. SQLite and inference belong to the worker thread."""
from __future__ import annotations

import hashlib
import json
import os
import queue
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QDateTime, QThread, Signal, Slot, Qt
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDateTimeEdit, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QPushButton, QSpinBox, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

from observer.product import NAMES, ProductStore, RuntimeLock, SessionService, minutes_valid, utc_now
from observer.product_cli import CameraSource, DemoSource, SourceStalled
from observer.rules import ActionKind

STATES = {'idle': '等待开始', 'monitoring': '观察中', 'paused': '已暂停',
          'ended': '已结束', 'interrupted': '已中断', 'pending': '待开始',
          'running': '进行中', 'conflict': '任务冲突', 'missed': '已错过',
          'cancelled': '已取消', 'failed': '启动失败', 'faulted': '运行异常'}


def summary_text(result):
    score = f"{result['score']:g} 分" if result['score'] is not None else '证据不足'
    mode = '模拟' if json.loads(result['identity']).get('source') == 'synthetic-demo' else '真实'
    lines = [f"{result['title']} · {mode}",
             f"{STATES.get(result['state'],result['state'])}  |  {score}  |  有效覆盖 {result['coverage']:.0%}", '']
    for action in result['actions']:
        if action['kind'] in ('alert', 'penalty'):
            detail = f"扣 {action['points']:g} 分" if action['kind']=='penalty' else '提醒'
            lines.append(f"{action['offset']:.1f} 秒  {NAMES.get(action['rule_id'], action['rule_id'])} · {detail}")
    if not result['actions']:
        lines.append('无提醒或扣分事件')
    soft = []
    for name in ('attention_away', 'posture_deviation', 'ergonomic_risk'):
        seconds = result['totals'].get(name, {}).get('violation', 0)
        if seconds >= .1:
            soft.append(f"{NAMES.get(name, name)} {seconds:.1f}秒")
    if soft:
        lines.extend(['', '软信号累计异常：' + ' · '.join(soft), '累计时间不等于连续触发时间'])
    if result.get('error'):
        lines.extend(['', '中断原因：' + result['error']])
    return '\n'.join(lines)


def play_reminder_sound():
    """Play an asynchronous Windows alert without adding a media dependency."""
    if os.name == 'nt':
        try:
            import winsound
            winsound.PlaySound('SystemExclamation', winsound.SND_ALIAS | winsound.SND_ASYNC)
            return
        except (ImportError, RuntimeError):
            pass
    QApplication.beep()


class DesktopWorker(QThread):
    snapshot = Signal(dict)
    notice = Signal(str)
    detail = Signal(str)
    busy = Signal(bool)

    def __init__(self, args, source_factory=None):
        super().__init__()
        self.args = args
        self.factory = source_factory or (lambda: DemoSource() if args.demo else
                                         CameraSource(args.models_dir, args.profile))
        self.commands = queue.Queue(maxsize=32)
        self.source = None
        self.store = None
        self.service = None

    def submit(self, command, *values):
        if not self.isRunning():
            return
        try:
            self.commands.put_nowait((command, values))
        except queue.Full:
            self.notice.emit('正在处理，请稍候')

    def stop_source(self):
        source, self.source = self.source, None
        if source:
            source.close()

    def begin(self, title, minutes, task_id=None):
        if self.service.identifier:
            raise ValueError('请先结束当前会话')
        minutes_valid(minutes)
        if not title.strip():
            raise ValueError('请输入任务名称')
        self.busy.emit(True)
        self.notice.emit('正在准备观察…')
        try:
            self.source = self.factory()
            if self.isInterruptionRequested():
                self.stop_source()
                return
            identity = dict(self.source.identity)
            digest = hashlib.sha256()
            for path in sorted(Path(__file__).parent.glob('*.py')):
                digest.update(path.name.encode())
                digest.update(path.read_bytes())
            identity['source_tree_sha256'] = digest.hexdigest()
            self.service.start(title, minutes, time.monotonic(), utc_now(), identity, task_id)
            self.notice.emit('已开始。可以自然使用电脑。')
        except Exception:
            self.stop_source()
            raise
        finally:
            self.busy.emit(False)

    def finish_session(self, reason='ended', detail=None):
        try:
            result = self.service.end(time.monotonic(), utc_now(), reason, detail)
            self.detail.emit(summary_text(result))
        finally:
            self.stop_source()

    def publish(self):
        service = self.service
        current = self.store.summary(service.identifier) if service.identifier else None
        signals = ({name: service.last.signal(name).state.value for name in NAMES}
                   if service.last is not None else {})
        self.snapshot.emit({'state': service.state, 'current': current,
                            'remaining': max(0, service.deadline-time.monotonic()) if current else 0,
                            'tasks': self.store.tasks(), 'history': self.store.history(),
                            'images': self.store.setting('images'),
                            'signals': signals,
                            'demo_signal': getattr(self.source, 'signal', 'normal')})

    def execute(self, command, values):
        service, store = self.service, self.store
        if command == 'start':
            self.begin(*values)
        elif command == 'pause':
            service.pause(time.monotonic(), utc_now())
            self.stop_source()
            self.notice.emit('已暂停。计划结束时间不变。')
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
                raise
            finally:
                self.busy.emit(False)
        elif command == 'end':
            self.finish_session()
            self.notice.emit('会话已保存')
        elif command == 'task':
            title, when, minutes = values
            if when <= utc_now():
                raise ValueError('请选择未来的开始时间')
            store.add_task(title, when, minutes)
            self.notice.emit('已安排。保持窗口打开，到点自动开始。')
        elif command == 'cancel':
            store.cancel_task(values[0])
        elif command == 'retry':
            store.retry_task(values[0], utc_now())
            self.notice.emit('任务已重新进入待开始状态')
        elif command == 'show':
            self.detail.emit(summary_text(store.summary(values[0])))
        elif command == 'images':
            store.set_images(values[0])
        elif command == 'signal':
            if isinstance(self.source, DemoSource):
                self.source.signal = values[0]

    def run(self):
        lock = None
        try:
            lock = RuntimeLock(Path(self.args.data_dir).resolve())
            self.store = ProductStore(self.args.data_dir)
            self.service = SessionService(self.store)
            self.store.interrupt_abandoned()
            self.store.expire_pending_tasks(utc_now())
            self.store.purge(utc_now())
            next_publish, next_purge = 0., time.monotonic()+3600
            while not self.isInterruptionRequested():
                at, now = time.monotonic(), utc_now()
                if self.service.identifier and at >= self.service.deadline:
                    self.finish_session()
                    self.notice.emit('计划时间已到，会话已保存')
                    next_publish = 0
                task = self.service.due(now)
                if task:
                    try:
                        self.begin(task['title'], task['minutes'], task['id'])
                    except Exception as exc:
                        self.store.fail_task(task['id'], str(exc))
                        self.notice.emit(f'任务启动失败：{exc}')
                    next_publish = 0
                try:
                    command, values = self.commands.get_nowait()
                except queue.Empty:
                    command = None
                if command:
                    try:
                        self.execute(command, values)
                    except (ValueError, OSError, RuntimeError) as exc:
                        self.notice.emit(str(exc))
                    next_publish = 0
                if self.service.state == 'monitoring' and self.source:
                    try:
                        observation = self.source.read(time.monotonic())
                    except SourceStalled as exc:
                        self.service.source_gap(time.monotonic(), utc_now(), str(exc))
                        self.notice.emit('摄像头断开，正在自动恢复（仅尝试一次）…')
                        try:
                            self.source.reconnect()
                        except Exception as reconnect_exc:
                            detail = f'{exc}；自动重连失败：{reconnect_exc}'
                            self.finish_session('interrupted', detail)
                            self.notice.emit(f'会话已中断：{detail}。可以重新开始。')
                        else:
                            self.notice.emit('摄像头已恢复，会话继续。')
                        next_publish = 0
                    except Exception as exc:
                        self.finish_session('interrupted', str(exc))
                        self.notice.emit(f'会话已中断：{exc}。可以重新开始。')
                        next_publish = 0
                    else:
                        actions = self.service.process(observation, utc_now())
                        for action in actions:
                            if action.kind is ActionKind.ALERT:
                                self.notice.emit('提醒：'+NAMES.get(action.rule_id, action.rule_id))
                            elif action.kind is ActionKind.ACTIVATED and action.rule_id in ('away', 'phone_held'):
                                self.store.capture(self.service.identifier, action.event_id, self.source.frame)
                if at >= next_publish:
                    self.publish()
                    next_publish = at + .5
                if at >= next_purge:
                    self.store.purge(now)
                    next_purge = at + 3600
                self.msleep(50)
        except Exception as exc:
            self.notice.emit(f'运行已停止：{exc}')
        finally:
            try:
                if self.service and self.service.identifier:
                    self.finish_session('interrupted')
                else:
                    self.stop_source()
            except Exception as exc:
                self.notice.emit(f'结束时发生错误：{exc}')
            finally:
                if self.store:
                    self.store.close()
                if lock:
                    lock.close()


class ObserverWindow(QMainWindow):
    def __init__(self, args, source_factory=None):
        super().__init__()
        self.setWindowTitle('Observer · 模拟' if args.demo else 'Observer')
        self.resize(560, 720)
        self.setMinimumSize(460, 600)
        self.closing = False
        self.blocked = True
        self.state = 'idle'
        self.worker = DesktopWorker(args, source_factory)
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setSpacing(14)
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
        form = QFormLayout()
        self.title = QLineEdit('专注')
        self.minutes = QSpinBox()
        self.minutes.setRange(1, 1440)
        self.minutes.setValue(25)
        self.minutes.setSuffix(' 分钟')
        form.addRow('任务', self.title)
        form.addRow('时长', self.minutes)
        self.scheduled = QCheckBox('指定开始时间')
        self.when = QDateTimeEdit(QDateTime.currentDateTime().addSecs(60))
        self.when.setCalendarPopup(True)
        self.when.setDisplayFormat('yyyy-MM-dd HH:mm:ss')
        self.when.setEnabled(False)
        self.scheduled.toggled.connect(self.when.setEnabled)
        form.addRow(self.scheduled, self.when)
        layout.addLayout(form)
        self.status = QLabel('正在准备…')
        self.status.setWordWrap(True)
        self.status.setStyleSheet('font-size: 17px; font-weight: 600; padding: 10px 0;')
        layout.addWidget(self.status)
        self.alert_banner = QLabel('')
        self.alert_banner.setWordWrap(True)
        self.alert_banner.setStyleSheet(
            'background:#fee2e2; color:#991b1b; border:1px solid #fca5a5; '
            'border-radius:5px; padding:10px; font-size:16px; font-weight:600;')
        self.alert_banner.hide()
        layout.addWidget(self.alert_banner)
        buttons = QHBoxLayout()
        self.start = QPushButton('开始')
        self.start.setStyleSheet('background: #2563eb; color: white; border: none;')
        self.pause = QPushButton('暂停')
        self.end = QPushButton('结束')
        for button in (self.start, self.pause, self.end):
            button.setMinimumHeight(38)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.start.clicked.connect(self.start_clicked)
        self.pause.clicked.connect(lambda: self.send('resume' if self.state == 'paused' else 'pause'))
        self.end.clicked.connect(lambda: self.send('end'))
        self.scheduled.toggled.connect(lambda on: self.start.setText('安排任务' if on else '开始'))
        self.images = QCheckBox('保存事件截图（最多一张/事件，保留30天）')
        self.images.toggled.connect(lambda enabled: self.worker.submit('images', enabled))
        layout.addWidget(self.images)
        if args.demo:
            self.demo_signal = QComboBox()
            for label, signal in [('正常', 'normal'), ('可见手机', 'phone_held'), ('离席', 'away'), ('无法判断', 'unknown')]:
                self.demo_signal.addItem('模拟：'+label, signal)
            self.demo_signal.currentIndexChanged.connect(lambda: self.worker.submit('signal', self.demo_signal.currentData()))
            layout.addWidget(self.demo_signal)
        if args.demo:
            message = '保持窗口打开，定时任务才能执行。'
        elif args.profile:
            message = '眼神离屏持续8秒后发出提示音；提醒不扣分。'
        else:
            message = '未找到个人校准；当前只检测离席和可见手机。'
        self.message = QLabel(message)
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        tabs = QTabWidget()
        self.history = QListWidget()
        self.tasks = QListWidget()
        tabs.addTab(self.history, '历史')
        task_page = QWidget()
        task_layout = QVBoxLayout(task_page)
        task_layout.addWidget(self.tasks)
        task_buttons = QHBoxLayout()
        self.retry_task = QPushButton('重试选中任务')
        self.retry_task.clicked.connect(self.retry_selected)
        self.cancel_task = QPushButton('取消选中任务')
        self.cancel_task.clicked.connect(self.cancel_selected)
        task_buttons.addWidget(self.retry_task)
        task_buttons.addWidget(self.cancel_task)
        task_layout.addLayout(task_buttons)
        tabs.addTab(task_page, '计划')
        layout.addWidget(tabs, 1)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlaceholderText('结束后显示摘要，也可以点击历史记录查看。')
        self.details.setMaximumHeight(140)
        layout.addWidget(self.details)
        self.history.itemClicked.connect(lambda item: self.worker.submit('show', item.data(Qt.ItemDataRole.UserRole)))
        self.setCentralWidget(root)
        self.setStyleSheet('QMainWindow {background:#f8fafc;} QWidget {font-family:"Microsoft YaHei UI"; font-size:13px;} '
                           'QLineEdit,QSpinBox,QDateTimeEdit,QTextEdit,QListWidget {background:white; border:1px solid #cbd5e1; border-radius:5px; padding:5px;} '
                           'QPushButton {padding:7px 12px; border:1px solid #cbd5e1; border-radius:5px;} '
                           'QPushButton:disabled {color:#94a3b8; background:#e2e8f0;}')
        self.worker.snapshot.connect(self.update_view)
        self.worker.notice.connect(self.show_notice)
        self.worker.detail.connect(self.details.setPlainText)
        self.worker.busy.connect(self.set_busy)
        self.worker.finished.connect(self.worker_finished)
        self.update_buttons()
        self.worker.start()

    def send(self, command, *values):
        self.set_busy(True)
        self.worker.submit(command, *values)

    @Slot(str)
    def show_notice(self, message):
        self.message.setText(message)
        if message.startswith('提醒：'):
            self.message.setStyleSheet('color:#b91c1c; font-weight:600;')
            self.alert_banner.setText(message + '（已记录）')
            self.alert_banner.show()
            play_reminder_sound()
            QApplication.alert(self, 3000)
        else:
            self.message.setStyleSheet('')

    @Slot()
    def start_clicked(self):
        self.alert_banner.hide()
        self.alert_banner.clear()
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

    @Slot(bool)
    def set_busy(self, busy):
        self.blocked = busy
        self.update_buttons()

    def update_buttons(self):
        available = not self.blocked and not self.closing
        self.start.setEnabled(available and self.state == 'idle')
        self.pause.setEnabled(available and self.state in ('monitoring', 'paused'))
        self.end.setEnabled(available and self.state in ('monitoring', 'paused'))
        self.pause.setText('继续' if self.state == 'paused' else '暂停')

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
        self.blocked = False
        self.update_buttons()
        current = snapshot['current']
        if current:
            seconds = int(snapshot['remaining'])
            score = f"{current['score']:g} 分" if current['score'] is not None else '证据不足'
            text = f"{STATES[self.state]} · {current['title']}\n剩余 {seconds//60:02}:{seconds%60:02} · {score}"
            if self.worker.args.demo or self.worker.args.profile:
                attention = snapshot.get('signals', {}).get('attention_away', 'unknown')
                label = {'compliant': '屏内', 'violation': '屏外候选',
                         'unknown': '不可判断', 'exempt': '已暂停'}.get(attention, attention)
                text += f"\n头眼：{label}"
            self.status.setText(text)
        else:
            self.status.setText('等待开始')
        self.images.blockSignals(True)
        self.images.setChecked(snapshot['images'])
        self.images.blockSignals(False)
        if hasattr(self, 'demo_signal'):
            self.demo_signal.setEnabled(self.state == 'monitoring')
            self.demo_signal.blockSignals(True)
            self.demo_signal.setCurrentIndex(self.demo_signal.findData(snapshot['demo_signal']))
            self.demo_signal.blockSignals(False)
        def local(stamp):
            return datetime.fromisoformat(stamp).astimezone().strftime('%m-%d %H:%M')
        self.fill_list(self.history, [(r['id'], f"{local(r['started'])}  {r['title']} · {STATES.get(r['state'],r['state'])}" +
                                      (' · 模拟' if json.loads(r['identity']).get('source') == 'synthetic-demo' else ''))
                                     for r in snapshot['history']])
        self.fill_list(self.tasks, [(r['id'], f"{local(r['starts'])}  {r['title']} · {STATES.get(r['state'],r['state'])}" +
                                    (f"\n{r['error'][:120]}" if r.get('error') else ''))
                                   for r in snapshot['tasks'] if r['state'] in ('pending', 'conflict', 'failed', 'missed')])

    @Slot()
    def worker_finished(self):
        self.blocked = True
        self.update_buttons()
        if self.closing:
            self.close()
        else:
            self.status.setText('运行已停止，请关闭后重新打开')

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.closing = True
            self.update_buttons()
            self.message.setText('正在结束会话并释放摄像头…')
            self.worker.requestInterruption()
            event.ignore()
        else:
            event.accept()


def launch(args):
    app = QApplication.instance() or QApplication([])
    app.setStyle('Fusion')
    window = ObserverWindow(args)
    window.show()
    return app.exec()
