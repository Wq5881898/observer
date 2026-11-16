"""Explicit internal-candidate diagnostics. Never open a camera or read user data."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


def check_resources(load_native):
    from observer.deployment import default_models_dir, resource_root, source_identity
    from observer.product_cli import verify_frozen_models, load_frozen_config, prepare_camera_resources
    verify_frozen_models(default_models_dir())
    load_frozen_config()
    result = {'identity': source_identity(), 'resource_root': str(resource_root()),
              'model_hashes': {str(p.relative_to(default_models_dir())):
                               hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in default_models_dir().rglob('*') if p.is_file()}}
    if load_native:
        import cv2
        import numpy as np
        import onnxruntime
        import mediapipe
        prepared = prepare_camera_resources(default_models_dir(), None)
        try:
            frame = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
            prepared.probe.process_due(frame, time.monotonic())
            result['native'] = {'cv2': cv2.__version__, 'numpy': np.__version__,
                                'onnxruntime': onnxruntime.__version__, 'mediapipe': mediapipe.__version__,
                                'synthetic_frame_inference': True, 'camera_opened': False}
            from observer.product_cli import FROZEN_PHONE_VERIFIER_SHA
            if FROZEN_PHONE_VERIFIER_SHA is not None:
                from observer.phone_verifier import VerifiedPhoneBackend
                from observer.observations import NormalizedBox
                backend = prepared.probe._object
                if not isinstance(backend, VerifiedPhoneBackend):
                    raise RuntimeError('Pinned phone verifier was not connected')
                decision = backend.verifier.reject(frame, NormalizedBox(.25, .25, .1, .2))
                result['native']['phone_verifier'] = {
                    'sha256': backend.verifier.sha256, 'connected': True,
                    'synthetic_proposal_evaluated': isinstance(decision, bool)}
        finally:
            prepared.close()
    return result


def desktop_smoke(root):
    # Diagnostics may run beside a real user session. Exercise the same signal
    # routes without sounding like the user's task ended or raising a toast.
    from unittest.mock import patch
    with patch('observer.desktop.play_reminder_sound') as alarm, \
            patch('observer.desktop.play_completion_sound') as complete, \
            patch('observer.desktop.DesktopTray.notify', return_value='合成通知已拦截'):
        result = _desktop_smoke(root)
        result['diagnostic_audio_suppressed'] = True
        result['diagnostic_notifications_suppressed'] = True
        result['completion_sound_requests'] = complete.call_count
        result['alarm_sound_requests'] = alarm.call_count
        return result


def _desktop_smoke(root):
    from PySide6.QtWidgets import QApplication
    from observer.desktop import ObserverWindow, _reminder_wave
    from observer.desktop_ipc import SingleInstanceWake
    from observer.deployment import default_models_dir, default_data_dir
    from observer.product import ProductStore, RuntimeLock, utc_now
    import numpy as np

    # Self-test data has an explicit marker; never adopt an existing personal DB.
    marker = root/'INTERNAL-SYNTHETIC-TEST.txt'
    if root.exists() and any(p.name != 'logs' for p in root.iterdir()) and not marker.is_file():
        raise RuntimeError('Self-test requires a new directory or its own marked synthetic data')
    root.mkdir(parents=True, exist_ok=True)
    marker.write_text('Only synthetic internal packaging test data.\n', encoding='utf-8')
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    args = SimpleNamespace(data_dir=root, demo=True, models_dir=default_models_dir(), profile=None)
    window = None
    wake = SingleInstanceWake(root, app)
    if not wake.start():
        raise RuntimeError('Test IPC listen failed: '+wake.server.errorString())

    def wait(predicate, timeout=15):
        deadline = time.monotonic()+timeout
        while not predicate() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        app.processEvents()
        if not predicate():
            raise RuntimeError('Frozen desktop event timeout')

    def close_window():
        window.request_exit()
        wait(lambda: not window.worker.isRunning(), 20)
        app.processEvents()
        window.close()
        app.processEvents()

    try:
        window = ObserverWindow(args)
        wake.activation_requested.connect(window.restore_window)
        window.show()
        wait(lambda: window.start.isEnabled())
        store = ProductStore(root)
        try:
            previous = [row['id'] for row in store.history()]
        finally:
            store.close()
        window.task_notifications.setChecked(False)
        window.title.setText('F 合成会话')
        window.start.click()
        wait(lambda: window.state == 'monitoring')
        window.pause.click()
        wait(lambda: window.state == 'paused')
        window.pause.click()
        wait(lambda: window.state == 'monitoring')
        window.end.click()
        wait(lambda: window.state == 'idle' and window.history.count() >= 1)
        app.processEvents()
        screenshot = root/'synthetic-desktop.png'
        if not window.grab().save(str(screenshot)):
            raise RuntimeError('Desktop render capture failed')
        window.tabs.setCurrentIndex(1)
        app.processEvents()
        calendar_report = {key: window.history_payload[key] for key in ('start', 'end', 'today')}
        calendar_report['selected'] = window.history_calendar.selected.isoformat()
        calendar_report['cells'] = len(window.history_calendar.cells)
        if calendar_report['selected'] != calendar_report['today']:
            raise RuntimeError('History did not select today')
        if not window.grab().save(str(root/'synthetic-history.png')):
            raise RuntimeError('History render capture failed')
        window.hide()
        second = subprocess.Popen([sys.executable, 'product', '--data-dir', str(root), 'gui', '--demo'],
                                  creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            wait(lambda: second.poll() is not None)
            if second.returncode != 0:
                raise RuntimeError(f'Second instance returned {second.returncode}')
            wait(window.isVisible)
        finally:
            if second.poll() is None:
                second.terminate()  # Only the child created by this diagnostic.
                second.wait(timeout=10)
        # Exercise the exact no-argument entry in another process. It must wake
        # this owner before creating a real-mode worker or loading a profile.
        noarg_wake = False
        if default_data_dir().resolve() == root:
            window.hide()
            noarg = subprocess.Popen([sys.executable], creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                wait(lambda: noarg.poll() is not None)
                if noarg.returncode != 0:
                    raise RuntimeError('No-argument launch failed')
                wait(window.isVisible)
                noarg_wake = True
            finally:
                if noarg.poll() is None:
                    noarg.terminate()
                    noarg.wait(timeout=10)
        close_window()
        lock = RuntimeLock(root)
        lock.close()
        wake.activation_requested.disconnect(window.restore_window)
        window.deleteLater()
        app.processEvents()

        store = ProductStore(root)
        try:
            rows = store.history()
            manual_id = rows[0]['id']
            if rows[0]['state'] != 'ended' or len(rows) != len(previous)+1:
                raise RuntimeError('Manual session did not close exactly once')
            store.set_images(True)
            # Explicit synthetic image checks codecs/data preservation, never a camera frame.
            if not store.capture(manual_id, manual_id+'-synthetic', np.zeros((32,32,3), dtype=np.uint8)):
                raise RuntimeError('Synthetic evidence write failed')
        finally:
            store.close()

        window = ObserverWindow(args)
        window.show()
        wait(lambda: window.start.isEnabled())
        if window.task_notifications.isChecked():
            raise RuntimeError('Notification setting did not survive reopening')
        window.task_notifications.setChecked(True)
        submitted = []
        original_notify = window.tray.notify
        def record_notify(message):
            submitted.append(message)
            return original_notify(message)
        window.tray.notify = record_notify
        store = ProductStore(root)
        try:
            task_id = store.add_task('F 定时合成任务', utc_now()+timedelta(seconds=.5), .04)
        finally:
            store.close()
        window.hide()
        wait(lambda: len(submitted) == 1)
        wait(lambda: window.state == 'idle' and window.history.count() >= 1)
        window.task_notifications.setChecked(False)
        close_window()
        lock = RuntimeLock(root)
        lock.close()
        store = ProductStore(root)
        try:
            rows = store.history()
            task_sessions = [dict(r) for r in store.db.execute(
                'SELECT id,state FROM sessions WHERE task_id=?', (task_id,))]
            if len(task_sessions) != 1 or task_sessions[0]['state'] != 'ended':
                raise RuntimeError('Scheduled task did not complete exactly once')
            if not set(previous).issubset({r['id'] for r in rows}) or not store.setting('images'):
                raise RuntimeError('Data/settings were lost on reopening')
            summary = store.summary(manual_id)
        finally:
            store.close()
        for health in (False, True):
            if not _reminder_wave(health).startswith(b'RIFF'):
                raise RuntimeError('Reminder sound generation failed')
        return {'qt_platform': app.platformName(), 'manual_cycle': 'passed',
                'second_process_wake': 'passed', 'lock_reacquired': True,
                'no_argument_entry_wake': noarg_wake,
                'previous_session_ids': previous, 'session_ids': [r['id'] for r in rows],
                'scheduled_task_count': len(task_sessions), 'notification_calls': len(submitted),
                'settings_persisted': True, 'sound_wave_generated': True,
                'audible_sound': 'pending human confirmation', 'windows_toast': 'pending human confirmation',
                'session_identity': json.loads(summary['identity']),
                'desktop_render': str(screenshot), 'history_calendar': calendar_report,
                'evidence_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (root/'evidence').glob('*.jpg')}}
    finally:
        if window is not None and window.worker.isRunning():
            close_window()
        wake.close()


def run(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-resources', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--load-native', action='store_true')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    result = {'started_utc': datetime.now(timezone.utc).isoformat(), 'executable': sys.executable,
              'frozen': bool(getattr(sys, 'frozen', False)), 'camera_opened': False}
    code = 0
    try:
        result['resources'] = check_resources(args.load_native)
        if args.self_test:
            result['desktop'] = desktop_smoke(args.data_dir.resolve())
        result['status'] = 'passed'
    except Exception as exc:
        import traceback
        result.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        code = 1
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
    return code
