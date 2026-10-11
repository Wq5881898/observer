"""Observe an independently authorized Windows camera cycle, without saving frames.

This runner never changes devices or requests elevation. Once its JSON says
waiting_for_device_cycle, run test_e3_camera_cycle.ps1 through normal UAC.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import psutil
from PySide6.QtWidgets import QApplication

from observer.desktop import ObserverWindow
from observer.product import ProductStore, RuntimeLock
from observer.product_cli import FRESH_FRAME_SECONDS
from run_e5_background import wait

CAMERA_ID = r'USB\VID_5986&PID_2130&MI_00\6&E71527D&0&0000'


def utc_text():
    return datetime.now(timezone.utc).isoformat()


def ensure_resource_gate_finished():
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        if process.info['name'] not in ('python.exe', 'pythonw.exe'):
            continue
        command = process.info['cmdline'] or []
        if any(Path(part).name == 'run_e5_background.py' for part in command):
            raise RuntimeError('Wait for the real background resource gate to finish.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ensure_resource_gate_finished()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {'started_utc': utc_text(), 'frames_saved': False, 'passed': False,
              'phase': 'starting', 'notices': [], 'trace': []}

    def checkpoint(phase):
        result['phase'] = phase
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n',
                          encoding='utf-8')

    app = QApplication.instance() or QApplication([])
    app.setStyle('Fusion')
    with tempfile.TemporaryDirectory(prefix='observer-e3-cycle-', dir=output.parent) as directory:
        window = ObserverWindow(SimpleNamespace(
            data_dir=Path(directory), demo=False, models_dir=Path('models').resolve(),
            profile=None))
        snapshot = {}
        window.worker.snapshot.connect(lambda value: snapshot.update(value))
        window.worker.notice.connect(lambda message: result['notices'].append(
            {'utc': utc_text(), 'message': message}))
        try:
            window.show()
            wait(app, lambda: window.start.isEnabled(), 180, 'desktop readiness')
            window.send('start', 'E3 设备不可用测试', 10)
            wait(app, lambda: window.state == 'monitoring', 60, 'first camera session')
            first_id = window.worker.service.identifier
            session_started = window.worker.service.started
            first_camera = window.worker.source.camera
            cycle_started_epoch = time.time()
            checkpoint('waiting_for_device_cycle')
            cycle_report = None
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                app.processEvents()
                source = window.worker.source
                stats = source.camera.stats if source is not None else None
                result['trace'].append({
                    'epoch': time.time(), 'offset': time.monotonic() - session_started,
                    'state': window.state, 'signals': snapshot.get('signals', {}),
                    'frame_age_seconds': (time.monotonic() - stats.last_frame_monotonic
                                          if stats and stats.last_frame_monotonic else None),
                    'frame_count': stats.frames_captured if stats else None,
                    'source_present': source is not None,
                })
                for candidate in output.parent.glob('e3-camera-cycle-*.json'):
                    if candidate.stat().st_mtime < cycle_started_epoch:
                        continue
                    try:
                        report = json.loads(candidate.read_text(encoding='utf-8-sig'))
                    except (OSError, ValueError):
                        continue  # The helper may still be writing its final report.
                    if (report.get('instance_id') == CAMERA_ID
                            and datetime.fromisoformat(report['started_utc']).timestamp()
                            >= cycle_started_epoch):
                        cycle_report = report
                        result['device_report_path'] = str(candidate)
                        break
                if cycle_report is not None:
                    break
                time.sleep(.2)
            if cycle_report is None:
                raise RuntimeError('No matching device-cycle report; no device result inferred.')
            result['device_cycle'] = cycle_report
            if (cycle_report.get('error') or cycle_report.get('restore_error')
                    or not cycle_report.get('disabled_verified')
                    or not cycle_report.get('restored_verified')):
                raise RuntimeError('The exact camera disable/restore was not verified.')
            disabled = datetime.fromisoformat(cycle_report['disabled_utc']).timestamp()
            restored = datetime.fromisoformat(cycle_report['restored_utc']).timestamp()
            unavailable = [entry for entry in result['trace']
                           if disabled <= entry['epoch'] <= restored
                           and (not entry['source_present'] or
                                (entry['frame_age_seconds'] is not None
                                 and entry['frame_age_seconds'] > FRESH_FRAME_SECONDS))]
            if not unavailable:
                raise RuntimeError('Device cycle did not demonstrate a real frame gap.')
            gap_start = unavailable[0]['offset']
            gap_end = gap_start + restored - unavailable[0]['epoch']
            result['unavailable_offsets'] = [gap_start, gap_end]
            wait(app, lambda: window.state == 'idle' or (
                window.state == 'monitoring' and window.worker.source is not None
                and window.worker.source.camera.stats.last_frame_monotonic is not None
                and time.monotonic() - window.worker.source.camera.stats.last_frame_monotonic
                <= FRESH_FRAME_SECONDS), 30, 'bounded interruption or fresh-frame recovery')
            if window.state == 'monitoring':
                window.send('end')
                wait(app, lambda: window.state == 'idle', 30, 'first session end')
            result['first_camera_released'] = not first_camera.stats.opened
            checkpoint('starting_recovery_session')
            window.send('start', 'E3 恢复后会话', 2)
            wait(app, lambda: window.state == 'monitoring', 60, 'new camera session')
            second_id = window.worker.service.identifier
            second_camera = window.worker.source.camera
            captured = second_camera.stats.last_frame_monotonic
            wait(app, lambda: second_camera.stats.last_frame_monotonic is not None
                 and second_camera.stats.last_frame_monotonic > captured,
                 5, 'new session fresh frames')
            window.send('end')
            wait(app, lambda: window.state == 'idle', 30, 'recovery session end')
            result['second_camera_released'] = not second_camera.stats.opened
            exit_started = time.monotonic()
            window.request_exit()
            wait(app, lambda: not window.worker.isRunning(), 30, 'desktop exit')
            result['exit_seconds'] = time.monotonic() - exit_started
            result['camera_threads_alive'] = [thread.name for thread in threading.enumerate()
                                              if thread.name == 'observer-camera']
            RuntimeLock(Path(directory)).close()
            result['runtime_lock_released'] = True
            store = ProductStore(Path(directory))
            try:
                summaries = [store.summary(identifier) for identifier in (first_id, second_id)]
                result['sessions'] = summaries
                result['session_count'] = len(store.history())
            finally:
                store.close()
            overlap = sum(max(0., min(end, gap_end) - max(start, gap_start))
                          for incident in summaries[0]['focus_incidents']
                          for segments in incident['segments'].values()
                          for start, end in segments)
            result['violation_seconds_in_observed_gap'] = overlap
            result['passed'] = (
                result['session_count'] == 2 and result['first_camera_released']
                and result['second_camera_released'] and result['runtime_lock_released']
                and summaries[0]['state'] in ('interrupted', 'ended')
                and summaries[1]['state'] == 'ended' and not summaries[1]['error']
                and overlap == 0 and result['exit_seconds'] <= 5
                and not result['camera_threads_alive']
                and all(not summary['evidence'] for summary in summaries))
        except Exception as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
        finally:
            if window.worker.isRunning():
                window.request_exit()
                try:
                    wait(app, lambda: not window.worker.isRunning(), 30, 'failure cleanup')
                except RuntimeError as exc:
                    result['cleanup_error'] = str(exc)
            result['worker_stopped'] = not window.worker.isRunning()
            result['passed'] = result['passed'] and result['worker_stopped']
            result['finished_utc'] = utc_text()
            checkpoint('completed' if result['passed'] else 'failed')
            if result['worker_stopped']:
                window.close()
                app.processEvents()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
