"""Run the real desktop owner hidden and record E5 background resources."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
from PySide6.QtWidgets import QApplication

from observer.desktop import ObserverWindow
from observer.product import ProductStore, RuntimeLock
from observer.product_cli import DEFAULT_PROFILE


def percentile(values, probability):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * probability) - 1)] if ordered else 0.


def wait(app, predicate, timeout, label):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    app.processEvents()
    if not predicate():
        raise RuntimeError(f'{label} timed out after {timeout:g}s')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=60.)
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--no-profile', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError('seconds must be positive')
    profile = None if args.no_profile else args.profile
    if profile is None and not args.no_profile and DEFAULT_PROFILE.is_file():
        profile = DEFAULT_PROFILE

    app = QApplication.instance() or QApplication([])
    app.setStyle('Fusion')
    process = psutil.Process()
    logical_cpus = max(psutil.cpu_count(logical=True) or 1, 1)
    process.cpu_percent(None)
    samples = []
    result = {'requested_seconds': args.seconds, 'profile': str(profile) if profile else None}

    with tempfile.TemporaryDirectory(prefix='observer-e5-') as directory:
        root = Path(directory)
        window = ObserverWindow(SimpleNamespace(
            data_dir=root, demo=False, models_dir=args.models_dir.resolve(), profile=profile))
        try:
            window.show()
            wait(app, lambda: window.start.isEnabled(), 180, 'desktop readiness')
            start_requested = time.monotonic()
            window.send('start', 'E5 后台长测', args.seconds / 60 + 5)
            wait(app, lambda: window.state == 'monitoring', 60, 'session startup')
            result['session_startup_seconds'] = round(time.monotonic() - start_requested, 3)
            window.hide()
            window.hidden_to_tray = True
            wait(app, lambda: not window.worker.ui_visible, 5, 'background transition')
            started = time.monotonic()
            next_sample = started
            while time.monotonic() - started < args.seconds:
                app.processEvents()
                now = time.monotonic()
                if now >= next_sample:
                    memory = process.memory_info().rss / (1024 * 1024)
                    samples.append({
                        'elapsed': now - started,
                        'cpu': process.cpu_percent(None) / logical_cpus,
                        'memory': memory,
                        'threads': process.num_threads(),
                        'handles': process.num_handles() if hasattr(process, 'num_handles') else None,
                    })
                    next_sample = now + 1
                time.sleep(.01)
            result['actual_seconds'] = round(time.monotonic() - started, 3)
            window.send('end')
            wait(app, lambda: window.state == 'idle', 30, 'session end')
            result['camera_released_after_end'] = window.worker.source is None
            session_id = window.worker.last_session_id
            exit_started = time.monotonic()
            window.request_exit()
            wait(app, lambda: not window.worker.isRunning(), 30, 'desktop exit')
            result['exit_seconds'] = round(time.monotonic() - exit_started, 3)
            window.close()
            app.processEvents()
            RuntimeLock(root).close()
            result['runtime_lock_released'] = True
            # The product connection belonged to the stopped worker. Open a
            # fresh connection here instead of crossing the thread boundary.
            store = ProductStore(root)
            try:
                summary = store.summary(session_id)
            finally:
                store.close()
            result['session_state'] = summary['state']
            result['product_resources'] = summary.get('resource_summary')
        finally:
            if window.worker.isRunning():
                window.request_exit()
                window.worker.wait(30_000)
            window.close()

        cpu = [sample['cpu'] for sample in samples]
        memory = [sample['memory'] for sample in samples]
        threads = [sample['threads'] for sample in samples]
        handles = [sample['handles'] for sample in samples if sample['handles'] is not None]
        elapsed_hours = ((samples[-1]['elapsed'] - samples[0]['elapsed']) / 3600
                         if len(samples) >= 2 else 0)
        result['samples'] = len(samples)
        result['cpu_average_percent'] = round(statistics.fmean(cpu), 3) if cpu else 0
        result['cpu_p95_percent'] = round(percentile(cpu, .95), 3)
        result['memory_peak_mb'] = round(max(memory), 3) if memory else 0
        result['memory_start_end_mb'] = (
            [round(memory[0], 3), round(memory[-1], 3)] if memory else [])
        result['memory_growth_mb_per_hour'] = (
            round(max(0., memory[-1] - memory[0]) / elapsed_hours, 3)
            if elapsed_hours >= .24 else None)
        result['threads_start_end_peak'] = (
            [threads[0], threads[-1], max(threads)] if threads else [])
        result['handles_start_end_peak'] = (
            [handles[0], handles[-1], max(handles)] if handles else [])
        result['minute_checkpoints'] = [
            {
                'elapsed': round(sample['elapsed'], 1),
                'memory_mb': round(sample['memory'], 3),
                'threads': sample['threads'],
                'handles': sample['handles'],
            }
            for index, sample in enumerate(samples)
            if index == 0 or index == len(samples)-1
            or int(sample['elapsed']) // 60 != int(samples[index-1]['elapsed']) // 60
        ]
        result['within_cpu_memory_budget'] = (
            result['cpu_average_percent'] <= 15
            and result['cpu_p95_percent'] <= 25
            and result['memory_peak_mb'] <= 500
            and (result['memory_growth_mb_per_hour'] is None
                 or result['memory_growth_mb_per_hour'] <= 50))

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n', encoding='utf-8')
    print(rendered)
    product_budget = result.get('product_resources') or {}
    return 0 if (result['within_cpu_memory_budget']
                 and product_budget.get('within_budget', False)
                 and result['camera_released_after_end']
                 and result['runtime_lock_released']
                 and result['session_state'] == 'ended') else 1


if __name__ == '__main__':
    raise SystemExit(main())
