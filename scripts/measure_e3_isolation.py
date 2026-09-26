"""Measure model ownership options without opening a camera.

Run from the repository root with the project Python. Results are JSON so they
can be checked into an E-stage evidence report without parsing human text.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import psutil


MIB = 1024 * 1024


def rss_mb(process: psutil.Process) -> float:
    try:
        return process.memory_info().rss / MIB
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


def tree_rss_mb(process: psutil.Process) -> float:
    total = rss_mb(process)
    try:
        children = process.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        children = []
    return total + sum(rss_mb(child) for child in children)


def load_models(models_dir: Path):
    from observer.product_cli import prepare_camera_resources
    started = time.monotonic()
    prepared = prepare_camera_resources(models_dir, None, background=True)
    return prepared, time.monotonic() - started


def child_main(models_dir: Path) -> int:
    process = psutil.Process()
    baseline = rss_mb(process)
    prepared = None
    try:
        prepared, elapsed = load_models(models_dir)
        print(json.dumps({
            'status': 'ready',
            'pid': os.getpid(),
            'baseline_rss_mb': round(baseline, 3),
            'ready_rss_mb': round(rss_mb(process), 3),
            'load_seconds': round(elapsed, 3),
        }), flush=True)
        sys.stdin.readline()
        close_started = time.monotonic()
        prepared.close()
        prepared = None
        gc.collect()
        print(json.dumps({
            'status': 'closed',
            'close_seconds': round(time.monotonic() - close_started, 3),
            'closed_rss_mb': round(rss_mb(process), 3),
        }), flush=True)
        return 0
    except BaseException as exc:
        print(json.dumps({
            'status': 'failed', 'error_type': type(exc).__name__,
            'error': str(exc)[:500],
        }), flush=True)
        return 1
    finally:
        if prepared is not None:
            prepared.close()


def read_json_line(process: subprocess.Popen, deadline_seconds=120.) -> dict:
    # Windows pipes cannot be selected. A daemon reader keeps the caller's
    # deadline bounded even if native model startup never writes a line.
    result = queue.Queue(maxsize=1)

    def read():
        try:
            result.put(process.stdout.readline())
        except BaseException as exc:
            result.put(exc)

    threading.Thread(target=read, name='e3-measure-reader', daemon=True).start()
    try:
        value = result.get(timeout=deadline_seconds)
    except queue.Empty as exc:
        raise RuntimeError('model child did not report readiness before timeout') from exc
    if isinstance(value, BaseException):
        raise value
    if not value:
        stderr = process.stderr.read() if process.poll() is not None else ''
        raise RuntimeError('model child exited without readiness: ' + stderr[-500:])
    return json.loads(value)


def isolated_measurement(models_dir: Path) -> dict:
    parent = psutil.Process()
    before = tree_rss_mb(parent)
    command = [sys.executable, str(Path(__file__).resolve()),
               '--child', '--models-dir', str(models_dir)]
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, bufsize=1,
    )
    child = psutil.Process(process.pid)
    try:
        ready = read_json_line(process)
        if ready.get('status') != 'ready':
            raise RuntimeError(f"model child failed: {ready.get('error')}")
        combined_ready = tree_rss_mb(parent)
        release_started = time.monotonic()
        process.stdin.write('\n')
        process.stdin.flush()
        closed = read_json_line(process)
        return_code = process.wait(timeout=30)
        exit_seconds = time.monotonic() - release_started
        if return_code != 0 or closed.get('status') != 'closed':
            raise RuntimeError('model child did not close cleanly')
        return {
            'parent_tree_before_mb': round(before, 3),
            'child_baseline_mb': ready['baseline_rss_mb'],
            'child_ready_mb': ready['ready_rss_mb'],
            'combined_ready_mb': round(combined_ready, 3),
            'child_load_seconds': ready['load_seconds'],
            'child_close_seconds': closed['close_seconds'],
            'child_closed_mb': closed['closed_rss_mb'],
            'child_exit_seconds': round(exit_seconds, 3),
            'parent_tree_after_mb': round(tree_rss_mb(parent), 3),
        }
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


def desktop_measurement(models_dir: Path, *, prewarm=True) -> dict:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    from observer.desktop import ObserverWindow

    process = psutil.Process()
    started = time.monotonic()
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory(prefix='observer-e3-') as directory:
        args = SimpleNamespace(
            data_dir=Path(directory), demo=not prewarm, models_dir=models_dir,
            profile=None,
        )
        window = ObserverWindow(args)
        window.show()
        deadline = time.monotonic() + 120
        peak = rss_mb(process)
        while ((prewarm and not window.worker.preloader.ready or
                not prewarm and window.blocked) and window.worker.isRunning()
               and time.monotonic() < deadline):
            app.processEvents()
            peak = max(peak, rss_mb(process))
            time.sleep(.01)
        app.processEvents()
        if prewarm and not window.worker.preloader.ready:
            raise RuntimeError('desktop prewarm did not become ready')
        ready = rss_mb(process)
        threads = process.num_threads()
        handles = process.num_handles() if hasattr(process, 'num_handles') else None
        close_started = time.monotonic()
        window.request_exit()
        while window.worker.isRunning() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        app.processEvents()
        close_seconds = time.monotonic() - close_started
        if window.worker.isRunning():
            raise RuntimeError('desktop worker did not close')
        window.close()
        app.processEvents()
        gc.collect()
        return {
            'startup_to_ready_seconds': round(time.monotonic() - started - close_seconds, 3),
            'ready_rss_mb': round(ready, 3),
            'observed_peak_rss_mb': round(peak, 3),
            'threads_ready': threads,
            'handles_ready': handles,
            'close_seconds': round(close_seconds, 3),
            'closed_rss_mb': round(rss_mb(process), 3),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--scenario', choices=(
        'inprocess', 'isolated', 'duplicate', 'desktop', 'desktop-shell'),
                        default='inprocess')
    args = parser.parse_args()
    models_dir = args.models_dir.resolve()
    if args.child:
        return child_main(models_dir)

    process = psutil.Process()
    baseline = rss_mb(process)
    result = {'scenario': args.scenario, 'models_dir': str(models_dir),
              'baseline_rss_mb': round(baseline, 3)}
    prepared = None
    try:
        if args.scenario in ('desktop', 'desktop-shell'):
            result['desktop'] = desktop_measurement(
                models_dir, prewarm=args.scenario == 'desktop')
        if args.scenario in ('inprocess', 'duplicate'):
            prepared, elapsed = load_models(models_dir)
            result.update({
                'parent_load_seconds': round(elapsed, 3),
                'parent_ready_rss_mb': round(rss_mb(process), 3),
            })
        if args.scenario in ('isolated', 'duplicate'):
            result['child'] = isolated_measurement(models_dir)
        if prepared is not None:
            close_started = time.monotonic()
            prepared.close()
            prepared = None
            gc.collect()
            result.update({
                'parent_close_seconds': round(time.monotonic() - close_started, 3),
                'parent_closed_rss_mb': round(rss_mb(process), 3),
            })
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    finally:
        if prepared is not None:
            prepared.close()


if __name__ == '__main__':
    raise SystemExit(main())
