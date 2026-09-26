"""Classify a built-in camera privacy switch without saving any frames."""
from __future__ import annotations

import argparse
import json
import statistics
import time

from observer.camera import CameraWorker
from observer.config import load_config


def cue(count: int) -> None:
    try:
        import winsound

        for _ in range(count):
            winsound.Beep(880, 180)
            time.sleep(.12)
    except (ImportError, RuntimeError):
        print('\a' * count, end='', flush=True)


def sample(camera: CameraWorker, seconds: float, after_sequence: int = 0) -> dict:
    started = time.monotonic()
    sequence = after_sequence
    first_sequence = None
    last_sequence = None
    brightness = []
    freshness = []
    while time.monotonic() - started < seconds:
        snapshot = camera.buffer.latest(after_sequence=sequence)
        if snapshot is not None:
            sequence = snapshot.sequence
            first_sequence = sequence if first_sequence is None else first_sequence
            last_sequence = sequence
            # Mean intensity is enough to distinguish a mechanical shutter from
            # a device disconnect. No frame leaves memory or is written to disk.
            brightness.append(float(snapshot.frame.mean()))
            freshness.append(time.monotonic() - snapshot.captured_monotonic)
        time.sleep(.03)
    stats = camera.stats
    return {
        'seconds': round(time.monotonic() - started, 3),
        'new_frames': (last_sequence - first_sequence + 1
                       if first_sequence is not None else 0),
        'first_sequence': first_sequence,
        'last_sequence': last_sequence,
        'brightness_mean': round(statistics.fmean(brightness), 3) if brightness else None,
        'brightness_min_max': ([round(min(brightness), 3), round(max(brightness), 3)]
                               if brightness else []),
        'freshness_max_seconds': round(max(freshness), 3) if freshness else None,
        'camera_frames_total': stats.frames_captured,
        'read_failures_total': stats.read_failures,
        'last_error': stats.last_error,
        'opened': stats.opened,
        'ending_sequence': sequence,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-seconds', type=float, default=8.)
    parser.add_argument('--closed-seconds', type=float, default=12.)
    parser.add_argument('--recovery-seconds', type=float, default=12.)
    args = parser.parse_args()
    camera = CameraWorker(load_config().camera)
    result = {'frames_saved': False}
    try:
        if not camera.start():
            result['classification'] = 'startup_failed'
            result['error'] = camera.stats.last_error
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 2
        print('一声提示：测试开始。保持摄像头开关为打开状态；正在采集基线。', flush=True)
        cue(1)
        result['baseline'] = sample(camera, args.baseline_seconds)
        print('两声提示：现在把摄像头开关拨到关闭。', flush=True)
        cue(2)
        result['switch_closed'] = sample(
            camera, args.closed_seconds, result['baseline']['ending_sequence'])
        print('三声提示：现在把摄像头开关拨回打开。', flush=True)
        cue(3)
        result['recovery'] = sample(
            camera, args.recovery_seconds, result['switch_closed']['ending_sequence'])

        closed = result['switch_closed']
        recovery = result['recovery']
        baseline = result['baseline']
        if closed['new_frames'] == 0 or closed['read_failures_total'] > baseline['read_failures_total']:
            result['classification'] = 'device_disconnect'
        else:
            result['classification'] = 'physical_shutter_or_occlusion'
        result['recovered'] = (
            recovery['new_frames'] > 0
            and recovery['last_error'] is None
            and recovery['opened']
        )
    finally:
        camera.stop()
        result['camera_released'] = not camera.stats.opened
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get('recovered') and result['camera_released'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
