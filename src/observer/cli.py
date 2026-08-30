from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from observer.camera import CameraWorker
from observer.config import load_config
from observer.models import verify_model_directory
from observer.resources import ResourceSampler, summarize_resources


def _doctor(config_path: Path | None) -> int:
    config = load_config(config_path)
    packages = {
        name: importlib.util.find_spec(name) is not None
        for name in ("cv2", "mediapipe", "numpy", "psutil")
    }
    version_supported = sys.version_info[:2] in {(3, 11), (3, 12)}
    report = {
        "python": platform.python_version(),
        "python_supported": version_supported,
        "platform": platform.platform(),
        "packages": packages,
        "config_valid": True,
        "camera": asdict(config.camera),
        "resource_budget": asdict(config.resources),
        "rules": [asdict(rule) for rule in config.rules],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if version_supported and all(packages.values()) else 1


def _camera_smoke(config_path: Path | None, seconds: float) -> int:
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    config = load_config(config_path)
    camera = CameraWorker(config.camera)
    sampler = ResourceSampler()
    samples = []
    start = time.monotonic()
    last_sequence = 0
    observed_frames = 0
    try:
        camera.start()
        startup_stats = camera.stats
        if not startup_stats.opened:
            result = {
                "duration_seconds": round(time.monotonic() - start, 2),
                "observed_latest_frames": 0,
                "camera": asdict(startup_stats),
                "resources": asdict(
                    summarize_resources([sampler.sample()], config.resources)
                ),
            }
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 2
        while time.monotonic() - start < seconds:
            snapshot = camera.buffer.latest(after_sequence=last_sequence)
            if snapshot is not None:
                observed_frames += 1
                last_sequence = snapshot.sequence
            samples.append(sampler.sample())
            time.sleep(1.0)
    finally:
        camera.stop()

    result = {
        "duration_seconds": round(time.monotonic() - start, 2),
        "observed_latest_frames": observed_frames,
        "camera": asdict(camera.stats),
        "resources": asdict(summarize_resources(samples, config.resources)),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if camera.stats.frames_captured > 0 and camera.stats.last_error is None else 2


def _vision_smoke(config_path: Path | None, models_dir: Path, seconds: float) -> int:
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    from observer.vision import MediapipeVisionProbe

    config = load_config(config_path)
    models = verify_model_directory(models_dir)
    camera = CameraWorker(config.camera)
    probe = None
    samples = []
    last_sequence = 0
    next_resource_sample = 0.0
    last_observation = None
    try:
        if not camera.start():
            print(json.dumps({"camera": asdict(camera.stats)}, indent=2, ensure_ascii=False))
            return 2
        probe = MediapipeVisionProbe(models, config.inference)
        sampler = ResourceSampler()
        started = time.monotonic()
        next_resource_sample = started
        while time.monotonic() - started < seconds:
            snapshot = camera.buffer.latest(after_sequence=last_sequence)
            if snapshot is None:
                time.sleep(0.005)
                continue
            last_sequence = snapshot.sequence
            probe.process_due(snapshot.frame, snapshot.captured_monotonic)
            last_observation = probe.observation(snapshot.captured_monotonic)
            now = time.monotonic()
            if now >= next_resource_sample:
                samples.append(sampler.sample())
                next_resource_sample = now + 1.0
    finally:
        if probe is not None:
            probe.close()
        camera.stop()

    result = {
        "duration_seconds": seconds,
        "camera": asdict(camera.stats),
        "vision": probe.summary() if probe is not None else None,
        "last_signals": (
            {name: value.state.value for name, value in last_observation.signals.items()}
            if last_observation is not None
            else None
        ),
        "resources": asdict(summarize_resources(samples, config.resources)),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if probe is None:
        return 3
    runs = probe.measurements
    return 0 if runs.face.runs and runs.pose.runs and runs.object.runs else 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="observer")
    parser.add_argument("--config", type=Path, help="optional JSON configuration path")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="validate runtime, packages, and configuration")
    camera_parser = subparsers.add_parser(
        "camera-smoke", help="capture without preview or recording and report resources"
    )
    camera_parser.add_argument("--seconds", type=float, default=15.0)
    vision_parser = subparsers.add_parser(
        "vision-smoke", help="run rate-limited Face, Pose, and Object models without recording"
    )
    vision_parser.add_argument("--seconds", type=float, default=30.0)
    vision_parser.add_argument("--models-dir", type=Path, default=Path("models"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return _doctor(args.config)
    if args.command == "camera-smoke":
        return _camera_smoke(args.config, args.seconds)
    if args.command == "vision-smoke":
        return _vision_smoke(args.config, args.models_dir, args.seconds)
    raise AssertionError(f"unknown command: {args.command}")
