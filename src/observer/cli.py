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
from observer.models import sha256_file, verify_model_directory
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


def _vision_smoke(
    config_path: Path | None,
    models_dir: Path,
    seconds: float,
    profile_path: Path | None = None,
) -> int:
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
        profile = None
        if profile_path is not None:
            from observer.calibration import load_profile
            from observer.calibration_workflow import current_environment

            first_snapshot = camera.buffer.latest()
            deadline = time.monotonic() + 5.0
            while first_snapshot is None and time.monotonic() < deadline:
                time.sleep(0.01)
                first_snapshot = camera.buffer.latest()
            if first_snapshot is None:
                raise RuntimeError("camera opened but did not produce a frame")
            environment = current_environment(config, first_snapshot.frame.shape)
            profile = load_profile(profile_path, environment)
        probe = MediapipeVisionProbe(models, config.inference, profile)
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


def _calibrate(config_path: Path | None, models_dir: Path, output_path: Path) -> int:
    from observer.calibration_workflow import run_guided_calibration

    config = load_config(config_path)
    models = verify_model_directory(models_dir)
    profile = run_guided_calibration(config, models, output_path)
    result = {
        "profile_path": str(output_path.resolve()),
        "profile_id": profile.profile_id,
        "schema_version": profile.schema_version,
        "posture_valid_samples": profile.posture.valid_sample_count,
        "posture_coverage": profile.posture.coverage,
        "attention_valid_samples": profile.attention.valid_sample_count,
        "attention_coverage": profile.attention.coverage,
        "attention_training_rmse": profile.attention.training_rmse,
        "attention_validation_rmse": profile.attention.validation_rmse,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _profile_check(profile_path: Path) -> int:
    from observer.calibration import load_profile

    profile = load_profile(profile_path)
    result = {
        "valid": True,
        "path": str(profile_path.resolve()),
        "sha256": sha256_file(profile_path),
        "profile_id": profile.profile_id,
        "schema_version": profile.schema_version,
        "feature_version": profile.posture.feature_version,
        "environment": asdict(profile.environment),
        "posture_coverage": profile.posture.coverage,
        "attention_coverage": profile.attention.coverage,
        "attention_validation_rmse": profile.attention.validation_rmse,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _validation_report(trials_path: Path, output_path: Path) -> int:
    from observer.validation import read_trials, write_metrics_report

    records = read_trials(trials_path)
    report = write_metrics_report(records, output_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _validation_scenarios() -> int:
    from observer.live_validation import scenario_catalog

    print(json.dumps(scenario_catalog(), indent=2, ensure_ascii=False))
    return 0


def _validate_live(
    config_path: Path | None,
    models_dir: Path,
    profile_path: Path,
    dataset_role: str,
    signals: str,
    output_root: Path,
    repetitions: int | None,
    save_error_evidence: bool,
) -> int:
    from observer.live_validation import run_live_validation

    config = load_config(config_path)
    models = verify_model_directory(models_dir)
    signal_names = tuple(name.strip() for name in signals.split(",") if name.strip())
    batch_directory = run_live_validation(
        config,
        models,
        profile_path,
        dataset_role=dataset_role,
        signal_names=signal_names,
        output_root=output_root,
        repetitions_override=repetitions,
        save_error_evidence=save_error_evidence,
    )
    print(
        json.dumps(
            {
                "batch_directory": str(batch_directory.resolve()),
                "report": str((batch_directory / "report.json").resolve()),
                "evidence_enabled": save_error_evidence,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _evidence_purge(index_path: Path, apply: bool) -> int:
    from observer.validation import purge_expired_evidence

    report = purge_expired_evidence(index_path, apply=apply)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


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
    vision_parser.add_argument("--profile", type=Path)
    calibrate_parser = subparsers.add_parser(
        "calibrate", help="run guided posture and 9-point attention calibration"
    )
    calibrate_parser.add_argument("--models-dir", type=Path, default=Path("models"))
    calibrate_parser.add_argument(
        "--output", type=Path, default=Path("data/calibration/default.json")
    )
    profile_parser = subparsers.add_parser(
        "profile-check", help="validate a calibration profile without opening the camera"
    )
    profile_parser.add_argument("profile", type=Path)
    report_parser = subparsers.add_parser(
        "validation-report", help="summarize a Stage C trials JSONL file"
    )
    report_parser.add_argument("trials", type=Path)
    report_parser.add_argument("--output", type=Path, default=Path("data/validation/report.json"))
    subparsers.add_parser(
        "validation-scenarios", help="list built-in Stage C real validation scenarios"
    )
    live_parser = subparsers.add_parser(
        "validate-live", help="run a new isolated tuning or holdout camera batch"
    )
    live_parser.add_argument("--profile", type=Path, required=True)
    live_parser.add_argument("--models-dir", type=Path, default=Path("models"))
    live_parser.add_argument("--role", choices=("tuning", "holdout"), required=True)
    live_parser.add_argument(
        "--signals",
        required=True,
        help="comma-separated: away,phone_held,posture_deviation,attention_away",
    )
    live_parser.add_argument("--output-root", type=Path, default=Path("data/validation"))
    live_parser.add_argument(
        "--repetitions", type=int, help="override only for dry runs; omit for protocol counts"
    )
    live_parser.set_defaults(save_error_evidence=True)
    live_parser.add_argument(
        "--no-error-evidence",
        action="store_false",
        dest="save_error_evidence",
        help="disable the approved one-error-frame validation policy",
    )
    purge_parser = subparsers.add_parser(
        "evidence-purge", help="dry-run or delete expired Stage C error evidence"
    )
    purge_parser.add_argument("index", type=Path)
    purge_parser.add_argument(
        "--apply", action="store_true", help="delete listed expired image files"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return _doctor(args.config)
    if args.command == "camera-smoke":
        return _camera_smoke(args.config, args.seconds)
    if args.command == "vision-smoke":
        return _vision_smoke(args.config, args.models_dir, args.seconds, args.profile)
    if args.command == "calibrate":
        return _calibrate(args.config, args.models_dir, args.output)
    if args.command == "profile-check":
        return _profile_check(args.profile)
    if args.command == "validation-report":
        return _validation_report(args.trials, args.output)
    if args.command == "validation-scenarios":
        return _validation_scenarios()
    if args.command == "validate-live":
        return _validate_live(
            args.config,
            args.models_dir,
            args.profile,
            args.role,
            args.signals,
            args.output_root,
            args.repetitions,
            args.save_error_evidence,
        )
    if args.command == "evidence-purge":
        return _evidence_purge(args.index, args.apply)
    raise AssertionError(f"unknown command: {args.command}")
