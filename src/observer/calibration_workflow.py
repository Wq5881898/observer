from __future__ import annotations

import ctypes
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from observer.calibration import (
    ATTENTION_POINTS_9,
    AttentionSample,
    CalibrationProfile,
    EnvironmentFingerprint,
    aggregate_attention_samples,
    create_profile,
    fit_attention_calibration,
    fit_posture_calibration,
    save_profile,
)
from observer.camera import CameraWorker
from observer.config import AppConfig
from observer.models import ModelFiles
from observer.observations import SignalState
from observer.perception import FeatureVector
from observer.vision import MediapipeVisionProbe


class CalibrationCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CalibrationTimings:
    precheck_seconds: float = 5.0
    posture_settle_seconds: float = 5.0
    posture_sample_seconds: float = 25.0
    posture_retest_seconds: float = 10.0
    attention_settle_seconds: float = 1.5
    attention_sample_seconds: float = 2.0
    attention_validation_settle_seconds: float = 1.0
    attention_validation_sample_seconds: float = 1.5


ATTENTION_VALIDATION_POINTS: tuple[tuple[str, float, float], ...] = (
    ("validate_top_left", 0.25, 0.08),
    ("validate_top_right", 0.75, 0.08),
    ("validate_bottom_left", 0.25, 0.92),
    ("validate_bottom_right", 0.75, 0.92),
    ("validate_left_top", 0.08, 0.25),
    ("validate_left_bottom", 0.08, 0.75),
    ("validate_right_top", 0.92, 0.25),
    ("validate_right_bottom", 0.92, 0.75),
)

MAX_ATTENTION_VALIDATION_RMSE = 0.18


def current_environment(
    config: AppConfig, frame_shape: tuple[int, ...], *, mirrored: bool = False
) -> EnvironmentFingerprint:
    user32 = ctypes.windll.user32
    screen_width = int(user32.GetSystemMetrics(0))
    screen_height = int(user32.GetSystemMetrics(1))
    try:
        display_scale = round(int(user32.GetDpiForSystem()) / 96 * 100)
    except (AttributeError, OSError):
        display_scale = 100
    return EnvironmentFingerprint(
        camera_index=config.camera.device_index,
        frame_width=int(frame_shape[1]),
        frame_height=int(frame_shape[0]),
        mirrored=mirrored,
        screen_width=screen_width,
        screen_height=screen_height,
        display_scale_percent=display_scale,
    )


def run_guided_calibration(
    config: AppConfig,
    models: ModelFiles,
    output_path: str | Path,
    *,
    timings: CalibrationTimings | None = None,
) -> CalibrationProfile:
    """Run the approved local posture and 9-point attention calibration."""

    import cv2

    selected = timings or CalibrationTimings()
    camera = CameraWorker(config.camera)
    probe: MediapipeVisionProbe | None = None
    window = "Observer calibration"
    try:
        if not camera.start():
            raise RuntimeError(camera.stats.last_error or "camera could not be opened")
        first = _wait_for_frame(camera, 5.0)
        environment = current_environment(config, first.frame.shape)
        probe = MediapipeVisionProbe(models, config.inference)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        _run_preview_phase(
            camera,
            probe,
            window,
            "Precheck: face, eyes and both shoulders visible",
            selected.precheck_seconds,
            require_quality_coverage=0.90,
        )
        _run_preview_phase(
            camera,
            probe,
            window,
            "Settle into a comfortable neutral posture",
            selected.posture_settle_seconds,
        )
        posture_samples, posture_attempts = _collect_feature_phase(
            camera,
            probe,
            window,
            "Hold your normal neutral posture",
            selected.posture_sample_seconds,
            feature_kind="posture",
        )
        posture = fit_posture_calibration(
            posture_samples,
            planned_sample_count=posture_attempts,
            minimum_valid_samples=75,
        )
        if posture.coverage < 0.90:
            raise RuntimeError(
                f"posture calibration coverage {posture.coverage:.1%} is below 90%"
            )

        retest_samples, _ = _collect_feature_phase(
            camera,
            probe,
            window,
            "Neutral posture retest",
            selected.posture_retest_seconds,
            feature_kind="posture",
        )
        if not retest_samples:
            raise RuntimeError("posture retest produced no observable samples")
        retest_violations = sum(
            posture.evaluate(sample).state is SignalState.VIOLATION
            for sample in retest_samples
        )
        if retest_violations / len(retest_samples) > 0.10:
            raise RuntimeError("posture retest rejected the calibration profile")

        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        attention_raw: list[AttentionSample] = []
        attention_attempts = 0
        for point_id, x, y in ATTENTION_POINTS_9:
            _show_attention_point(
                camera,
                probe,
                window,
                point_id,
                x,
                y,
                selected.attention_settle_seconds,
                collect=False,
            )
            collected, attempts = _show_attention_point(
                camera,
                probe,
                window,
                point_id,
                x,
                y,
                selected.attention_sample_seconds,
                collect=True,
            )
            attention_raw.extend(
                AttentionSample(sample, x, y, point_id) for sample in collected
            )
            attention_attempts += attempts

        validation_raw: list[AttentionSample] = []
        for point_id, x, y in ATTENTION_VALIDATION_POINTS:
            _show_attention_point(
                camera,
                probe,
                window,
                point_id,
                x,
                y,
                selected.attention_validation_settle_seconds,
                collect=False,
            )
            collected, _ = _show_attention_point(
                camera,
                probe,
                window,
                point_id,
                x,
                y,
                selected.attention_validation_sample_seconds,
                collect=True,
            )
            validation_raw.extend(
                AttentionSample(sample, x, y, point_id) for sample in collected
            )

        attention_points = aggregate_attention_samples(attention_raw)
        validation_points = aggregate_attention_samples(validation_raw)
        if len(attention_points) != len(ATTENTION_POINTS_9):
            raise RuntimeError("one or more attention calibration points had no valid samples")
        if len(validation_points) != len(ATTENTION_VALIDATION_POINTS):
            raise RuntimeError("one or more independent attention validation points failed")
        attention = fit_attention_calibration(
            attention_raw,
            validation_samples=validation_raw,
            planned_sample_count=attention_attempts,
            valid_frame_count=len(attention_raw),
        )
        valid_attention_frames = len(attention_raw)
        if valid_attention_frames / max(attention_attempts, 1) < 0.90:
            raise RuntimeError("attention calibration coverage is below 90%")
        if (
            attention.validation_rmse is None
            or attention.validation_rmse > MAX_ATTENTION_VALIDATION_RMSE
        ):
            raise RuntimeError(
                "attention edge validation RMSE "
                f"{attention.validation_rmse!r} exceeds {MAX_ATTENTION_VALIDATION_RMSE}"
            )

        profile = create_profile(environment, posture, attention)
        save_profile(profile, output_path)
        _save_derived_samples(
            output_path,
            profile,
            posture_samples,
            retest_samples,
            attention_raw,
            validation_raw,
        )
        return profile
    finally:
        if probe is not None:
            probe.close()
        camera.stop()
        cv2.destroyAllWindows()


def _wait_for_frame(camera: CameraWorker, timeout_seconds: float):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        snapshot = camera.buffer.latest()
        if snapshot is not None:
            return snapshot
        time.sleep(0.01)
    raise RuntimeError("camera opened but did not produce a frame")


def _run_preview_phase(
    camera: CameraWorker,
    probe: MediapipeVisionProbe,
    window: str,
    message: str,
    duration_seconds: float,
    *,
    require_quality_coverage: float | None = None,
) -> None:
    import cv2

    started = time.monotonic()
    last_sequence = 0
    checked = 0
    observable = 0
    while time.monotonic() - started < duration_seconds:
        snapshot = camera.buffer.latest(after_sequence=last_sequence)
        if snapshot is None:
            _check_cancel(cv2.waitKey(1))
            continue
        last_sequence = snapshot.sequence
        probe.process_due(
            snapshot.frame, snapshot.captured_monotonic, frozenset({"face", "pose"})
        )
        if probe.latest_quality is not None:
            checked += 1
            observable += int(
                probe.latest_quality.observable
                and probe.latest_attention_features is not None
                and probe.latest_attention_features.observable
                and probe.latest_posture_features is not None
                and probe.latest_posture_features.observable
            )
        preview = cv2.flip(snapshot.frame, 1)
        remaining = max(0.0, duration_seconds - (time.monotonic() - started))
        _put_text(preview, f"{message}  {remaining:0.1f}s")
        cv2.imshow(window, preview)
        _check_cancel(cv2.waitKey(1))
    if require_quality_coverage is not None:
        coverage = observable / max(checked, 1)
        if coverage < require_quality_coverage:
            reasons: list[str] = []
            if probe.latest_quality is None:
                reasons.append("no quality result")
            else:
                reasons.extend(probe.latest_quality.reasons)
            if probe.latest_attention_features is None:
                reasons.append("no observable face/eye features")
            else:
                reasons.extend(probe.latest_attention_features.reasons)
            if probe.latest_posture_features is None:
                reasons.append("no observable shoulder/posture features")
            else:
                reasons.extend(probe.latest_posture_features.reasons)
            reason = "; ".join(dict.fromkeys(reasons)) or "required landmarks unavailable"
            raise RuntimeError(
                f"precheck quality coverage {coverage:.1%} is below "
                f"{require_quality_coverage:.0%}: {reason}"
            )


def _collect_feature_phase(
    camera: CameraWorker,
    probe: MediapipeVisionProbe,
    window: str,
    message: str,
    duration_seconds: float,
    *,
    feature_kind: str,
) -> tuple[list[FeatureVector], int]:
    import cv2

    if feature_kind not in {"attention", "posture"}:
        raise ValueError("feature kind must be attention or posture")
    started = time.monotonic()
    last_sequence = 0
    previous_runs = getattr(probe.measurements, "face" if feature_kind == "attention" else "pose").runs
    samples: list[FeatureVector] = []
    attempts = 0
    while time.monotonic() - started < duration_seconds:
        snapshot = camera.buffer.latest(after_sequence=last_sequence)
        if snapshot is None:
            _check_cancel(cv2.waitKey(1))
            continue
        last_sequence = snapshot.sequence
        probe.process_due(
            snapshot.frame, snapshot.captured_monotonic, frozenset({"face", "pose"})
        )
        measurement = getattr(
            probe.measurements, "face" if feature_kind == "attention" else "pose"
        )
        if measurement.runs > previous_runs:
            attempts += measurement.runs - previous_runs
            feature = (
                probe.latest_attention_features
                if feature_kind == "attention"
                else probe.latest_posture_features
            )
            if feature is not None and feature.observable and (
                probe.latest_quality is not None and probe.latest_quality.observable
            ):
                samples.append(feature)
            previous_runs = measurement.runs
        preview = cv2.flip(snapshot.frame, 1)
        remaining = max(0.0, duration_seconds - (time.monotonic() - started))
        _put_text(preview, f"{message}  {remaining:0.1f}s  valid={len(samples)}")
        cv2.imshow(window, preview)
        _check_cancel(cv2.waitKey(1))
    return samples, attempts


def _show_attention_point(
    camera: CameraWorker,
    probe: MediapipeVisionProbe,
    window: str,
    point_id: str,
    x: float,
    y: float,
    duration_seconds: float,
    *,
    collect: bool,
) -> tuple[list[FeatureVector], int]:
    import cv2

    screen_width = max(640, int(cv2.getWindowImageRect(window)[2]))
    screen_height = max(480, int(cv2.getWindowImageRect(window)[3]))
    canvas = np.full((screen_height, screen_width, 3), 245, dtype=np.uint8)
    point = (int(x * (screen_width - 1)), int(y * (screen_height - 1)))
    cv2.circle(canvas, point, 18, (30, 80, 220), -1)
    cv2.circle(canvas, point, 5, (255, 255, 255), -1)
    _put_text(canvas, f"Look at the dot: {point_id}  ESC cancels")
    started = time.monotonic()
    last_sequence = 0
    previous_runs = probe.measurements.face.runs
    attempts = 0
    samples: list[FeatureVector] = []
    while time.monotonic() - started < duration_seconds:
        snapshot = camera.buffer.latest(after_sequence=last_sequence)
        if snapshot is not None:
            last_sequence = snapshot.sequence
            probe.process_due(snapshot.frame, snapshot.captured_monotonic, frozenset({"face"}))
            if probe.measurements.face.runs > previous_runs:
                attempts += probe.measurements.face.runs - previous_runs
                feature = probe.latest_attention_features
                if (
                    collect
                    and feature is not None
                    and feature.observable
                    and probe.latest_quality is not None
                    and probe.latest_quality.observable
                ):
                    samples.append(feature)
                previous_runs = probe.measurements.face.runs
        cv2.imshow(window, canvas)
        _check_cancel(cv2.waitKey(1))
    return samples, attempts


def _put_text(frame: Any, message: str) -> None:
    import cv2

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (20, 20, 20), -1)
    cv2.putText(
        frame,
        message,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _check_cancel(key: int) -> None:
    if key & 0xFF == 27:
        raise CalibrationCancelled("calibration cancelled by user")


def _save_derived_samples(
    profile_path: str | Path,
    profile: CalibrationProfile,
    posture_samples: list[FeatureVector],
    posture_retest_samples: list[FeatureVector],
    attention_samples: list[AttentionSample],
    attention_validation_samples: list[AttentionSample],
) -> Path:
    """Save compact derived features, never frames or raw landmarks, for complexity audits."""

    target = Path(profile_path).with_suffix(".samples.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "profile_id": profile.profile_id,
        "feature_version": profile.posture.feature_version,
        "posture": [dict(sample.values) for sample in posture_samples],
        "posture_retest": [dict(sample.values) for sample in posture_retest_samples],
        "attention": [
            {
                "point_id": sample.point_id,
                "target_x": sample.target_x,
                "target_y": sample.target_y,
                "features": dict(sample.features.values),
            }
            for sample in attention_samples
        ],
        "attention_validation": [
            {
                "point_id": sample.point_id,
                "target_x": sample.target_x,
                "target_y": sample.target_y,
                "features": dict(sample.features.values),
            }
            for sample in attention_validation_samples
        ],
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(target)
    return target
