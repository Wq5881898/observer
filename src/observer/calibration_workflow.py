from __future__ import annotations

import ctypes
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from observer.calibration import (
    ATTENTION_POINTS_9,
    AttentionCalibration,
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
    posture_sample_seconds: float = 30.0
    posture_retest_seconds: float = 10.0
    attention_settle_seconds: float = 1.0
    attention_sample_seconds: float = 1.5
    attention_validation_settle_seconds: float = 1.0
    attention_validation_sample_seconds: float = 1.5


ATTENTION_EDGE_POINTS_8: tuple[tuple[str, float, float], ...] = (
    ("edge_top_left", 0.25, 0.08),
    ("edge_top_right", 0.75, 0.08),
    ("edge_bottom_left", 0.25, 0.92),
    ("edge_bottom_right", 0.75, 0.92),
    ("edge_left_top", 0.08, 0.25),
    ("edge_left_bottom", 0.08, 0.75),
    ("edge_right_top", 0.92, 0.25),
    ("edge_right_bottom", 0.92, 0.75),
)

ATTENTION_CALIBRATION_POINTS = ATTENTION_POINTS_9 + ATTENTION_EDGE_POINTS_8

ATTENTION_VALIDATION_POINTS: tuple[tuple[str, float, float], ...] = (
    ("validate_inner_top_left", 0.25, 0.25),
    ("validate_inner_top", 0.50, 0.25),
    ("validate_inner_top_right", 0.75, 0.25),
    ("validate_inner_left", 0.25, 0.50),
    ("validate_inner_right", 0.75, 0.50),
    ("validate_inner_bottom_left", 0.25, 0.75),
    ("validate_inner_bottom", 0.50, 0.75),
    ("validate_inner_bottom_right", 0.75, 0.75),
)

MAX_ATTENTION_VALIDATION_RMSE = 0.18
MINIMUM_ATTENTION_SAMPLES_PER_POINT = 7


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
        _await_calibration_ready(camera, window)

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
        if len(posture_samples) < 75:
            raise RuntimeError(
                "posture calibration collected "
                f"{len(posture_samples)}/{posture_attempts} valid samples; "
                "75 are required. Keep both ears and shoulders fully visible."
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
        for point_id, x, y in ATTENTION_CALIBRATION_POINTS:
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
        if len(attention_points) != len(ATTENTION_CALIBRATION_POINTS):
            raise RuntimeError("one or more attention calibration points had no valid samples")
        if len(validation_points) != len(ATTENTION_VALIDATION_POINTS):
            raise RuntimeError("one or more independent attention validation points failed")
        _require_attention_point_sample_counts(
            attention_raw,
            ATTENTION_CALIBRATION_POINTS,
            phase="calibration",
        )
        _require_attention_point_sample_counts(
            validation_raw,
            ATTENTION_VALIDATION_POINTS,
            phase="validation",
        )
        attention = fit_attention_calibration(
            attention_points,
            validation_samples=validation_points,
            planned_sample_count=attention_attempts,
            valid_frame_count=len(attention_raw),
        )
        valid_attention_frames = len(attention_raw)
        profile = create_profile(environment, posture, attention)
        failure_reason: str | None = None
        if valid_attention_frames / max(attention_attempts, 1) < 0.90:
            failure_reason = "attention calibration coverage is below 90%"
        elif (
            attention.validation_rmse is None
            or attention.validation_rmse > MAX_ATTENTION_VALIDATION_RMSE
        ):
            failure_reason = (
                "attention validation RMSE "
                f"{attention.validation_rmse!r} exceeds {MAX_ATTENTION_VALIDATION_RMSE}"
            )
        if failure_reason is not None:
            point_rmse = _attention_validation_rmse_by_point(
                attention, validation_raw
            )
            diagnostics_path = _save_rejected_derived_samples(
                output_path,
                profile,
                posture_samples,
                retest_samples,
                attention_raw,
                validation_raw,
                failure_reason=failure_reason,
                attention_validation_rmse_by_point=point_rmse,
            )
            worst_points = ", ".join(
                f"{point_id}={error:.3f}"
                for point_id, error in sorted(
                    point_rmse.items(), key=lambda item: item[1], reverse=True
                )[:3]
            )
            raise RuntimeError(
                f"{failure_reason}; worst points: {worst_points}; "
                f"derived diagnostics saved to {diagnostics_path}"
            )

        save_profile(profile, output_path)
        _save_derived_samples(
            output_path,
            profile,
            posture_samples,
            retest_samples,
            attention_raw,
            validation_raw,
            attention_validation_rmse_by_point=(
                _attention_validation_rmse_by_point(attention, validation_raw)
            ),
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


def _await_calibration_ready(camera: CameraWorker, window: str) -> None:
    import cv2

    clicked = False

    def on_mouse(event: int, _x: int, _y: int, _flags: int, _parameter: Any) -> None:
        nonlocal clicked
        if event == cv2.EVENT_LBUTTONUP:
            clicked = True

    cv2.setMouseCallback(window, on_mouse)
    if hasattr(cv2, "WND_PROP_TOPMOST"):
        cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
    last_sequence = 0
    try:
        while True:
            snapshot = camera.buffer.latest(after_sequence=last_sequence)
            if snapshot is None:
                _check_cancel(cv2.waitKeyEx(10))
                continue
            last_sequence = snapshot.sequence
            preview = cv2.flip(snapshot.frame, 1)
            _put_text(preview, "Click preview or press SPACE to start; ESC cancels")
            cv2.imshow(window, preview)
            key = cv2.waitKeyEx(1)
            _check_cancel(key)
            if clicked or key & 0xFF == 32:
                return
    finally:
        if hasattr(cv2, "WND_PROP_TOPMOST"):
            cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 0)


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
    previous_pose_runs = probe.measurements.pose.runs
    rejection_reasons: dict[str, int] = {}
    while time.monotonic() - started < duration_seconds:
        snapshot = camera.buffer.latest(after_sequence=last_sequence)
        if snapshot is None:
            _check_cancel(cv2.waitKey(1))
            continue
        last_sequence = snapshot.sequence
        probe.process_due(
            snapshot.frame, snapshot.captured_monotonic, frozenset({"face", "pose"})
        )
        if probe.measurements.pose.runs > previous_pose_runs:
            checked += 1
            accepted = (
                probe.latest_quality is not None
                and probe.latest_quality.observable
                and probe.latest_attention_features is not None
                and probe.latest_attention_features.observable
                and probe.latest_posture_features is not None
                and probe.latest_posture_features.observable
            )
            observable += int(accepted)
            if not accepted:
                current_reasons: list[str] = []
                if probe.latest_quality is None:
                    current_reasons.append("no quality result")
                else:
                    current_reasons.extend(probe.latest_quality.reasons)
                if probe.latest_attention_features is None:
                    current_reasons.append("no observable face/eye features")
                else:
                    current_reasons.extend(probe.latest_attention_features.reasons)
                if probe.latest_posture_features is None:
                    current_reasons.append("no observable shoulder/posture features")
                else:
                    current_reasons.extend(probe.latest_posture_features.reasons)
                for reason in dict.fromkeys(current_reasons):
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            previous_pose_runs = probe.measurements.pose.runs
        preview = cv2.flip(snapshot.frame, 1)
        remaining = max(0.0, duration_seconds - (time.monotonic() - started))
        _put_text(preview, f"{message}  {remaining:0.1f}s")
        cv2.imshow(window, preview)
        _check_cancel(cv2.waitKey(1))
    if require_quality_coverage is not None:
        coverage = observable / max(checked, 1)
        if coverage < require_quality_coverage:
            reason = "; ".join(
                f"{name} ({count}/{checked})"
                for name, count in sorted(
                    rejection_reasons.items(), key=lambda item: item[1], reverse=True
                )[:3]
            ) or "required landmarks unavailable"
            raise RuntimeError(
                f"precheck quality coverage {coverage:.1%} is below "
                f"{require_quality_coverage:.0%} across {checked} model observations: "
                f"{reason}"
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
    point_color = (40, 175, 40) if collect else (30, 80, 220)
    phase_label = "RECORDING - keep looking" if collect else "READY - move eyes to dot"
    cv2.circle(canvas, point, 18, point_color, -1)
    cv2.circle(canvas, point, 5, (255, 255, 255), -1)
    _put_text(canvas, f"{phase_label}: {point_id}  ESC cancels")
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
    *,
    status: str = "accepted",
    failure_reason: str | None = None,
    attention_validation_rmse_by_point: dict[str, float] | None = None,
) -> Path:
    """Save compact derived features, never frames or raw landmarks, for complexity audits."""

    target = Path(profile_path).with_suffix(".samples.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": status,
        "failure_reason": failure_reason,
        "profile_id": profile.profile_id,
        "feature_version": profile.posture.feature_version,
        "attention_training_rmse": profile.attention.training_rmse,
        "attention_validation_rmse": profile.attention.validation_rmse,
        "attention_feature_names": list(profile.attention.feature_names),
        "attention_validation_rmse_by_point": (
            attention_validation_rmse_by_point or {}
        ),
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


def _attention_validation_rmse_by_point(
    calibration: AttentionCalibration,
    samples: list[AttentionSample],
) -> dict[str, float]:
    squared_errors: dict[str, list[float]] = {}
    for sample in aggregate_attention_samples(samples):
        if not sample.features.observable:
            continue
        predicted_x, predicted_y = calibration.predict(sample.features)
        squared_errors.setdefault(sample.point_id, []).append(
            (predicted_x - sample.target_x) ** 2
            + (predicted_y - sample.target_y) ** 2
        )
    return {
        point_id: float(np.sqrt(sum(errors) / (2 * len(errors))))
        for point_id, errors in squared_errors.items()
        if errors
    }


def _require_attention_point_sample_counts(
    samples: list[AttentionSample],
    expected_points: tuple[tuple[str, float, float], ...],
    *,
    phase: str,
) -> None:
    counts: dict[str, int] = {}
    for sample in samples:
        if sample.features.observable:
            counts[sample.point_id] = counts.get(sample.point_id, 0) + 1
    insufficient = {
        point_id: counts.get(point_id, 0)
        for point_id, _x, _y in expected_points
        if counts.get(point_id, 0) < MINIMUM_ATTENTION_SAMPLES_PER_POINT
    }
    if insufficient:
        detail = ", ".join(
            f"{point_id}={count}" for point_id, count in insufficient.items()
        )
        raise RuntimeError(
            f"attention {phase} needs at least "
            f"{MINIMUM_ATTENTION_SAMPLES_PER_POINT} valid samples per point: {detail}"
        )


def _save_rejected_derived_samples(
    profile_path: str | Path,
    profile: CalibrationProfile,
    posture_samples: list[FeatureVector],
    posture_retest_samples: list[FeatureVector],
    attention_samples: list[AttentionSample],
    attention_validation_samples: list[AttentionSample],
    *,
    failure_reason: str,
    attention_validation_rmse_by_point: dict[str, float],
) -> Path:
    active_path = Path(profile_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rejected_base = (
        active_path.parent
        / "rejected"
        / f"{active_path.stem}-{timestamp}{active_path.suffix}"
    )
    return _save_derived_samples(
        rejected_base,
        profile,
        posture_samples,
        posture_retest_samples,
        attention_samples,
        attention_validation_samples,
        status="rejected",
        failure_reason=failure_reason,
        attention_validation_rmse_by_point=attention_validation_rmse_by_point,
    )
