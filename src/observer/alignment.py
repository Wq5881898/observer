from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from statistics import median
from typing import Any, Callable

from observer.calibration import CalibrationProfile, load_profile
from observer.camera import CameraWorker
from observer.config import AppConfig
from observer.models import ModelFiles
from observer.observations import SignalState
from observer.perception import FeatureVector
from observer.vision import MediapipeVisionProbe


FACE_CONTEXT_FEATURES = ("face_center_x", "face_center_y", "face_scale")
FACE_ALIGNMENT_TARGET_X = 0.5
FACE_ALIGNMENT_MIN_HORIZONTAL_TOLERANCE = 0.05
FACE_ALIGNMENT_MIN_VERTICAL_TOLERANCE = 0.06


class AlignmentCancelled(RuntimeError):
    pass


class AlignmentTimeout(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        last_assessment: "AlignmentAssessment | None",
        best_assessment: "AlignmentAssessment | None",
        longest_stable_seconds: float,
        diagnostics: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.last_assessment = last_assessment
        self.best_assessment = best_assessment
        self.longest_stable_seconds = longest_stable_seconds
        self.diagnostics = diagnostics


@dataclass(frozen=True, slots=True)
class AlignmentAssessment:
    aligned: bool
    reason: str
    posture_state: SignalState
    context_z: float | None
    target_x: float | None
    target_y: float | None
    target_scale: float | None
    current_x: float | None
    current_y: float | None
    current_scale: float | None
    tolerance_x: float | None
    tolerance_y: float | None
    tolerance_scale: float | None
    scale_required: bool = True


@dataclass(slots=True)
class AlignmentStability:
    required_seconds: float = 3.0
    dropout_grace_seconds: float = 0.5
    misalignment_grace_seconds: float = 0.5
    aligned_since: float | None = None
    interruption_since: float | None = None
    interruption_indeterminate: bool = False

    def __post_init__(self) -> None:
        if (
            self.required_seconds <= 0
            or self.dropout_grace_seconds < 0
            or self.misalignment_grace_seconds < 0
        ):
            raise ValueError("required alignment duration must be positive")

    def update(self, now_monotonic: float, aligned: bool | None) -> tuple[float, bool]:
        if aligned is not True:
            if self.aligned_since is None:
                return 0.0, False
            if self.interruption_since is None:
                self.interruption_since = now_monotonic
                self.interruption_indeterminate = aligned is None
            grace = (
                self.dropout_grace_seconds
                if self.interruption_indeterminate
                else self.misalignment_grace_seconds
            )
            held = max(0.0, self.interruption_since - self.aligned_since)
            if now_monotonic - self.interruption_since > grace:
                self.aligned_since = None
                self.interruption_since = None
                return 0.0, False
            # Brief landmark jitter pauses progress instead of advancing or
            # immediately erasing an otherwise stable alignment.
            return held, False
        if self.interruption_since is not None and self.aligned_since is not None:
            self.aligned_since += now_monotonic - self.interruption_since
        self.interruption_since = None
        if self.aligned_since is None:
            self.aligned_since = now_monotonic
        elapsed = max(0.0, now_monotonic - self.aligned_since)
        return elapsed, elapsed >= self.required_seconds


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    profile_id: str
    stable_seconds: float
    elapsed_seconds: float
    final_context_z: float
    posture_state: str
    attention_session_baseline: dict[str, float] = field(default_factory=dict)
    session_face_scale: float | None = None
    frames_saved: int = 0
    verification: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AttentionBaselineResult:
    elapsed_seconds: float
    stable_seconds: float
    observations: int
    values: dict[str, float]


def attention_session_baseline_from_samples(
    samples: list[FeatureVector],
) -> dict[str, float]:
    """Return a robust neutral head-pose baseline from valid live samples."""

    names = ("head_yaw_proxy", "head_pitch_proxy")
    values = {
        name: [sample.values[name] for sample in samples if name in sample.values]
        for name in names
    }
    if not samples or any(not entries for entries in values.values()):
        raise ValueError("attention baseline samples are incomplete")
    return {name: median(entries) for name, entries in values.items()}


def assess_alignment(
    profile: CalibrationProfile,
    attention_features: FeatureVector | None,
    posture_features: FeatureVector | None,
    *,
    require_posture: bool = True,
    require_face_scale: bool = True,
) -> AlignmentAssessment:
    attention = profile.attention
    targets: dict[str, float] = {}
    scales: dict[str, float] = {}
    context_statistics = attention.context_statistics()
    for name in FACE_CONTEXT_FEATURES:
        if name not in context_statistics:
            return _unavailable_assessment(
                f"calibration profile has no {name}", posture_state=SignalState.UNKNOWN
            )
        targets[name], scales[name] = context_statistics[name]

    # The historical calibration x-position describes where the user happened
    # to sit during calibration; it is not a screen-gaze anchor.  Requiring that
    # narrow, possibly off-centre position made later sessions unnecessarily
    # difficult.  Keep the calibrated vertical/scale context, but centre the
    # horizontal guide and allow natural face-position bands.  The vertical
    # minimum prevents ordinary changes in chair height or viewing distance
    # from blocking a session when the full face and eyes remain observable.
    targets["face_center_x"] = FACE_ALIGNMENT_TARGET_X
    scales["face_center_x"] = max(
        scales["face_center_x"],
        FACE_ALIGNMENT_MIN_HORIZONTAL_TOLERANCE / attention.maximum_context_z,
    )
    scales["face_center_y"] = max(
        scales["face_center_y"],
        FACE_ALIGNMENT_MIN_VERTICAL_TOLERANCE / attention.maximum_context_z,
    )

    posture_signal = (
        profile.posture.evaluate(posture_features)
        if posture_features is not None
        else None
    )
    posture_state = posture_signal.state if posture_signal is not None else SignalState.UNKNOWN
    shared = {
        "posture_state": posture_state,
        "target_x": targets["face_center_x"],
        "target_y": targets["face_center_y"],
        "target_scale": targets["face_scale"],
        "tolerance_x": scales["face_center_x"] * attention.maximum_context_z,
        "tolerance_y": scales["face_center_y"] * attention.maximum_context_z,
        "tolerance_scale": scales["face_scale"] * attention.maximum_context_z,
        "scale_required": require_face_scale,
    }
    if attention_features is None:
        return AlignmentAssessment(
            False, "waiting for face and eye features", context_z=None,
            current_x=None, current_y=None, current_scale=None, **shared
        )
    if attention_features.kind != "attention" or (
        attention_features.version != attention.feature_version
    ):
        return AlignmentAssessment(
            False, "attention feature version mismatch", context_z=None,
            current_x=None, current_y=None, current_scale=None, **shared
        )
    if not attention_features.observable:
        return AlignmentAssessment(
            False, "; ".join(attention_features.reasons), context_z=None,
            current_x=None, current_y=None, current_scale=None, **shared
        )
    missing = [name for name in FACE_CONTEXT_FEATURES if name not in attention_features.values]
    if missing:
        return AlignmentAssessment(
            False, f"attention features are missing: {missing}", context_z=None,
            current_x=None, current_y=None, current_scale=None, **shared
        )

    current = {name: attention_features.values[name] for name in FACE_CONTEXT_FEATURES}
    gated_context_features = ["face_center_x", "face_center_y"]
    if require_face_scale:
        gated_context_features.append("face_scale")
    context_z = max(
        abs(current[name] - targets[name]) / scales[name]
        for name in gated_context_features
    )
    posture_aligned = posture_state is SignalState.COMPLIANT
    position_aligned = all(
        abs(current[name] - targets[name])
        <= scales[name] * attention.maximum_context_z
        for name in ("face_center_x", "face_center_y")
    )
    scale_aligned = not require_face_scale or (
        abs(current["face_scale"] - targets["face_scale"])
        <= scales["face_scale"] * attention.maximum_context_z
    )
    if not position_aligned:
        reason = "move face cross into the green position box"
    elif not scale_aligned and current["face_scale"] > targets["face_scale"]:
        reason = "position matched; move slightly farther from the camera"
    elif not scale_aligned:
        reason = "position matched; move slightly closer to the camera"
    elif not require_posture and posture_state is not SignalState.COMPLIANT:
        reason = f"face aligned; posture={posture_state.value} is recorded but not gating"
    elif posture_signal is None:
        reason = "waiting for posture features"
    elif posture_state is SignalState.UNKNOWN:
        reason = posture_signal.reason or "posture is not observable"
    elif posture_state is SignalState.VIOLATION:
        reason = "return to the calibrated neutral posture"
    else:
        reason = "aligned - hold still and look at the center dot"
    return AlignmentAssessment(
        position_aligned and scale_aligned and (posture_aligned or not require_posture),
        reason,
        context_z=context_z,
        current_x=current["face_center_x"],
        current_y=current["face_center_y"],
        current_scale=current["face_scale"],
        **shared,
    )


def run_alignment(
    config: AppConfig,
    models: ModelFiles,
    profile_path: str | Path,
    *,
    stable_seconds: float = 3.0,
    timeout_seconds: float = 120.0,
    verify_seconds: float = 0.0,
    require_posture: bool = False,
) -> AlignmentResult:
    """Guide the user back to calibration pose without retaining any frames."""

    if timeout_seconds <= 0:
        raise ValueError("alignment timeout must be positive")
    if verify_seconds < 0:
        raise ValueError("alignment verification duration cannot be negative")
    import cv2

    # Kept local to avoid coupling the pure assessment helpers to Windows display APIs.
    from observer.calibration_workflow import current_environment

    camera = CameraWorker(config.camera)
    probe: MediapipeVisionProbe | None = None
    window = "Observer alignment - no recording"
    try:
        if not camera.start():
            raise RuntimeError(camera.stats.last_error or "camera could not be opened")
        first = _wait_for_frame(camera, 5.0)
        profile = load_profile(profile_path, current_environment(config, first.frame.shape))
        probe = MediapipeVisionProbe(models, config.inference, profile)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        if hasattr(cv2, "WND_PROP_TOPMOST"):
            cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
        result = guide_alignment(
            camera,
            probe,
            profile,
            window,
            stable_seconds=stable_seconds,
            timeout_seconds=timeout_seconds,
            require_posture=require_posture,
        )
        if verify_seconds > 0:
            result = replace(
                result,
                verification=verify_aligned_baseline(
                    camera, probe, window, seconds=verify_seconds
                ),
            )
        return result
    finally:
        if probe is not None:
            probe.close()
        camera.stop()
        cv2.destroyAllWindows()


def guide_alignment(
    camera: CameraWorker,
    probe: MediapipeVisionProbe,
    profile: CalibrationProfile,
    window: str,
    *,
    stable_seconds: float = 3.0,
    timeout_seconds: float = 120.0,
    require_posture: bool = False,
    require_face_scale: bool = True,
) -> AlignmentResult:
    """Run the alignment gate inside an already-open camera and probe session."""

    if timeout_seconds <= 0:
        raise ValueError("alignment timeout must be positive")
    import cv2

    started = time.monotonic()
    final_assessment: AlignmentAssessment | None = None
    best_assessment: AlignmentAssessment | None = None
    longest_stable_seconds = 0.0
    stability = AlignmentStability(stable_seconds)
    last_sequence = 0
    initial_face_runs = probe.measurements.face.runs
    previous_face_runs = initial_face_runs
    previous_pose_runs = probe.measurements.pose.runs
    attention_feature_counts: dict[str, int] = {}
    posture_feature_counts: dict[str, int] = {}
    quality_counts: dict[str, int] = {}
    alignment_reason_counts: dict[str, int] = {}
    posture_state_counts: dict[str, int] = {}
    posture_metric_values: dict[str, list[float]] = {}
    baseline_values: dict[str, list[float]] = {
        "head_yaw_proxy": [],
        "head_pitch_proxy": [],
    }
    face_scale_values: list[float] = []
    while True:
        now = time.monotonic()
        if now - started >= timeout_seconds:
            raise AlignmentTimeout(
                f"alignment did not remain stable for {stable_seconds:g}s "
                f"within {timeout_seconds:g}s",
                last_assessment=final_assessment,
                best_assessment=best_assessment,
                longest_stable_seconds=longest_stable_seconds,
                diagnostics={
                    "camera": asdict(camera.stats),
                    "vision": probe.summary(),
                    "quality_observations": quality_counts,
                    "attention_feature_observations": attention_feature_counts,
                    "posture_feature_observations": posture_feature_counts,
                    "alignment_reasons": alignment_reason_counts,
                    "posture_states": posture_state_counts,
                    "posture_metrics": {
                        name: {
                            "minimum": min(values),
                            "median": median(values),
                            "maximum": max(values),
                        }
                        for name, values in posture_metric_values.items()
                        if values
                    },
                    "frames_saved": 0,
                },
            )
        snapshot = camera.buffer.latest(after_sequence=last_sequence)
        if snapshot is None:
            _check_cancel(cv2.waitKeyEx(10))
            continue
        last_sequence = snapshot.sequence
        probe.process_due(
            snapshot.frame,
            snapshot.captured_monotonic,
            frozenset({"face", "pose"}),
        )
        if probe.measurements.face.runs > previous_face_runs:
            count = probe.measurements.face.runs - previous_face_runs
            _count_feature_observation(
                attention_feature_counts, probe.latest_attention_features, count
            )
            previous_face_runs = probe.measurements.face.runs
        if probe.measurements.pose.runs > previous_pose_runs:
            count = probe.measurements.pose.runs - previous_pose_runs
            _count_feature_observation(
                posture_feature_counts, probe.latest_posture_features, count
            )
            previous_pose_runs = probe.measurements.pose.runs
        if probe.latest_quality is not None:
            quality_key = (
                "observable"
                if probe.latest_quality.observable
                else "; ".join(probe.latest_quality.reasons) or "rejected"
            )
            quality_counts[quality_key] = quality_counts.get(quality_key, 0) + 1
        final_assessment = assess_alignment(
            profile,
            probe.latest_smoothed_attention_features,
            probe.latest_posture_features,
            require_posture=require_posture,
            require_face_scale=require_face_scale,
        )
        alignment_reason_counts[final_assessment.reason] = (
            alignment_reason_counts.get(final_assessment.reason, 0) + 1
        )
        posture_key = final_assessment.posture_state.value
        posture_state_counts[posture_key] = posture_state_counts.get(posture_key, 0) + 1
        if probe.latest_posture_features is not None:
            posture_signal = profile.posture.evaluate(probe.latest_posture_features)
            for name, value in posture_signal.metrics.items():
                posture_metric_values.setdefault(name, []).append(float(value))
        indeterminate = (
            final_assessment.context_z is None
            or (
                require_posture
                and final_assessment.posture_state is SignalState.UNKNOWN
            )
        )
        held_seconds, complete = stability.update(
            now, None if indeterminate else final_assessment.aligned
        )
        if final_assessment.aligned:
            features = probe.latest_smoothed_attention_features
            if features is not None and {
                "head_yaw_proxy",
                "head_pitch_proxy",
            }.issubset(features.values):
                for name in baseline_values:
                    baseline_values[name].append(features.values[name])
                if "face_scale" in features.values:
                    face_scale_values.append(features.values["face_scale"])
        elif held_seconds == 0.0:
            for values in baseline_values.values():
                values.clear()
            face_scale_values.clear()
        longest_stable_seconds = max(longest_stable_seconds, held_seconds)
        if final_assessment.context_z is not None and (
            best_assessment is None
            or best_assessment.context_z is None
            or final_assessment.context_z < best_assessment.context_z
        ):
            best_assessment = final_assessment
        preview = cv2.flip(snapshot.frame, 1)
        _draw_alignment_overlay(
            preview,
            final_assessment,
            held_seconds=held_seconds,
            required_seconds=stable_seconds,
        )
        cv2.imshow(window, preview)
        _check_cancel(cv2.waitKeyEx(1))
        if complete and final_assessment.context_z is not None:
            if not all(baseline_values.values()):
                raise RuntimeError("aligned session did not capture a head-pose baseline")
            baseline = {
                name: median(values) for name, values in baseline_values.items()
            }
            probe.set_attention_session_baseline(
                FeatureVector("attention", baseline)
            )
            return AlignmentResult(
                profile_id=profile.profile_id,
                stable_seconds=stable_seconds,
                elapsed_seconds=round(now - started, 3),
                final_context_z=round(final_assessment.context_z, 3),
                posture_state=final_assessment.posture_state.value,
                attention_session_baseline=baseline,
                session_face_scale=(
                    median(face_scale_values) if face_scale_values else None
                ),
            )


def guide_attention_session_baseline(
    camera: CameraWorker,
    probe: MediapipeVisionProbe,
    *,
    stable_seconds: float = 3.0,
    timeout_seconds: float = 60.0,
    show_preview: bool = True,
    progress: Callable[[float, float, str | None], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> AttentionBaselineResult:
    """Capture neutral head pose at the user's natural working distance.

    This setup deliberately does not enforce the calibration-time face scale or
    shoulder visibility. It only requires a usable face/eye observation while
    the user follows the instruction to look at the screen centre.
    """

    if stable_seconds <= 0 or timeout_seconds <= 0:
        raise ValueError("attention baseline durations must be positive")
    if cancelled and cancelled():
        raise AlignmentCancelled("session setup cancelled")
    cv2 = None
    window = "Observer session setup - no recording"
    if show_preview:
        import cv2 as cv2_module
        cv2 = cv2_module
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        if hasattr(cv2, "WND_PROP_TOPMOST"):
            cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)

    started = time.monotonic()
    valid_since: float | None = None
    samples: list[FeatureVector] = []
    last_sequence = 0
    initial_face_runs = probe.measurements.face.runs
    previous_face_runs = initial_face_runs
    valid_frame_count = 0
    invalid_reason_counts: dict[str, int] = {}
    longest_valid_seconds = 0.0
    last_invalid_reason = "no camera frame"
    last_progress_at = 0.0
    try:
        while True:
            now = time.monotonic()
            if cancelled and cancelled():
                raise AlignmentCancelled("session setup cancelled")
            if now - started >= timeout_seconds:
                raise RuntimeError(
                    _attention_setup_timeout_message(
                        face_runs=probe.measurements.face.runs - initial_face_runs,
                        valid_frames=valid_frame_count,
                        longest_valid_seconds=longest_valid_seconds,
                        invalid_reason_counts=invalid_reason_counts,
                    )
                )
            snapshot = camera.buffer.latest(after_sequence=last_sequence)
            if snapshot is None:
                if cv2 is not None:
                    _check_cancel(cv2.waitKeyEx(10))
                else:
                    time.sleep(.01)
                continue
            last_sequence = snapshot.sequence
            probe.process_due(
                snapshot.frame,
                snapshot.captured_monotonic,
                frozenset({"face"}),
            )
            feature = probe.latest_smoothed_attention_features
            quality = probe.latest_quality
            invalid_reasons: list[str] = []
            if feature is None:
                invalid_reasons.append("face_or_eyes_not_observable")
            else:
                invalid_reasons.extend(f"feature:{reason}" for reason in feature.reasons)
                if not {
                    "head_yaw_proxy",
                    "head_pitch_proxy",
                }.issubset(feature.values):
                    invalid_reasons.append("head_pose_unavailable")
            if quality is None:
                invalid_reasons.append("quality_unavailable")
            elif not quality.observable:
                invalid_reasons.extend(
                    f"quality:{reason}" for reason in quality.reasons
                )
            valid = (
                not invalid_reasons
            )
            if valid:
                valid_frame_count += 1
                if valid_since is None:
                    valid_since = now
                    samples.clear()
                if probe.measurements.face.runs > previous_face_runs:
                    samples.append(feature)
                    previous_face_runs = probe.measurements.face.runs
            else:
                last_invalid_reason = invalid_reasons[0]
                for reason in set(invalid_reasons):
                    invalid_reason_counts[reason] = invalid_reason_counts.get(reason, 0) + 1
                valid_since = None
                samples.clear()

            held = 0.0 if valid_since is None else now - valid_since
            longest_valid_seconds = max(longest_valid_seconds, held)
            if progress and (last_progress_at == 0.0 or now-last_progress_at >= .2):
                progress(held, stable_seconds, None if valid else last_invalid_reason)
                last_progress_at = now
            status = (
                f"Hold still {min(held, stable_seconds):.1f}/{stable_seconds:.1f}s"
                if valid
                else f"Not ready: {last_invalid_reason}"
            )
            if cv2 is not None:
                preview = cv2.flip(snapshot.frame, 1)
                cv2.rectangle(preview, (0, 0), (preview.shape[1], 112), (20, 20, 20), -1)
                cv2.putText(
                    preview,
                    "SESSION SETUP - NO RECORDING",
                    (18, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    preview,
                    "Sit naturally. Look at the SCREEN CENTER.",
                    (18, 66),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    preview,
                    status,
                    (18, 98),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (80, 255, 80) if valid else (0, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window, preview)
                _check_cancel(cv2.waitKeyEx(1))
            if held >= stable_seconds and samples:
                baseline = attention_session_baseline_from_samples(samples)
                probe.set_attention_session_baseline(
                    FeatureVector("attention", baseline)
                )
                return AttentionBaselineResult(
                    elapsed_seconds=round(now - started, 3),
                    stable_seconds=stable_seconds,
                    observations=len(samples),
                    values=baseline,
                )
    finally:
        if cv2 is not None:
            try:
                cv2.destroyWindow(window)
            except cv2.error:
                pass


def _attention_setup_timeout_message(
    *,
    face_runs: int,
    valid_frames: int,
    longest_valid_seconds: float,
    invalid_reason_counts: dict[str, int],
) -> str:
    ranked_reasons = sorted(
        invalid_reason_counts.items(), key=lambda item: (-item[1], item[0])
    )
    reason_summary = ", ".join(
        f"{reason}={count}" for reason, count in ranked_reasons[:3]
    ) or "none"
    return (
        "session attention setup timed out; keep the full face visible "
        "and look at the screen centre; "
        f"face_runs={face_runs}, valid_frames={valid_frames}, "
        f"longest_valid={longest_valid_seconds:.2f}s, "
        f"invalid_reasons={reason_summary}"
    )


def verify_aligned_baseline(
    camera: CameraWorker,
    probe: MediapipeVisionProbe,
    window: str,
    *,
    seconds: float,
) -> dict[str, Any]:
    """Measure signals immediately after alignment without closing the camera."""

    if seconds <= 0:
        raise ValueError("verification duration must be positive")
    import cv2

    started = time.monotonic()
    last_sequence = 0
    state_counts: dict[str, dict[str, int]] = {
        "attention_away": {},
        "posture_deviation": {},
    }
    context_values: list[float] = []
    frames_observed = 0
    while time.monotonic() - started < seconds:
        snapshot = camera.buffer.latest(after_sequence=last_sequence)
        if snapshot is None:
            _check_cancel(cv2.waitKeyEx(10))
            continue
        last_sequence = snapshot.sequence
        frames_observed += 1
        probe.process_due(
            snapshot.frame,
            snapshot.captured_monotonic,
            frozenset({"face", "pose"}),
        )
        observation = probe.observation(snapshot.captured_monotonic)
        for name in state_counts:
            signal = observation.signal(name)
            counts = state_counts[name]
            counts[signal.state.value] = counts.get(signal.state.value, 0) + 1
            if name == "attention_away" and "attention_context_z" in signal.metrics:
                context_values.append(signal.metrics["attention_context_z"])

        preview = cv2.flip(snapshot.frame, 1)
        remaining = max(0.0, seconds - (time.monotonic() - started))
        cv2.circle(
            preview,
            (preview.shape[1] // 2, preview.shape[0] // 2),
            11,
            (0, 215, 255),
            -1,
        )
        cv2.rectangle(preview, (0, 0), (preview.shape[1], 74), (20, 20, 20), -1)
        for index, line in enumerate(
            (
                "Same-session baseline: stay normal and look at the center dot",
                f"Verifying {remaining:.1f}s   ESC cancels   no frames saved",
            )
        ):
            cv2.putText(
                preview,
                line,
                (12, 28 + index * 31),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.61,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.imshow(window, preview)
        _check_cancel(cv2.waitKeyEx(1))

    return {
        "duration_seconds": seconds,
        "frames_observed": frames_observed,
        "signal_state_counts": state_counts,
        "attention_context_z_median": (
            round(median(context_values), 3) if context_values else None
        ),
        "attention_context_z_maximum": (
            round(max(context_values), 3) if context_values else None
        ),
        "frames_saved": 0,
    }


def _count_feature_observation(
    counts: dict[str, int], feature: FeatureVector | None, increment: int
) -> None:
    if feature is None:
        key = "missing"
    elif feature.observable:
        key = "observable"
    else:
        key = "; ".join(feature.reasons) or "unobservable"
    counts[key] = counts.get(key, 0) + increment


def _unavailable_assessment(
    reason: str, *, posture_state: SignalState
) -> AlignmentAssessment:
    return AlignmentAssessment(
        aligned=False,
        reason=reason,
        posture_state=posture_state,
        context_z=None,
        target_x=None,
        target_y=None,
        target_scale=None,
        current_x=None,
        current_y=None,
        current_scale=None,
        tolerance_x=None,
        tolerance_y=None,
        tolerance_scale=None,
    )


def _wait_for_frame(camera: CameraWorker, timeout_seconds: float) -> Any:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        snapshot = camera.buffer.latest()
        if snapshot is not None:
            return snapshot
        time.sleep(0.01)
    raise RuntimeError("camera opened but did not produce a frame")


def _draw_alignment_overlay(
    frame: Any,
    assessment: AlignmentAssessment,
    *,
    held_seconds: float,
    required_seconds: float,
) -> None:
    import cv2

    height, width = frame.shape[:2]
    position_aligned = False
    if (
        assessment.current_x is not None
        and assessment.current_y is not None
        and assessment.target_x is not None
        and assessment.target_y is not None
        and assessment.tolerance_x is not None
        and assessment.tolerance_y is not None
    ):
        position_aligned = (
            abs(assessment.current_x - assessment.target_x) <= assessment.tolerance_x
            and abs(assessment.current_y - assessment.target_y) <= assessment.tolerance_y
        )
    position_color = (55, 200, 55) if position_aligned else (40, 90, 230)
    if (
        assessment.target_x is not None
        and assessment.target_y is not None
        and assessment.tolerance_x is not None
        and assessment.tolerance_y is not None
    ):
        # The preview is mirrored, while features and the saved profile are not.
        target_x = (1.0 - assessment.target_x) * width
        target_y = assessment.target_y * height
        half_width = max(8, int(assessment.tolerance_x * width))
        half_height = max(8, int(assessment.tolerance_y * height))
        cv2.rectangle(
            frame,
            (int(target_x) - half_width, int(target_y) - half_height),
            (int(target_x) + half_width, int(target_y) + half_height),
            (55, 200, 55),
            2,
        )
    if assessment.current_x is not None and assessment.current_y is not None:
        current = (
            int((1.0 - assessment.current_x) * width),
            int(assessment.current_y * height),
        )
        cv2.drawMarker(frame, current, position_color, cv2.MARKER_CROSS, 28, 3)

    # The center dot keeps gaze direction consistent while the face cross is aligned.
    cv2.circle(frame, (width // 2, height // 2), 11, (0, 215, 255), -1)
    cv2.circle(frame, (width // 2, height // 2), 3, (255, 255, 255), -1)
    cv2.rectangle(frame, (0, 0), (width, 142), (20, 20, 20), -1)
    lines = ["Align face with the box and look at the yellow dot"]
    if assessment.context_z is not None:
        size_delta = assessment.current_scale - assessment.target_scale
        if not assessment.scale_required:
            size_status = "NATURAL DISTANCE OK"
        elif abs(size_delta) <= (assessment.tolerance_scale or 0.0):
            size_status = "SIZE OK"
        elif size_delta > 0:
            size_status = "TOO CLOSE - MOVE FARTHER"
        else:
            size_status = "TOO FAR - MOVE CLOSER"
        lines.extend(
            (
                f"FACE POSITION: {'PASS' if position_aligned else 'ADJUST'}   "
                f"FACE SIZE: {size_status}",
                f"FACE CONTEXT: {'PASS' if assessment.context_z <= 5.0 else 'ADJUST'} "
                f"(z={assessment.context_z:.1f}, need <=5.0)   "
                f"POSTURE: {assessment.posture_state.value}",
                (
                    "OVERALL: READY - hold steady"
                    if assessment.aligned
                    else f"OVERALL: BLOCKED - {assessment.reason}"
                ),
            )
        )
    else:
        lines.extend(
            (
                "FACE POSITION/SIZE: waiting for stable landmarks",
                f"POSTURE: {assessment.posture_state.value}",
                f"OVERALL: BLOCKED - {assessment.reason}",
            )
        )
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (12, 28 + index * 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.61,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    progress = min(1.0, held_seconds / required_seconds)
    cv2.rectangle(frame, (12, height - 31), (width - 12, height - 12), (55, 55, 55), -1)
    cv2.rectangle(
        frame,
        (12, height - 31),
        (12 + int((width - 24) * progress), height - 12),
        (55, 200, 55),
        -1,
    )
    cv2.putText(
        frame,
        f"Hold aligned {held_seconds:.1f}/{required_seconds:.1f}s   ESC cancels   no frames saved",
        (12, height - 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _check_cancel(key: int) -> None:
    if key & 0xFF == 27:
        raise AlignmentCancelled("alignment cancelled by user")
