from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from statistics import median
from typing import Any

from observer.calibration import CalibrationProfile, load_profile
from observer.camera import CameraWorker
from observer.config import AppConfig
from observer.models import ModelFiles
from observer.observations import SignalState
from observer.perception import FeatureVector
from observer.vision import MediapipeVisionProbe


FACE_CONTEXT_FEATURES = ("face_center_x", "face_center_y", "face_scale")


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


@dataclass(slots=True)
class AlignmentStability:
    required_seconds: float = 3.0
    dropout_grace_seconds: float = 0.5
    aligned_since: float | None = None
    indeterminate_since: float | None = None

    def __post_init__(self) -> None:
        if self.required_seconds <= 0 or self.dropout_grace_seconds < 0:
            raise ValueError("required alignment duration must be positive")

    def update(self, now_monotonic: float, aligned: bool | None) -> tuple[float, bool]:
        if aligned is False:
            self.aligned_since = None
            self.indeterminate_since = None
            return 0.0, False
        if aligned is None:
            if self.aligned_since is None:
                return 0.0, False
            if self.indeterminate_since is None:
                self.indeterminate_since = now_monotonic
            if now_monotonic - self.indeterminate_since > self.dropout_grace_seconds:
                self.aligned_since = None
                self.indeterminate_since = None
                return 0.0, False
            return max(0.0, now_monotonic - self.aligned_since), False
        self.indeterminate_since = None
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
    frames_saved: int = 0
    verification: dict[str, Any] = field(default_factory=dict)


def assess_alignment(
    profile: CalibrationProfile,
    attention_features: FeatureVector | None,
    posture_features: FeatureVector | None,
    *,
    require_posture: bool = True,
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
    context_z = max(
        abs(current[name] - targets[name]) / scales[name]
        for name in FACE_CONTEXT_FEATURES
    )
    context_aligned = context_z <= attention.maximum_context_z
    posture_aligned = posture_state is SignalState.COMPLIANT
    position_aligned = all(
        abs(current[name] - targets[name])
        <= scales[name] * attention.maximum_context_z
        for name in ("face_center_x", "face_center_y")
    )
    scale_aligned = (
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
        context_aligned and (posture_aligned or not require_posture),
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
    previous_face_runs = probe.measurements.face.runs
    previous_pose_runs = probe.measurements.pose.runs
    attention_feature_counts: dict[str, int] = {}
    posture_feature_counts: dict[str, int] = {}
    quality_counts: dict[str, int] = {}
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
        )
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
            return AlignmentResult(
                profile_id=profile.profile_id,
                stable_seconds=stable_seconds,
                elapsed_seconds=round(now - started, 3),
                final_context_z=round(final_assessment.context_z, 3),
                posture_state=final_assessment.posture_state.value,
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
    lines = [
        "Look at the yellow center dot; move the face cross into the green box",
        assessment.reason,
    ]
    if assessment.context_z is not None:
        lines.append(
            f"face context z={assessment.context_z:.1f} "
            f"(need <= 5.0)  face size={assessment.current_scale:.3f}/"
            f"{assessment.target_scale:.3f}  posture={assessment.posture_state.value}"
        )
        size_delta = assessment.current_scale - assessment.target_scale
        if abs(size_delta) <= (assessment.tolerance_scale or 0.0):
            size_status = "SIZE OK"
        elif size_delta > 0:
            size_status = "TOO CLOSE - MOVE FARTHER"
        else:
            size_status = "TOO FAR - MOVE CLOSER"
        lines.append(
            f"POSITION={'OK' if position_aligned else 'ADJUST'}   {size_status}"
        )
    else:
        lines.append(f"warming up face smoothing  posture={assessment.posture_state.value}")
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
