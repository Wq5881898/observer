from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from statistics import median
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from observer.observations import PhoneCandidate


FEATURE_VERSION = "observer-c1-v1"


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    brightness_min: float = 35.0
    brightness_max: float = 225.0
    blur_variance_min: float = 25.0
    occlusion_brightness_max: float = 110.0
    occlusion_blur_variance_max: float = 30.0
    landmark_visibility_min: float = 0.55
    eye_openness_min: float = 0.08

    def __post_init__(self) -> None:
        if not 0 <= self.brightness_min < self.brightness_max <= 255:
            raise ValueError("brightness thresholds must satisfy 0 <= min < max <= 255")
        if self.blur_variance_min < 0:
            raise ValueError("blur threshold cannot be negative")
        if not 0 <= self.occlusion_brightness_max <= 255:
            raise ValueError("occlusion brightness threshold must be between 0 and 255")
        if self.occlusion_blur_variance_max < self.blur_variance_min:
            raise ValueError("occlusion blur threshold must not be below blur minimum")
        if not 0 <= self.landmark_visibility_min <= 1:
            raise ValueError("landmark visibility threshold must be between 0 and 1")
        if not 0 <= self.eye_openness_min <= 1:
            raise ValueError("eye openness threshold must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class FrameQuality:
    brightness_mean: float
    blur_variance: float
    observable: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FeatureVector:
    kind: str
    values: Mapping[str, float]
    version: str = FEATURE_VERSION
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        copied = {str(name): float(value) for name, value in self.values.items()}
        if any(not math.isfinite(value) for value in copied.values()):
            raise ValueError("feature values must be finite")
        object.__setattr__(self, "values", MappingProxyType(copied))

    @property
    def observable(self) -> bool:
        return not self.reasons


@dataclass(frozen=True, slots=True)
class PerceptionFeatures:
    frame_quality: FrameQuality
    attention: FeatureVector = field(
        default_factory=lambda: FeatureVector("attention", {}, reasons=("face missing",))
    )
    posture: FeatureVector = field(
        default_factory=lambda: FeatureVector("posture", {}, reasons=("pose missing",))
    )


class FeatureMedianSmoother:
    """Return a robust short-window feature vector without retaining frames."""

    def __init__(
        self,
        kind: str,
        *,
        window_size: int = 7,
        minimum_samples: int | None = None,
        maximum_consecutive_unobservable: int = 3,
    ) -> None:
        selected_minimum = minimum_samples or window_size
        if (
            not kind
            or window_size < 1
            or not 1 <= selected_minimum <= window_size
            or maximum_consecutive_unobservable < 1
        ):
            raise ValueError("feature smoother configuration is invalid")
        self.kind = kind
        self.window_size = window_size
        self.minimum_samples = selected_minimum
        self.maximum_consecutive_unobservable = maximum_consecutive_unobservable
        self._samples: deque[FeatureVector] = deque(maxlen=window_size)
        self._version: str | None = None
        self._consecutive_unobservable = 0

    def reset(self) -> None:
        self._samples.clear()
        self._version = None
        self._consecutive_unobservable = 0

    def update(self, features: FeatureVector) -> FeatureVector:
        if features.kind != self.kind:
            raise ValueError("feature kind does not match smoother")
        if not features.observable:
            self._consecutive_unobservable += 1
            if (
                self._consecutive_unobservable
                >= self.maximum_consecutive_unobservable
            ):
                self._samples.clear()
                self._version = None
            return features
        self._consecutive_unobservable = 0
        if self._version is not None and features.version != self._version:
            self.reset()
        self._version = features.version
        self._samples.append(features)
        if len(self._samples) < self.minimum_samples:
            return FeatureVector(
                self.kind,
                {},
                version=features.version,
                reasons=(
                    f"{self.kind} temporal smoother is warming up "
                    f"({len(self._samples)}/{self.minimum_samples})",
                ),
            )
        names = set.intersection(*(set(item.values) for item in self._samples))
        if not names:
            return FeatureVector(
                self.kind,
                {},
                version=features.version,
                reasons=("temporal smoother has no common features",),
            )
        values = {
            name: float(median(item.values[name] for item in self._samples))
            for name in names
        }
        return FeatureVector(self.kind, values, version=features.version)


class PresenceHitTracker:
    """Require consecutive detector hits before accepting object presence."""

    def __init__(self, confirmation_hits: int = 2) -> None:
        if confirmation_hits < 1:
            raise ValueError("presence confirmation hits must be positive")
        self.confirmation_hits = confirmation_hits
        self._hits = 0

    def update(self, count: int) -> int:
        if count <= 0:
            self._hits = 0
            return 0
        self._hits += 1
        return count if self._hits >= self.confirmation_hits else 0


class PhoneCandidateTracker:
    """Confirm nearby detections across frames and request a short detector burst."""

    def __init__(
        self,
        *,
        confirmation_hits: int = 2,
        maximum_center_distance: float = 0.18,
        retain_seconds: float = 0.45,
        burst_seconds: float = 2.0,
    ) -> None:
        if confirmation_hits < 1 or min(retain_seconds, burst_seconds) < 0:
            raise ValueError("phone tracker timing and hit count must be non-negative")
        self.confirmation_hits = confirmation_hits
        self.maximum_center_distance = maximum_center_distance
        self.retain_seconds = retain_seconds
        self.burst_seconds = burst_seconds
        self._last: PhoneCandidate | None = None
        self._last_seen: float | None = None
        self._hits = 0
        self._burst_until = 0.0

    def update(
        self, at_monotonic: float, candidates: Sequence[PhoneCandidate]
    ) -> tuple[PhoneCandidate, ...]:
        if candidates:
            selected = max(candidates, key=lambda candidate: candidate.confidence)
            if (
                self._last is not None
                and self._center_distance(self._last, selected)
                <= self.maximum_center_distance
            ):
                self._hits += 1
            else:
                self._hits = 1
            self._last = selected
            self._last_seen = at_monotonic
            self._burst_until = max(self._burst_until, at_monotonic + self.burst_seconds)
        elif self._last_seen is None or at_monotonic - self._last_seen > self.retain_seconds:
            self._last = None
            self._last_seen = None
            self._hits = 0

        if self._last is not None and self._hits >= self.confirmation_hits:
            return (self._last,)
        return ()

    def burst_active(self, at_monotonic: float) -> bool:
        return at_monotonic < self._burst_until

    @staticmethod
    def _center_distance(first: PhoneCandidate, second: PhoneCandidate) -> float:
        first_x = first.box.x + first.box.width / 2.0
        first_y = first.box.y + first.box.height / 2.0
        second_x = second.box.x + second.box.width / 2.0
        second_y = second.box.y + second.box.height / 2.0
        return math.hypot(first_x - second_x, first_y - second_y)


def assess_frame_quality(
    bgr_frame: Any, thresholds: QualityThresholds | None = None
) -> FrameQuality:
    """Measure cheap image-level quality without retaining the frame."""

    import cv2
    import numpy as np

    selected = thresholds or QualityThresholds()
    if bgr_frame is None or not hasattr(bgr_frame, "shape") or bgr_frame.size == 0:
        return FrameQuality(0.0, 0.0, False, ("empty frame",))
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    reasons: list[str] = []
    if brightness < selected.brightness_min:
        reasons.append("frame too dark")
    elif brightness > selected.brightness_max:
        reasons.append("frame too bright")
    if blur < selected.blur_variance_min:
        reasons.append("frame too blurry")
    if (
        brightness <= selected.occlusion_brightness_max
        and blur <= selected.occlusion_blur_variance_max
    ):
        reasons.append("camera view may be covered")
    return FrameQuality(brightness, blur, not reasons, tuple(reasons))


def extract_attention_features(
    landmarks: Sequence[Any] | None,
    thresholds: QualityThresholds | None = None,
) -> FeatureVector:
    """Extract lightweight iris, head and face-position features from Face Mesh."""

    selected = thresholds or QualityThresholds()
    if landmarks is None or len(landmarks) < 478:
        return FeatureVector("attention", {}, reasons=("face or iris landmarks missing",))

    def center(indices: Sequence[int]) -> tuple[float, float]:
        return (
            sum(float(landmarks[index].x) for index in indices) / len(indices),
            sum(float(landmarks[index].y) for index in indices) / len(indices),
        )

    points = [(float(point.x), float(point.y)) for point in landmarks]
    face_left = min(point[0] for point in points)
    face_right = max(point[0] for point in points)
    face_top = min(point[1] for point in points)
    face_bottom = max(point[1] for point in points)
    face_width = face_right - face_left
    face_height = face_bottom - face_top
    if face_width <= 1e-6 or face_height <= 1e-6:
        return FeatureVector("attention", {}, reasons=("degenerate face landmarks",))

    iris_a = center((468, 469, 470, 471, 472))
    iris_b = center((473, 474, 475, 476, 477))

    def eye_features(
        iris: tuple[float, float], corners: tuple[int, int], lids: tuple[int, int]
    ) -> tuple[float, float, float]:
        corner_x = sorted((float(landmarks[corners[0]].x), float(landmarks[corners[1]].x)))
        lid_y = sorted((float(landmarks[lids[0]].y), float(landmarks[lids[1]].y)))
        width = max(corner_x[1] - corner_x[0], 1e-6)
        height = max(lid_y[1] - lid_y[0], 1e-6)
        return (
            (iris[0] - corner_x[0]) / width,
            (iris[1] - lid_y[0]) / height,
            height / width,
        )

    eye_a_x, eye_a_y, eye_a_open = eye_features(iris_a, (33, 133), (159, 145))
    eye_b_x, eye_b_y, eye_b_open = eye_features(iris_b, (362, 263), (386, 374))
    eye_mid_x = (iris_a[0] + iris_b[0]) / 2.0
    eye_mid_y = (iris_a[1] + iris_b[1]) / 2.0
    interocular = max(abs(iris_a[0] - iris_b[0]), 1e-6)
    nose_x = float(landmarks[1].x)
    nose_y = float(landmarks[1].y)
    reasons: list[str] = []
    if min(eye_a_open, eye_b_open) < selected.eye_openness_min:
        reasons.append("eyes closed or eye landmarks unreliable")

    return FeatureVector(
        "attention",
        {
            "iris_a_x": eye_a_x,
            "iris_a_y": eye_a_y,
            "iris_b_x": eye_b_x,
            "iris_b_y": eye_b_y,
            "eye_a_open": eye_a_open,
            "eye_b_open": eye_b_open,
            "head_yaw_proxy": (nose_x - eye_mid_x) / interocular,
            "head_pitch_proxy": (nose_y - eye_mid_y) / face_height,
            "face_center_x": (face_left + face_right) / 2.0,
            "face_center_y": (face_top + face_bottom) / 2.0,
            "face_scale": math.sqrt(face_width * face_height),
        },
        reasons=tuple(reasons),
    )


def extract_posture_features(
    pose_landmarks: Sequence[Any] | None,
    face_landmarks: Sequence[Any] | None = None,
    thresholds: QualityThresholds | None = None,
) -> FeatureVector:
    """Extract scale-normalized upper-body features from Pose and optional Face Mesh."""

    selected = thresholds or QualityThresholds()
    if pose_landmarks is None or len(pose_landmarks) < 25:
        return FeatureVector("posture", {}, reasons=("pose landmarks missing",))

    required = (0, 7, 8, 11, 12)
    invisible = [
        index
        for index in required
        if float(getattr(pose_landmarks[index], "visibility", 0.0) or 0.0)
        < selected.landmark_visibility_min
    ]
    if invisible:
        return FeatureVector(
            "posture", {}, reasons=(f"required pose landmarks not visible: {invisible}",)
        )

    left_shoulder = pose_landmarks[11]
    right_shoulder = pose_landmarks[12]
    shoulder_dx = float(right_shoulder.x) - float(left_shoulder.x)
    shoulder_dy = float(right_shoulder.y) - float(left_shoulder.y)
    shoulder_width = math.hypot(shoulder_dx, shoulder_dy)
    if shoulder_width <= 1e-6:
        return FeatureVector("posture", {}, reasons=("shoulder width is degenerate",))

    shoulder_mid_x = (float(left_shoulder.x) + float(right_shoulder.x)) / 2.0
    shoulder_mid_y = (float(left_shoulder.y) + float(right_shoulder.y)) / 2.0
    ear_mid_x = (float(pose_landmarks[7].x) + float(pose_landmarks[8].x)) / 2.0
    ear_mid_y = (float(pose_landmarks[7].y) + float(pose_landmarks[8].y)) / 2.0
    values = {
        "shoulder_slope": shoulder_dy / max(abs(shoulder_dx), 1e-6),
        "head_offset_x": (ear_mid_x - shoulder_mid_x) / shoulder_width,
        "head_offset_y": (ear_mid_y - shoulder_mid_y) / shoulder_width,
        "shoulder_width": shoulder_width,
    }

    hip_indices = (23, 24)
    if all(
        float(getattr(pose_landmarks[index], "visibility", 0.0) or 0.0)
        >= selected.landmark_visibility_min
        for index in hip_indices
    ):
        hip_mid_x = (float(pose_landmarks[23].x) + float(pose_landmarks[24].x)) / 2.0
        hip_mid_y = (float(pose_landmarks[23].y) + float(pose_landmarks[24].y)) / 2.0
        torso_height = math.hypot(shoulder_mid_x - hip_mid_x, shoulder_mid_y - hip_mid_y)
        values["torso_lean"] = (shoulder_mid_x - hip_mid_x) / max(torso_height, 1e-6)

    if face_landmarks is not None and len(face_landmarks) >= 10:
        face_x = [float(point.x) for point in face_landmarks]
        face_y = [float(point.y) for point in face_landmarks]
        face_width = max(face_x) - min(face_x)
        face_height = max(face_y) - min(face_y)
        values["face_to_shoulder_scale"] = math.sqrt(face_width * face_height) / shoulder_width

    return FeatureVector("posture", values)
