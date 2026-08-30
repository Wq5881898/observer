from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from observer.calibration import CalibrationProfile
from observer.config import InferenceConfig
from observer.fusion import ObservationFusion
from observer.models import ModelFiles
from observer.observations import (
    NormalizedBox,
    NormalizedPoint,
    Observation,
    PhoneCandidate,
    SignalValue,
)
from observer.perception import (
    FeatureVector,
    FrameQuality,
    PhoneCandidateTracker,
    QualityThresholds,
    assess_frame_quality,
    extract_attention_features,
    extract_posture_features,
)
from observer.resources import percentile


_matplotlib_cache = Path(tempfile.gettempdir()) / "observer-matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))


@dataclass(slots=True)
class ModuleMeasurements:
    runs: int = 0
    positive_frames: int = 0
    latency_ms: list[float] = field(default_factory=list)

    def record(self, positive: bool, elapsed_ms: float) -> None:
        self.runs += 1
        self.positive_frames += int(positive)
        self.latency_ms.append(elapsed_ms)

    def summary(self) -> dict[str, Any]:
        average = sum(self.latency_ms) / len(self.latency_ms) if self.latency_ms else 0.0
        return {
            "runs": self.runs,
            "positive_frames": self.positive_frames,
            "latency_average_ms": round(average, 2),
            "latency_p95_ms": round(percentile(self.latency_ms, 0.95), 2),
        }


@dataclass(slots=True)
class VisionProbeMeasurements:
    face: ModuleMeasurements = field(default_factory=ModuleMeasurements)
    pose: ModuleMeasurements = field(default_factory=ModuleMeasurements)
    object: ModuleMeasurements = field(default_factory=ModuleMeasurements)
    maximum_faces: int = 0
    maximum_poses: int = 0
    phone_detections: int = 0
    quality_rejected_frames: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "face": self.face.summary(),
            "pose": self.pose.summary(),
            "object": self.object.summary(),
            "maximum_faces": self.maximum_faces,
            "maximum_poses": self.maximum_poses,
            "phone_detections": self.phone_detections,
            "quality_rejected_frames": self.quality_rejected_frames,
        }


class MediapipeVisionProbe:
    """Rate-limited synchronous probe for target-device performance validation."""

    def __init__(
        self,
        models: ModelFiles,
        rates: InferenceConfig,
        calibration: CalibrationProfile | None = None,
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._rates = rates
        mode = vision.RunningMode.VIDEO
        self._face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(models.face)),
                running_mode=mode,
                num_faces=1,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
        )
        self._pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(models.pose)),
                running_mode=mode,
                num_poses=1,
                output_segmentation_masks=False,
            )
        )
        self._object = vision.ObjectDetector.create_from_options(
            vision.ObjectDetectorOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(models.object)),
                running_mode=mode,
                max_results=3,
                score_threshold=0.25,
                category_allowlist=["cell phone"],
            )
        )
        self._next_due = {"face": 0.0, "pose": 0.0, "object": 0.0}
        self.measurements = VisionProbeMeasurements()
        self._fusion = ObservationFusion()
        self._calibration = calibration
        self._quality_thresholds = calibration.quality if calibration else QualityThresholds()
        self._last_quality: FrameQuality | None = None
        self._last_attention_features: FeatureVector | None = None
        self._last_posture_features: FeatureVector | None = None
        self._latest_face_landmarks: Any = None
        self._latest_face_at: float | None = None
        self._latest_pose_at: float | None = None
        self._latest_object_at: float | None = None
        self._last_raw_phones: tuple[PhoneCandidate, ...] = ()
        self._last_tracked_phones: tuple[PhoneCandidate, ...] = ()
        self._phone_tracker = PhoneCandidateTracker()

    def close(self) -> None:
        self._object.close()
        self._pose.close()
        self._face.close()

    def process_due(
        self,
        bgr_frame: Any,
        now_monotonic: float,
        enabled: frozenset[str] = frozenset({"face", "pose", "object"}),
    ) -> bool:
        unknown_modules = enabled - {"face", "pose", "object"}
        if unknown_modules:
            raise ValueError(f"unknown vision modules: {sorted(unknown_modules)}")
        due = {
            "face": "face" in enabled and now_monotonic >= self._next_due["face"],
            "pose": "pose" in enabled and now_monotonic >= self._next_due["pose"],
            "object": "object" in enabled and now_monotonic >= self._next_due["object"],
        }
        if not any(due.values()):
            return False

        import cv2

        quality = assess_frame_quality(bgr_frame, self._quality_thresholds)
        self._last_quality = quality
        if not quality.observable:
            self.measurements.quality_rejected_frames += 1
        self._fusion.update_quality(
            now_monotonic, quality.observable, "; ".join(quality.reasons) or None
        )
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(now_monotonic * 1000)

        if due["face"]:
            started = time.perf_counter()
            result = self._face.detect_for_video(image, timestamp_ms)
            elapsed = (time.perf_counter() - started) * 1000
            count = len(result.face_landmarks)
            self.measurements.maximum_faces = max(self.measurements.maximum_faces, count)
            self.measurements.face.record(count > 0, elapsed)
            self._fusion.update_face(now_monotonic, count)
            self._latest_face_landmarks = result.face_landmarks[0] if count > 0 else None
            self._latest_face_at = now_monotonic
            self._last_attention_features = extract_attention_features(
                self._latest_face_landmarks, self._quality_thresholds
            )
            if not quality.observable:
                attention_signal = self._unknown_quality_signal(quality)
            elif self._calibration is None:
                attention_signal = self._unknown_signal("attention calibration not available")
            else:
                attention_signal = self._calibration.attention.evaluate(
                    self._last_attention_features
                )
            self._fusion.update_attention(now_monotonic, attention_signal)
            self._next_due["face"] = now_monotonic + 1.0 / self._rates.face_fps

        if due["pose"]:
            started = time.perf_counter()
            result = self._pose.detect_for_video(image, timestamp_ms)
            elapsed = (time.perf_counter() - started) * 1000
            count = len(result.pose_landmarks)
            self.measurements.maximum_poses = max(self.measurements.maximum_poses, count)
            self.measurements.pose.record(count > 0, elapsed)
            wrists: tuple[NormalizedPoint, ...] = ()
            pose = None
            if count > 0:
                pose = result.pose_landmarks[0]
                wrists = tuple(
                    NormalizedPoint(
                        x=self._clamp(pose[index].x),
                        y=self._clamp(pose[index].y),
                        visibility=self._clamp(pose[index].visibility or 0.0),
                    )
                    for index in (15, 16)
                )
            self._fusion.update_pose(now_monotonic, count, wrists)
            self._latest_pose_at = now_monotonic
            face_is_fresh = (
                self._latest_face_at is not None
                and 0 <= now_monotonic - self._latest_face_at <= 0.75
            )
            self._last_posture_features = extract_posture_features(
                pose,
                self._latest_face_landmarks if face_is_fresh else None,
                self._quality_thresholds,
            )
            if not quality.observable:
                posture_signal = self._unknown_quality_signal(quality)
            elif self._calibration is None:
                posture_signal = self._unknown_signal("posture calibration not available")
            else:
                posture_signal = self._calibration.posture.evaluate(self._last_posture_features)
            self._fusion.update_posture(now_monotonic, posture_signal)
            self._next_due["pose"] = now_monotonic + 1.0 / self._rates.pose_fps

        if due["object"]:
            started = time.perf_counter()
            result = self._object.detect_for_video(image, timestamp_ms)
            elapsed = (time.perf_counter() - started) * 1000
            phone_count = 0
            phones: list[PhoneCandidate] = []
            frame_height, frame_width = bgr_frame.shape[:2]
            for detection in result.detections:
                phone_categories = [
                    category
                    for category in detection.categories
                    if category.category_name == "cell phone"
                ]
                if not phone_categories:
                    continue
                phone_count += 1
                box = detection.bounding_box
                left = self._clamp(box.origin_x / frame_width)
                top = self._clamp(box.origin_y / frame_height)
                right = self._clamp((box.origin_x + box.width) / frame_width)
                bottom = self._clamp((box.origin_y + box.height) / frame_height)
                phones.append(
                    PhoneCandidate(
                        NormalizedBox(left, top, right - left, bottom - top),
                        confidence=max(category.score or 0.0 for category in phone_categories),
                    )
                )
            self.measurements.phone_detections += phone_count
            self.measurements.object.record(phone_count > 0, elapsed)
            tracked_phones = self._phone_tracker.update(now_monotonic, phones)
            self._latest_object_at = now_monotonic
            self._last_raw_phones = tuple(phones)
            self._last_tracked_phones = tracked_phones
            self._fusion.update_phones(now_monotonic, tracked_phones)
            object_fps = (
                self._rates.object_burst_fps
                if self._phone_tracker.burst_active(now_monotonic)
                else self._rates.object_fps
            )
            self._next_due["object"] = now_monotonic + 1.0 / object_fps
        return True

    def observation(self, now_monotonic: float) -> Observation:
        return self._fusion.build(now_monotonic)

    def summary(self) -> dict[str, Any]:
        return self.measurements.summary()

    @property
    def latest_quality(self) -> FrameQuality | None:
        return self._last_quality

    @property
    def latest_attention_features(self) -> FeatureVector | None:
        return self._last_attention_features

    @property
    def latest_posture_features(self) -> FeatureVector | None:
        return self._last_posture_features

    def derived_snapshot(self) -> dict[str, Any]:
        """Return numeric, non-image data needed for validation and offline ablations."""

        def phone_value(phone: PhoneCandidate) -> dict[str, float]:
            return {
                "x": phone.box.x,
                "y": phone.box.y,
                "width": phone.box.width,
                "height": phone.box.height,
                "confidence": phone.confidence,
            }

        return {
            "feature_version": (
                self._last_attention_features.version
                if self._last_attention_features is not None
                else None
            ),
            "face_at": self._latest_face_at,
            "pose_at": self._latest_pose_at,
            "object_at": self._latest_object_at,
            "quality": (
                {
                    "observable": self._last_quality.observable,
                    "brightness_mean": self._last_quality.brightness_mean,
                    "blur_variance": self._last_quality.blur_variance,
                    "reasons": self._last_quality.reasons,
                }
                if self._last_quality is not None
                else None
            ),
            "attention": (
                dict(self._last_attention_features.values)
                if self._last_attention_features is not None
                else None
            ),
            "attention_reasons": (
                self._last_attention_features.reasons
                if self._last_attention_features is not None
                else ()
            ),
            "posture": (
                dict(self._last_posture_features.values)
                if self._last_posture_features is not None
                else None
            ),
            "posture_reasons": (
                self._last_posture_features.reasons
                if self._last_posture_features is not None
                else ()
            ),
            "raw_phones": [phone_value(phone) for phone in self._last_raw_phones],
            "tracked_phones": [phone_value(phone) for phone in self._last_tracked_phones],
            "runs": {
                "face": self.measurements.face.runs,
                "pose": self.measurements.pose.runs,
                "object": self.measurements.object.runs,
            },
        }

    @staticmethod
    def _unknown_signal(reason: str) -> SignalValue:
        return SignalValue.unknown(reason)

    @classmethod
    def _unknown_quality_signal(cls, quality: FrameQuality) -> SignalValue:
        return cls._unknown_signal("; ".join(quality.reasons) or "frame quality rejected")

    @staticmethod
    def _clamp(value: float) -> float:
        return min(1.0, max(0.0, float(value)))
