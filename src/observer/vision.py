from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from observer.config import InferenceConfig
from observer.fusion import ObservationFusion
from observer.models import ModelFiles
from observer.observations import NormalizedBox, NormalizedPoint, Observation, PhoneCandidate
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

    def summary(self) -> dict[str, Any]:
        return {
            "face": self.face.summary(),
            "pose": self.pose.summary(),
            "object": self.object.summary(),
            "maximum_faces": self.maximum_faces,
            "maximum_poses": self.maximum_poses,
            "phone_detections": self.phone_detections,
        }


class MediapipeVisionProbe:
    """Rate-limited synchronous probe for target-device performance validation."""

    def __init__(self, models: ModelFiles, rates: InferenceConfig) -> None:
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

    def close(self) -> None:
        self._object.close()
        self._pose.close()
        self._face.close()

    def process_due(self, bgr_frame: Any, now_monotonic: float) -> bool:
        due = {
            "face": now_monotonic >= self._next_due["face"],
            "pose": now_monotonic >= self._next_due["pose"],
            "object": now_monotonic >= self._next_due["object"],
        }
        if not any(due.values()):
            return False

        import cv2

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
            self._next_due["face"] = now_monotonic + 1.0 / self._rates.face_fps

        if due["pose"]:
            started = time.perf_counter()
            result = self._pose.detect_for_video(image, timestamp_ms)
            elapsed = (time.perf_counter() - started) * 1000
            count = len(result.pose_landmarks)
            self.measurements.maximum_poses = max(self.measurements.maximum_poses, count)
            self.measurements.pose.record(count > 0, elapsed)
            wrists: tuple[NormalizedPoint, ...] = ()
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
            self._fusion.update_phones(now_monotonic, tuple(phones))
            self._next_due["object"] = now_monotonic + 1.0 / self._rates.object_fps
        return True

    def observation(self, now_monotonic: float) -> Observation:
        return self._fusion.build(now_monotonic)

    def summary(self) -> dict[str, Any]:
        return self.measurements.summary()

    @staticmethod
    def _clamp(value: float) -> float:
        return min(1.0, max(0.0, float(value)))
