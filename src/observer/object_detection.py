"""Pluggable object-detector backends used by Observer.

The production default remains MediaPipe's EfficientDet-Lite0 model.  The
YOLO backend is intentionally optional and imports ONNX Runtime lazily so a
normal Observer installation does not acquire a heavyweight training stack.
Both backends expose the same small, normalized result type; phone tracking,
evidence retention, and scoring stay outside this module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from observer.observations import NormalizedBox
from observer.models import sha256_file


class DetectorRuntimeUnavailable(RuntimeError):
    """Raised when an optional detector runtime is not installed."""


PHONE_CLASS_NAMES = frozenset({"cell phone", "cell_phone"})


def is_phone_class_name(class_name: str | None) -> bool:
    """Return whether a detector class name denotes a mobile phone.

    MediaPipe uses the COCO spelling ``cell phone`` while the project's
    single-class YOLO export uses the dataset label ``cell_phone``.  Keep the
    alias handling at the detector boundary so live rules and replay reports
    cannot silently drop a valid phone detection.
    """

    return class_name in PHONE_CLASS_NAMES


@dataclass(frozen=True, slots=True)
class ObjectDetection:
    """A detector-agnostic detection in normalized frame coordinates."""

    class_name: str
    confidence: float
    box: NormalizedBox
    model_id: str
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("detection confidence must be between 0 and 1")
        if self.latency_ms < 0:
            raise ValueError("detection latency cannot be negative")


class ObjectDetectorBackend(Protocol):
    """Common interface for frame-based object detection."""

    model_id: str

    def detect(self, bgr_frame: Any, timestamp_ms: int) -> tuple[ObjectDetection, ...]:
        ...

    def close(self) -> None:
        ...


class MediaPipeEfficientDetBackend:
    """Adapt a MediaPipe Tasks ObjectDetector to the Observer interface."""

    model_id = "efficientdet_lite0_int8_mediapipe"

    @property
    def supports_person_detection(self) -> bool:
        return 'person' in self._category_allowlist

    def __init__(
        self,
        *,
        mp_module: Any,
        detector: Any,
        category_allowlist: Sequence[str] = ("cell phone", "person"),
    ) -> None:
        self._mp = mp_module
        self._detector = detector
        self._category_allowlist = frozenset(category_allowlist)
        self.last_latency_ms = 0.0

    def detect(self, bgr_frame: Any, timestamp_ms: int) -> tuple[ObjectDetection, ...]:
        import cv2

        frame_height, frame_width = bgr_frame.shape[:2]
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=rgb_frame,
        )
        started = time.perf_counter()
        result = self._detector.detect_for_video(image, timestamp_ms)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.last_latency_ms = latency_ms
        detections: list[ObjectDetection] = []
        for detection in result.detections:
            categories = [
                category
                for category in detection.categories
                if category.category_name in self._category_allowlist
            ]
            if not categories:
                continue
            category = max(categories, key=lambda item: item.score or 0.0)
            box = detection.bounding_box
            left = _clamp(box.origin_x / frame_width)
            top = _clamp(box.origin_y / frame_height)
            right = _clamp((box.origin_x + box.width) / frame_width)
            bottom = _clamp((box.origin_y + box.height) / frame_height)
            detections.append(
                ObjectDetection(
                    class_name=str(category.category_name),
                    confidence=float(category.score or 0.0),
                    box=NormalizedBox(left, top, max(0.0, right - left), max(0.0, bottom - top)),
                    model_id=self.model_id,
                    latency_ms=latency_ms,
                )
            )
        return tuple(detections)

    def close(self) -> None:
        close = getattr(self._detector, "close", None)
        if close is not None:
            close()


class YoloOnnxBackend:
    """Run a COCO-trained YOLO detection export through ONNX Runtime.

    The implementation supports the common Ultralytics export layouts
    ``(1, 84, N)``/``(1, N, 84)`` and the older objectness layout ``85``.  It
    intentionally accepts an injected session for deterministic unit tests;
    normal use creates an ONNX Runtime CPU session lazily.
    """

    model_id = "yolo11n_onnx_coco"
    COCO_NAMES = {
        0: "person",
        67: "cell phone",
    }

    @property
    def supports_person_detection(self) -> bool:
        if self.single_class_name is not None:
            return self.single_class_name == 'person'
        return 'person' in self.class_names.values()

    def __init__(
        self,
        model_path: str | Path,
        *,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        session: Any | None = None,
        providers: Sequence[str] | None = None,
        max_detections: int = 100,
        class_names: dict[int, str] | None = None,
        single_class_name: str | None = None,
    ) -> None:
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("YOLO score threshold must be between 0 and 1")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("YOLO IoU threshold must be between 0 and 1")
        if input_size <= 0:
            raise ValueError("YOLO input size must be positive")
        if max_detections <= 0:
            raise ValueError("YOLO max detections must be positive")
        self.model_path = Path(model_path)
        if not self.model_path.is_file() and session is None:
            raise FileNotFoundError(f"YOLO ONNX model not found: {self.model_path}")
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.max_detections = max_detections
        self.class_names = class_names or dict(self.COCO_NAMES)
        self.single_class_name = single_class_name
        self._session = session or self._create_session(providers)
        self.last_latency_ms = 0.0

    def _create_session(self, providers: Sequence[str] | None) -> Any:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on host setup
            raise DetectorRuntimeUnavailable(
                "YOLO ONNX Runtime is not installed. Install the optional "
                "dependency with `uv sync --extra yolo` or "
                "`python -m pip install onnxruntime`."
            ) from exc
        if providers is None:
            providers = ("CPUExecutionProvider",)
        options = ort.SessionOptions()
        # Observer runs intermittently; idle inference workers should sleep
        # instead of competing with the user's foreground applications.
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        return ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=list(providers)
        )

    @property
    def input_name(self) -> str:
        return str(self._session.get_inputs()[0].name)

    def detect(self, bgr_frame: Any, timestamp_ms: int) -> tuple[ObjectDetection, ...]:
        del timestamp_ms  # the ONNX API is synchronous and timestamp-free
        started = time.perf_counter()
        blob, scale, pad_x, pad_y = _letterbox(bgr_frame, self.input_size)
        outputs = self._session.run(None, {self.input_name: blob})
        raw = _first_output(outputs)
        candidates = self._decode(raw, bgr_frame.shape[1], bgr_frame.shape[0], scale, pad_x, pad_y)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.last_latency_ms = latency_ms
        return tuple(
            ObjectDetection(
                class_name=name,
                confidence=confidence,
                box=box,
                model_id=self.model_id,
                latency_ms=latency_ms,
            )
            for name, confidence, box in candidates
        )

    def _decode(
        self,
        raw: Any,
        frame_width: int,
        frame_height: int,
        scale: float,
        pad_x: float,
        pad_y: float,
    ) -> list[tuple[str, float, NormalizedBox]]:
        import numpy as np

        array = np.asarray(raw)
        if array.ndim == 3:
            array = array[0]
        if array.ndim != 2:
            raise ValueError(f"unsupported YOLO output shape: {tuple(array.shape)}")
        # Ultralytics exports are commonly [channels, candidates].
        if array.shape[0] in (5, 84, 85) or (
            array.shape[0] < 128 and array.shape[1] > 128
        ):
            array = array.T
        if array.shape[1] < 5:
            raise ValueError(f"unsupported YOLO output width: {array.shape[1]}")

        decoded: list[tuple[str, float, NormalizedBox]] = []
        if array.shape[1] == 5:
            # Single-class exports have thousands of low-score candidates.
            # Filter in NumPy before entering Python; retain original order
            # and float64 threshold semantics used by float(row[4]) below.
            scores64 = array[:, 4].astype(np.float64, copy=False)
            array = array[~(scores64 < self.score_threshold)]
        for row in array:
            cx, cy, width, height = (float(value) for value in row[:4])
            scores = row[4:]
            objectness = 1.0
            single_class_output = len(scores) == 1
            # Older exports include one objectness column before the class
            # scores (85 values total for COCO); current Ultralytics exports
            # use 84 values (4 box + 80 class scores).
            if len(scores) == 81:
                objectness = float(scores[0])
                scores = scores[1:]
            class_index = int(np.argmax(scores))
            confidence = objectness * float(scores[class_index])
            class_name = (
                self.single_class_name
                if single_class_output
                else self.class_names.get(class_index)
            )
            if class_name is None or confidence < self.score_threshold:
                continue
            # Exported YOLO coordinates are in input-image pixels.  A small
            # number of custom exports use normalized coordinates; support
            # those without affecting the standard 640px export.
            if max(abs(cx), abs(cy), abs(width), abs(height)) <= 2.0:
                cx *= self.input_size
                cy *= self.input_size
                width *= self.input_size
                height *= self.input_size
            left = (cx - width / 2.0 - pad_x) / scale
            top = (cy - height / 2.0 - pad_y) / scale
            right = (cx + width / 2.0 - pad_x) / scale
            bottom = (cy + height / 2.0 - pad_y) / scale
            box = _normalized_box(left, top, right, bottom, frame_width, frame_height)
            decoded.append((class_name, confidence, box))
        return _nms(decoded, self.iou_threshold, self.max_detections)

    def close(self) -> None:
        self._session = None


def detector_runtime_report(model_path: str | Path | None = None) -> dict[str, object]:
    """Return a side-effect-free readiness report for the optional YOLO path."""

    try:
        import onnxruntime as ort

        runtime = {"available": True, "version": getattr(ort, "__version__", "unknown")}
    except ImportError:
        runtime = {"available": False, "version": None}
    path = Path(model_path) if model_path is not None else None
    return {
        "backend": "yolo_onnx",
        "runtime": runtime,
        "model": {
            "path": str(path) if path is not None else None,
            "exists": bool(path and path.is_file()),
            "size_bytes": path.stat().st_size if path and path.is_file() else None,
            "sha256": sha256_file(path) if path and path.is_file() else None,
        },
        "ready": bool(runtime["available"] and path and path.is_file()),
    }


def _first_output(outputs: Any) -> Any:
    if not outputs:
        raise ValueError("YOLO ONNX session returned no outputs")
    return outputs[0]


def _letterbox(frame: Any, input_size: int) -> tuple[Any, float, float, float]:
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    scale = min(input_size / width, input_size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x = (input_size - resized_width) / 2.0
    pad_y = (input_size - resized_height) / 2.0
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    left, top = round(pad_x), round(pad_y)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return blob, scale, pad_x, pad_y


def _normalized_box(
    left: float,
    top: float,
    right: float,
    bottom: float,
    frame_width: int,
    frame_height: int,
) -> NormalizedBox:
    left = _clamp(left / frame_width)
    top = _clamp(top / frame_height)
    right = _clamp(right / frame_width)
    bottom = _clamp(bottom / frame_height)
    return NormalizedBox(left, top, max(0.0, right - left), max(0.0, bottom - top))


def _nms(
    candidates: list[tuple[str, float, NormalizedBox]],
    iou_threshold: float,
    max_detections: int,
) -> list[tuple[str, float, NormalizedBox]]:
    kept: list[tuple[str, float, NormalizedBox]] = []
    # Suppress only within the same class; person and phone may overlap.
    for class_name in sorted({candidate[0] for candidate in candidates}):
        pending = sorted(
            (candidate for candidate in candidates if candidate[0] == class_name),
            key=lambda candidate: candidate[1],
            reverse=True,
        )
        while pending and len(kept) < max_detections:
            current = pending.pop(0)
            kept.append(current)
            pending = [
                candidate
                for candidate in pending
                if _iou(current[2], candidate[2]) < iou_threshold
            ]
    return sorted(kept, key=lambda candidate: candidate[1], reverse=True)[:max_detections]


def _iou(first: NormalizedBox, second: NormalizedBox) -> float:
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.x + first.width, second.x + second.width)
    bottom = min(first.y + first.height, second.y + second.height)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union if union else 0.0


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
