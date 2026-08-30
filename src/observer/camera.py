from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from typing import Any

from observer.config import CameraConfig
from observer.frames import LatestFrameBuffer


@dataclass(frozen=True, slots=True)
class CameraStats:
    opened: bool = False
    active_backend: str | None = None
    frames_captured: int = 0
    read_failures: int = 0
    started_monotonic: float | None = None
    last_frame_monotonic: float | None = None
    last_error: str | None = None


class CameraWorker:
    """Read a webcam in one background thread and expose only its latest frame."""

    def __init__(self, config: CameraConfig, buffer: LatestFrameBuffer | None = None) -> None:
        config.validate()
        self.config = config
        self.buffer = buffer or LatestFrameBuffer()
        self._stop_event = threading.Event()
        self._started_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stats_lock = threading.Lock()
        self._stats = CameraStats()

    def start(self, wait_seconds: float = 12.0) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return self.stats.opened
        self._stop_event.clear()
        self._started_event.clear()
        self._thread = threading.Thread(target=self._run, name="observer-camera", daemon=True)
        self._thread.start()
        startup_finished = self._started_event.wait(timeout=wait_seconds)
        if not startup_finished:
            self._set_stats(last_error=f"camera startup timed out after {wait_seconds:g}s")
            return False
        return self.stats.opened

    def stop(self, join_seconds: float = 3.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=join_seconds)

    @property
    def stats(self) -> CameraStats:
        with self._stats_lock:
            return replace(self._stats)

    def _set_stats(self, **changes: Any) -> None:
        with self._stats_lock:
            self._stats = replace(self._stats, **changes)

    def _run(self) -> None:
        capture = None
        try:
            import cv2

            backend_ids = {
                "auto": 0,
                "msmf": cv2.CAP_MSMF,
                "dshow": cv2.CAP_DSHOW,
            }
            fallback_order = {
                "auto": ("auto", "msmf", "dshow"),
                "msmf": ("msmf", "dshow", "auto"),
                "dshow": ("dshow", "msmf", "auto"),
            }
            active_backend = None
            for backend_name in fallback_order[self.config.backend]:
                candidate = cv2.VideoCapture(
                    self.config.device_index, backend_ids[backend_name]
                )
                if candidate.isOpened():
                    capture = candidate
                    active_backend = backend_name
                    break
                candidate.release()
            if capture is None or active_backend is None:
                self._set_stats(last_error="camera could not be opened with any backend")
                return
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            capture.set(cv2.CAP_PROP_FPS, self.config.capture_fps)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._set_stats(
                opened=True,
                active_backend=active_backend,
                started_monotonic=time.monotonic(),
            )
            self._started_event.set()

            minimum_interval = 1.0 / self.config.capture_fps
            next_read = time.monotonic()
            while not self._stop_event.is_set():
                now = time.monotonic()
                if now < next_read:
                    self._stop_event.wait(next_read - now)
                    continue
                ok, frame = capture.read()
                captured_at = time.monotonic()
                next_read = captured_at + minimum_interval
                if not ok or frame is None:
                    current = self.stats
                    self._set_stats(
                        read_failures=current.read_failures + 1,
                        last_error="camera frame read failed",
                    )
                    self._stop_event.wait(self.config.read_failure_backoff_ms / 1000.0)
                    continue
                self.buffer.publish(frame, captured_at)
                current = self.stats
                self._set_stats(
                    frames_captured=current.frames_captured + 1,
                    last_frame_monotonic=captured_at,
                    last_error=None,
                )
        except Exception as exc:  # surfaced through stats for a non-blocking worker
            self._set_stats(last_error=f"{type(exc).__name__}: {exc}")
        finally:
            if capture is not None:
                capture.release()
            self._set_stats(opened=False)
            self._started_event.set()
