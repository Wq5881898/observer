from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FrameSnapshot:
    sequence: int
    captured_monotonic: float
    frame: Any


class LatestFrameBuffer:
    """A single-slot frame buffer that never lets camera frames accumulate."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: FrameSnapshot | None = None
        self._sequence = 0

    def publish(self, frame: Any, captured_monotonic: float) -> FrameSnapshot:
        with self._lock:
            self._sequence += 1
            snapshot = FrameSnapshot(self._sequence, captured_monotonic, frame)
            self._latest = snapshot
            return snapshot

    def latest(self, after_sequence: int | None = None) -> FrameSnapshot | None:
        with self._lock:
            snapshot = self._latest
            if snapshot is None:
                return None
            if after_sequence is not None and snapshot.sequence <= after_sequence:
                return None
            return snapshot

