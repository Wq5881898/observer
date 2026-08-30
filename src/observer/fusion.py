from __future__ import annotations

from dataclasses import dataclass

from observer.observations import (
    NormalizedPoint,
    Observation,
    PhoneCandidate,
    SignalValue,
    phone_held_signal,
)


@dataclass(frozen=True, slots=True)
class SignalFreshness:
    face_seconds: float = 0.75
    pose_seconds: float = 1.0
    object_seconds: float = 2.0


class ObservationFusion:
    """Cache differently paced model results and produce one conservative observation."""

    def __init__(self, freshness: SignalFreshness | None = None) -> None:
        self.freshness = freshness or SignalFreshness()
        self._face_at: float | None = None
        self._face_count = 0
        self._pose_at: float | None = None
        self._pose_count = 0
        self._wrists: tuple[NormalizedPoint, ...] = ()
        self._object_at: float | None = None
        self._phones: tuple[PhoneCandidate, ...] = ()

    def update_face(self, at_monotonic: float, face_count: int) -> None:
        self._face_at = at_monotonic
        self._face_count = max(0, face_count)

    def update_pose(
        self,
        at_monotonic: float,
        pose_count: int,
        wrists: tuple[NormalizedPoint, ...],
    ) -> None:
        self._pose_at = at_monotonic
        self._pose_count = max(0, pose_count)
        self._wrists = wrists if pose_count > 0 else ()

    def update_phones(
        self, at_monotonic: float, phones: tuple[PhoneCandidate, ...]
    ) -> None:
        self._object_at = at_monotonic
        self._phones = phones

    def build(self, now_monotonic: float) -> Observation:
        face_fresh = self._is_fresh(self._face_at, now_monotonic, self.freshness.face_seconds)
        pose_fresh = self._is_fresh(self._pose_at, now_monotonic, self.freshness.pose_seconds)
        object_fresh = self._is_fresh(
            self._object_at, now_monotonic, self.freshness.object_seconds
        )

        if (face_fresh and self._face_count > 0) or (pose_fresh and self._pose_count > 0):
            away = SignalValue.compliant(confidence=1.0)
        elif face_fresh and pose_fresh:
            away = SignalValue.violation(confidence=1.0)
        else:
            away = SignalValue.unknown("face and pose presence are not both fresh")

        if not object_fresh:
            phone_held = SignalValue.unknown("phone detector result is stale")
        elif not self._phones:
            phone_held = phone_held_signal((), ())
        elif not pose_fresh:
            phone_held = SignalValue.unknown("phone visible but pose result is stale")
        else:
            phone_held = phone_held_signal(self._phones, self._wrists)

        return Observation(
            monotonic_seconds=now_monotonic,
            signals={
                "away": away,
                "phone_held": phone_held,
                "attention_away": SignalValue.unknown("attention calibration not available"),
                "posture_deviation": SignalValue.unknown("posture calibration not available"),
            },
        )

    @staticmethod
    def _is_fresh(at: float | None, now: float, maximum_age: float) -> bool:
        return at is not None and 0.0 <= now - at <= maximum_age

