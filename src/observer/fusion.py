from __future__ import annotations

from dataclasses import dataclass

from observer.observations import (
    NormalizedPoint,
    Observation,
    PhoneCandidate,
    SignalState,
    SignalValue,
    phone_held_signal,
)


@dataclass(frozen=True, slots=True)
class SignalFreshness:
    quality_seconds: float = 0.75
    face_seconds: float = 0.75
    pose_seconds: float = 1.0
    object_seconds: float = 2.0
    attention_seconds: float = 0.75
    posture_seconds: float = 1.0


class ObservationFusion:
    """Cache differently paced model results and produce one conservative observation."""

    def __init__(self, freshness: SignalFreshness | None = None) -> None:
        self.freshness = freshness or SignalFreshness()
        self._quality_at: float | None = None
        self._quality_observable = False
        self._quality_reason = "frame quality is not available"
        self._face_at: float | None = None
        self._face_count = 0
        self._pose_at: float | None = None
        self._pose_count = 0
        self._wrists: tuple[NormalizedPoint, ...] = ()
        self._object_at: float | None = None
        self._phones: tuple[PhoneCandidate, ...] = ()
        self._person_count = 0
        self._attention_at: float | None = None
        self._attention = SignalValue.unknown("attention calibration not available")
        self._posture_at: float | None = None
        self._posture = SignalValue.unknown("posture calibration not available")
        self._ergonomic_posture_at: float | None = None
        self._ergonomic_posture = SignalValue.unknown(
            "ergonomic posture calibration not available"
        )

    def update_quality(self, at_monotonic: float, observable: bool, reason: str | None) -> None:
        self._quality_at = at_monotonic
        self._quality_observable = observable
        self._quality_reason = reason or "frame quality gate rejected the observation"

    def update_face(self, at_monotonic: float, face_count: int) -> None:
        self._face_at = at_monotonic
        self._face_count = max(0, face_count)

    def update_pose(
        self,
        at_monotonic: float,
        pose_count: int,
        wrists: tuple[NormalizedPoint, ...],
        *,
        presence_credible: bool = True,
    ) -> None:
        self._pose_at = at_monotonic
        self._pose_count = max(0, pose_count) if presence_credible else 0
        self._wrists = wrists if self._pose_count > 0 else ()

    def update_phones(
        self, at_monotonic: float, phones: tuple[PhoneCandidate, ...]
    ) -> None:
        self._object_at = at_monotonic
        self._phones = phones

    def update_person_presence(self, at_monotonic: float, person_count: int) -> None:
        self._object_at = at_monotonic
        self._person_count = max(0, person_count)

    def update_attention(self, at_monotonic: float, signal: SignalValue) -> None:
        self._attention_at = at_monotonic
        self._attention = signal

    def update_posture(self, at_monotonic: float, signal: SignalValue) -> None:
        self._posture_at = at_monotonic
        self._posture = signal

    def update_ergonomic_posture(
        self, at_monotonic: float, signal: SignalValue
    ) -> None:
        self._ergonomic_posture_at = at_monotonic
        self._ergonomic_posture = signal

    def build(self, now_monotonic: float) -> Observation:
        quality_fresh = self._is_fresh(
            self._quality_at, now_monotonic, self.freshness.quality_seconds
        )
        face_fresh = self._is_fresh(self._face_at, now_monotonic, self.freshness.face_seconds)
        pose_fresh = self._is_fresh(self._pose_at, now_monotonic, self.freshness.pose_seconds)
        object_fresh = self._is_fresh(
            self._object_at, now_monotonic, self.freshness.object_seconds
        )
        attention_fresh = self._is_fresh(
            self._attention_at, now_monotonic, self.freshness.attention_seconds
        )
        posture_fresh = self._is_fresh(
            self._posture_at, now_monotonic, self.freshness.posture_seconds
        )
        ergonomic_posture_fresh = self._is_fresh(
            self._ergonomic_posture_at,
            now_monotonic,
            self.freshness.posture_seconds,
        )

        if quality_fresh and not self._quality_observable:
            away = SignalValue.unknown(self._quality_reason)
        elif (
            (face_fresh and self._face_count > 0)
            or (pose_fresh and self._pose_count > 0)
            or (object_fresh and self._person_count > 0)
        ):
            away = SignalValue.compliant(confidence=1.0)
        elif face_fresh and pose_fresh and object_fresh:
            away = SignalValue.violation(confidence=1.0)
        else:
            away = SignalValue.unknown(
                "face, pose, and person detector presence are not all fresh"
            )

        if quality_fresh and not self._quality_observable:
            phone_held = SignalValue.unknown(self._quality_reason)
        elif not object_fresh:
            phone_held = SignalValue.unknown("phone detector result is stale")
        elif not self._phones:
            visible_wrists = tuple(wrist for wrist in self._wrists if wrist.visibility >= 0.5)
            if pose_fresh and self._pose_count > 0 and visible_wrists:
                phone_held = phone_held_signal((), visible_wrists)
            else:
                phone_held = SignalValue.unknown(
                    "no phone is visible but hands are outside the observable area"
                )
        elif not pose_fresh:
            phone_held = SignalValue.unknown("phone visible but pose result is stale")
        else:
            phone_held = phone_held_signal(self._phones, self._wrists)

        if quality_fresh and not self._quality_observable:
            attention = SignalValue.unknown(self._quality_reason)
            posture = SignalValue.unknown(self._quality_reason)
        else:
            attention = (
                self._attention
                if attention_fresh
                else SignalValue.unknown("attention result is stale or calibration unavailable")
            )
            posture = (
                self._posture
                if posture_fresh
                else SignalValue.unknown("posture result is stale or calibration unavailable")
            )

        if quality_fresh and not self._quality_observable:
            ergonomic_risk = SignalValue.unknown(self._quality_reason)
        else:
            risk_reasons: list[str] = []
            risk_metrics: dict[str, float] = {}
            risk_confidences: list[float] = []
            if attention_fresh and self._attention.reason and "conflict" in self._attention.reason:
                risk_reasons.append("eye_head_compensation")
                risk_metrics.update(self._attention.metrics)
            if ergonomic_posture_fresh:
                risk_metrics.update(self._ergonomic_posture.metrics)
                if self._ergonomic_posture.state is SignalState.VIOLATION:
                    if self._ergonomic_posture.reason:
                        risk_reasons.extend(self._ergonomic_posture.reason.split(","))
                    if self._ergonomic_posture.confidence is not None:
                        risk_confidences.append(self._ergonomic_posture.confidence)
            if risk_reasons:
                ergonomic_risk = SignalValue.violation(
                    confidence=max(risk_confidences, default=0.7),
                    reason=",".join(sorted(set(risk_reasons))),
                    **risk_metrics,
                )
            elif (
                ergonomic_posture_fresh
                and self._ergonomic_posture.state is SignalState.COMPLIANT
            ):
                ergonomic_risk = SignalValue.compliant(
                    self._ergonomic_posture.confidence, **risk_metrics
                )
            else:
                ergonomic_risk = SignalValue.unknown(
                    "ergonomic risk features are not currently observable",
                    **risk_metrics,
                )

        return Observation(
            monotonic_seconds=now_monotonic,
            signals={
                "away": away,
                "phone_held": phone_held,
                "attention_away": attention,
                "posture_deviation": posture,
                "ergonomic_risk": ergonomic_risk,
            },
        )

    @staticmethod
    def _is_fresh(at: float | None, now: float, maximum_age: float) -> bool:
        return at is not None and 0.0 <= now - at <= maximum_age
