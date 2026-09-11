from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class SignalState(str, Enum):
    COMPLIANT = "compliant"
    VIOLATION = "violation"
    UNKNOWN = "unknown"
    EXEMPT = "exempt"


@dataclass(frozen=True, slots=True)
class SignalValue:
    state: SignalState
    confidence: float | None = None
    reason: str | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("signal confidence must be between 0 and 1")
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    @classmethod
    def compliant(cls, confidence: float | None = None, **metrics: float) -> "SignalValue":
        return cls(SignalState.COMPLIANT, confidence=confidence, metrics=metrics)

    @classmethod
    def violation(
        cls,
        confidence: float | None = None,
        *,
        reason: str | None = None,
        **metrics: float,
    ) -> "SignalValue":
        return cls(
            SignalState.VIOLATION,
            confidence=confidence,
            reason=reason,
            metrics=metrics,
        )

    @classmethod
    def unknown(cls, reason: str, **metrics: float) -> "SignalValue":
        return cls(SignalState.UNKNOWN, reason=reason, metrics=metrics)

    @classmethod
    def exempt(cls, reason: str) -> "SignalValue":
        return cls(SignalState.EXEMPT, reason=reason)


@dataclass(frozen=True, slots=True)
class Observation:
    monotonic_seconds: float
    signals: Mapping[str, SignalValue]
    captured_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if self.monotonic_seconds < 0:
            raise ValueError("observation timestamp must be non-negative")
        object.__setattr__(self, "signals", MappingProxyType(dict(self.signals)))

    def signal(self, name: str) -> SignalValue:
        return self.signals.get(name, SignalValue.unknown(f"missing signal: {name}"))


@dataclass(frozen=True, slots=True)
class NormalizedPoint:
    x: float
    y: float
    visibility: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.x <= 1.0 or not 0.0 <= self.y <= 1.0:
            raise ValueError("point coordinates must be normalized to 0..1")
        if not 0.0 <= self.visibility <= 1.0:
            raise ValueError("point visibility must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class NormalizedBox:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width < 0 or self.height < 0:
            raise ValueError("box size cannot be negative")
        if not 0.0 <= self.x <= 1.0 or not 0.0 <= self.y <= 1.0:
            raise ValueError("box origin must be normalized to 0..1")
        if self.x + self.width > 1.000001 or self.y + self.height > 1.000001:
            raise ValueError("box must fit inside normalized frame")

    def distance_to(self, point: NormalizedPoint) -> float:
        nearest_x = min(max(point.x, self.x), self.x + self.width)
        nearest_y = min(max(point.y, self.y), self.y + self.height)
        return ((point.x - nearest_x) ** 2 + (point.y - nearest_y) ** 2) ** 0.5


@dataclass(frozen=True, slots=True)
class PhoneCandidate:
    box: NormalizedBox
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("phone confidence must be between 0 and 1")


def phone_visible_signal(phones: tuple[PhoneCandidate, ...]) -> SignalValue:
    """Classify a confirmed in-frame phone candidate as a phone-use violation."""
    if not phones:
        return SignalValue.compliant(confidence=1.0, phone_count=0.0)
    return SignalValue.violation(
        confidence=max(phone.confidence for phone in phones),
        phone_count=float(len(phones)),
    )
