from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum

from observer.observations import Observation, SignalState


class RulePhase(str, Enum):
    CLEAR = "clear"
    PENDING = "pending"
    ACTIVE = "active"
    RECOVERING = "recovering"


class ActionKind(str, Enum):
    ACTIVATED = "activated"
    ALERT = "alert"
    PENALTY = "penalty"
    RESOLVED = "resolved"


@dataclass(frozen=True, slots=True)
class RuleConfig:
    rule_id: str
    signal_name: str
    trigger_after_seconds: float
    recovery_after_seconds: float = 2.0
    unknown_grace_seconds: float = 3.0
    alert_cooldown_seconds: float = 30.0
    initial_penalty_points: float = 5.0
    repeat_penalty_points: float = 2.0
    repeat_penalty_interval_seconds: float = 60.0
    maximum_penalty_points: float = 25.0
    maximum_sample_gap_seconds: float = 2.0

    def __post_init__(self) -> None:
        if not self.rule_id or not self.signal_name:
            raise ValueError("rule_id and signal_name are required")
        durations = (
            self.trigger_after_seconds,
            self.recovery_after_seconds,
            self.unknown_grace_seconds,
            self.alert_cooldown_seconds,
            self.repeat_penalty_interval_seconds,
            self.maximum_sample_gap_seconds,
        )
        if any(value < 0 for value in durations):
            raise ValueError("rule durations cannot be negative")
        if self.maximum_sample_gap_seconds == 0:
            raise ValueError("maximum_sample_gap_seconds must be positive")
        if min(
            self.initial_penalty_points,
            self.repeat_penalty_points,
            self.maximum_penalty_points,
        ) < 0:
            raise ValueError("penalty points cannot be negative")


@dataclass(frozen=True, slots=True)
class RuleAction:
    kind: ActionKind
    rule_id: str
    event_id: str
    at_monotonic: float
    event_started_monotonic: float
    observed_violation_seconds: float
    requested_points: float = 0.0
    reason: str | None = None


class SustainedRule:
    def __init__(self, config: RuleConfig) -> None:
        self.config = config
        self.phase = RulePhase.CLEAR
        self._last_timestamp: float | None = None
        self._last_signal: SignalState | None = None
        self._pending_started: float | None = None
        self._pending_seconds = 0.0
        self._event_id: str | None = None
        self._event_started: float | None = None
        self._violation_seconds = 0.0
        self._next_penalty_seconds = 0.0
        self._next_alert_at = 0.0
        self._recovery_seconds = 0.0
        self._unknown_since: float | None = None

    def update(self, observation: Observation) -> tuple[RuleAction, ...]:
        now = observation.monotonic_seconds
        if self._last_timestamp is not None and now < self._last_timestamp:
            raise ValueError("observations must be processed in monotonic order")
        signal = observation.signal(self.config.signal_name).state
        delta = 0.0 if self._last_timestamp is None else now - self._last_timestamp
        continuous = delta <= self.config.maximum_sample_gap_seconds
        actions: list[RuleAction] = []

        if signal is SignalState.EXEMPT:
            if self.phase in {RulePhase.ACTIVE, RulePhase.RECOVERING}:
                actions.append(self._action(ActionKind.RESOLVED, now, reason="exempt"))
            self._reset()
        elif self.phase is RulePhase.CLEAR:
            if signal is SignalState.VIOLATION:
                self.phase = RulePhase.PENDING
                self._pending_started = now
                self._pending_seconds = 0.0
                if self.config.trigger_after_seconds == 0:
                    actions.extend(self._activate(now))
        elif self.phase is RulePhase.PENDING:
            if signal is SignalState.VIOLATION:
                if self._last_signal is SignalState.VIOLATION and continuous:
                    self._pending_seconds += delta
                else:
                    self._pending_started = now
                    self._pending_seconds = 0.0
                if self._pending_seconds >= self.config.trigger_after_seconds:
                    actions.extend(self._activate(now))
            else:
                self._reset()
        elif self.phase is RulePhase.ACTIVE:
            if signal is SignalState.VIOLATION:
                self._unknown_since = None
                if self._last_signal is SignalState.VIOLATION and continuous:
                    self._violation_seconds += delta
                actions.extend(self._due_actions(now))
            elif signal is SignalState.COMPLIANT:
                self.phase = RulePhase.RECOVERING
                self._recovery_seconds = 0.0
                self._unknown_since = None
                if self.config.recovery_after_seconds == 0:
                    actions.append(self._action(ActionKind.RESOLVED, now, reason="recovered"))
                    self._reset()
            else:
                actions.extend(self._handle_unknown(now))
        elif self.phase is RulePhase.RECOVERING:
            if signal is SignalState.VIOLATION:
                self.phase = RulePhase.ACTIVE
                self._recovery_seconds = 0.0
                self._unknown_since = None
                actions.extend(self._due_actions(now))
            elif signal is SignalState.COMPLIANT:
                self._unknown_since = None
                if self._last_signal is SignalState.COMPLIANT and continuous:
                    self._recovery_seconds += delta
                else:
                    self._recovery_seconds = 0.0
                if self._recovery_seconds >= self.config.recovery_after_seconds:
                    actions.append(self._action(ActionKind.RESOLVED, now, reason="recovered"))
                    self._reset()
            else:
                actions.extend(self._handle_unknown(now))

        self._last_timestamp = now
        self._last_signal = signal
        return tuple(actions)

    def _activate(self, now: float) -> list[RuleAction]:
        self.phase = RulePhase.ACTIVE
        self._event_id = str(uuid.uuid4())
        self._event_started = self._pending_started if self._pending_started is not None else now
        self._violation_seconds = self._pending_seconds
        self._next_penalty_seconds = (
            self._violation_seconds + self.config.repeat_penalty_interval_seconds
        )
        self._next_alert_at = now + self.config.alert_cooldown_seconds
        return [
            self._action(ActionKind.ACTIVATED, now),
            self._action(ActionKind.ALERT, now),
            self._action(
                ActionKind.PENALTY,
                now,
                requested_points=self.config.initial_penalty_points,
            ),
        ]

    def _due_actions(self, now: float) -> list[RuleAction]:
        actions: list[RuleAction] = []
        interval = self.config.repeat_penalty_interval_seconds
        if interval > 0:
            while self._violation_seconds >= self._next_penalty_seconds:
                actions.append(
                    self._action(
                        ActionKind.PENALTY,
                        now,
                        requested_points=self.config.repeat_penalty_points,
                    )
                )
                self._next_penalty_seconds += interval
        if self.config.alert_cooldown_seconds > 0 and now >= self._next_alert_at:
            actions.append(self._action(ActionKind.ALERT, now))
            self._next_alert_at = now + self.config.alert_cooldown_seconds
        return actions

    def _handle_unknown(self, now: float) -> list[RuleAction]:
        if self._unknown_since is None:
            self._unknown_since = now
            if self.config.unknown_grace_seconds > 0:
                return []
        if now - self._unknown_since >= self.config.unknown_grace_seconds:
            action = self._action(ActionKind.RESOLVED, now, reason="observation_lost")
            self._reset()
            return [action]
        return []

    def _action(
        self,
        kind: ActionKind,
        now: float,
        *,
        requested_points: float = 0.0,
        reason: str | None = None,
    ) -> RuleAction:
        if self._event_id is None or self._event_started is None:
            raise RuntimeError("cannot emit an action without an active event")
        return RuleAction(
            kind=kind,
            rule_id=self.config.rule_id,
            event_id=self._event_id,
            at_monotonic=now,
            event_started_monotonic=self._event_started,
            observed_violation_seconds=self._violation_seconds,
            requested_points=requested_points,
            reason=reason,
        )

    def _reset(self) -> None:
        self.phase = RulePhase.CLEAR
        self._pending_started = None
        self._pending_seconds = 0.0
        self._event_id = None
        self._event_started = None
        self._violation_seconds = 0.0
        self._next_penalty_seconds = 0.0
        self._next_alert_at = 0.0
        self._recovery_seconds = 0.0
        self._unknown_since = None


def default_rule_configs() -> tuple[RuleConfig, ...]:
    """Provisional, user-configurable MVP defaults; not medical or productivity claims."""

    return (
        RuleConfig(
            "away",
            "away",
            trigger_after_seconds=20.0,
            initial_penalty_points=3.0,
            maximum_penalty_points=15.0,
        ),
        RuleConfig(
            "attention_away",
            "attention_away",
            trigger_after_seconds=8.0,
            initial_penalty_points=3.0,
            maximum_penalty_points=15.0,
        ),
        RuleConfig(
            "posture_deviation",
            "posture_deviation",
            trigger_after_seconds=20.0,
            initial_penalty_points=2.0,
            maximum_penalty_points=10.0,
        ),
        RuleConfig(
            "phone_held",
            "phone_held",
            trigger_after_seconds=10.0,
            initial_penalty_points=5.0,
            repeat_penalty_points=2.0,
            repeat_penalty_interval_seconds=60.0,
        ),
    )
