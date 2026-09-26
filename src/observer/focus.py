from __future__ import annotations

import json
from dataclasses import dataclass, field
from uuid import uuid4

from observer.observations import Observation, SignalState
from observer.rules import ActionKind, RuleAction, RuleConfig, RulePhase


FOCUS_RULE_ID = "focus_violation"
FOCUS_SIGNALS = ("away", "phone_held", "attention_away")
FOCUS_NAMES = {
    "away": "离席",
    "phone_held": "可见手机",
    "attention_away": "离开屏幕",
}


def merged_seconds(segments: list[tuple[float, float]]) -> float:
    """Return the duration of the interval union without double counting."""
    normalized = sorted((start, end) for start, end in segments if end >= start)
    if not normalized:
        return 0.0
    total = 0.0
    current_start, current_end = normalized[0]
    for start, end in normalized[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


@dataclass(slots=True)
class FocusIncident:
    identifier: str
    started: float
    first_activated: float
    signals: set[str] = field(default_factory=set)
    segments: dict[str, list[list[float]]] = field(default_factory=dict)
    alert_at: float | None = None
    ended: float | None = None

    @property
    def violation_seconds(self) -> float:
        return merged_seconds([
            (segment[0], segment[1])
            for values in self.segments.values()
            for segment in values
        ])

    @property
    def label(self) -> str:
        return " + ".join(
            FOCUS_NAMES[name] for name in FOCUS_SIGNALS if name in self.signals
        ) or "专注违规"


class FocusIncidentTracker:
    """Merge confirmed scored rules while preserving their individual intervals."""

    def __init__(self, configs: tuple[RuleConfig, ...], session_started: float,
                 scoring_signals: tuple[str, ...], alert_wait_seconds: float = 10.0):
        self.session_started = session_started
        self.scoring_signals = tuple(
            signal for signal in FOCUS_SIGNALS if signal in scoring_signals
        )
        self.alert_wait_seconds = alert_wait_seconds
        self.max_gap = {
            config.signal_name: config.maximum_sample_gap_seconds
            for config in configs if config.signal_name in self.scoring_signals
        }
        self.buffers: dict[str, list[list[float]]] = {
            signal: [] for signal in self.scoring_signals
        }
        self.confirmed = {signal: False for signal in self.scoring_signals}
        self.last_states = {signal: SignalState.UNKNOWN for signal in self.scoring_signals}
        self.last_at: float | None = None
        self.current: FocusIncident | None = None
        self.incidents: list[FocusIncident] = []

    def _track_segments(self, observation: Observation) -> None:
        now = observation.monotonic_seconds
        delta = None if self.last_at is None else now - self.last_at
        for signal in self.scoring_signals:
            state = observation.signal(signal).state
            previous = self.last_states[signal]
            values = self.buffers[signal]
            continuous = (delta is not None and delta <= self.max_gap.get(signal, 2.0))
            if state is SignalState.VIOLATION:
                if previous is SignalState.VIOLATION and continuous and values:
                    values[-1][1] = now
                else:
                    values.append([now, now])
            self.last_states[signal] = state
        self.last_at = now

    def _new_action(self, kind: ActionKind, now: float,
                    reason: str | None = None) -> RuleAction:
        incident = self.current
        if incident is None:
            raise RuntimeError("focus action requires an incident")
        return RuleAction(
            kind=kind,
            rule_id=FOCUS_RULE_ID,
            event_id=incident.identifier,
            at_monotonic=now,
            event_started_monotonic=incident.started,
            observed_violation_seconds=incident.violation_seconds,
            reason=reason,
        )

    def update(self, observation: Observation, rule_actions: tuple[RuleAction, ...],
               phases: dict[str, RulePhase]) -> tuple[RuleAction, ...]:
        now = observation.monotonic_seconds
        self._track_segments(observation)
        output: list[RuleAction] = []

        for action in rule_actions:
            signal = action.rule_id
            if signal not in self.scoring_signals:
                continue
            if action.kind is ActionKind.ACTIVATED:
                if self.current is None:
                    start = min(
                        (segment[0] for segment in self.buffers[signal]),
                        default=action.event_started_monotonic,
                    )
                    self.current = FocusIncident(str(uuid4()), start, now)
                    self.incidents.append(self.current)
                    created = True
                else:
                    created = False
                    self.current.started = min(
                        self.current.started, action.event_started_monotonic
                    )
                target = self.current.segments.setdefault(signal, [])
                target.extend(self.buffers[signal])
                self.buffers[signal] = target
                self.confirmed[signal] = True
                self.current.signals.add(signal)
                if created:
                    output.append(self._new_action(
                        ActionKind.ACTIVATED, now, self.current.label
                    ))
            elif action.kind is ActionKind.RESOLVED and self.confirmed[signal]:
                self.confirmed[signal] = False
                self.buffers[signal] = []

        for signal in self.scoring_signals:
            if phases.get(signal, RulePhase.CLEAR) is RulePhase.CLEAR:
                if not self.confirmed[signal]:
                    self.buffers[signal] = []

        incident = self.current
        if incident is None:
            return tuple(output)

        pending = any(
            phases.get(signal, RulePhase.CLEAR) is RulePhase.PENDING
            for signal in self.scoring_signals
        )
        if incident.alert_at is None and (
            not pending or now - incident.first_activated >= self.alert_wait_seconds
        ):
            incident.alert_at = now
            output.append(self._new_action(ActionKind.ALERT, now, incident.label))

        if all(
            phases.get(signal, RulePhase.CLEAR) is RulePhase.CLEAR
            for signal in self.scoring_signals
        ):
            incident.ended = max(
                (segment[1] for values in incident.segments.values() for segment in values),
                default=now,
            )
            output.append(self._new_action(ActionKind.RESOLVED, now, "recovered"))
            self.current = None

        return tuple(output)

    def rows(self) -> tuple[dict, ...]:
        rows = []
        for incident in self.incidents:
            rows.append({
                "id": incident.identifier,
                "started": incident.started - self.session_started,
                "ended": (incident.ended - self.session_started
                          if incident.ended is not None else None),
                "signals": json.dumps(
                    [signal for signal in FOCUS_SIGNALS if signal in incident.signals],
                    ensure_ascii=False,
                ),
                "segments": json.dumps({
                    signal: [[start - self.session_started, end - self.session_started]
                             for start, end in values]
                    for signal, values in incident.segments.items()
                }, ensure_ascii=False, sort_keys=True),
                "violation_seconds": incident.violation_seconds,
                "alert_offset": (incident.alert_at - self.session_started
                                 if incident.alert_at is not None else None),
            })
        return tuple(rows)

