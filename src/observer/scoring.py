from __future__ import annotations

from dataclasses import dataclass

from observer.observations import Observation
from observer.rules import ActionKind, RuleAction, RuleConfig, RulePhase, SustainedRule


@dataclass(frozen=True, slots=True)
class PenaltyEntry:
    rule_id: str
    event_id: str
    at_monotonic: float
    points: float
    observed_violation_seconds: float


class PenaltyLedger:
    def __init__(self, baseline_score: float = 100.0) -> None:
        if baseline_score <= 0:
            raise ValueError("baseline score must be positive")
        self.baseline_score = baseline_score
        self._entries: list[PenaltyEntry] = []

    @property
    def entries(self) -> tuple[PenaltyEntry, ...]:
        return tuple(self._entries)

    @property
    def total_points(self) -> float:
        return sum(entry.points for entry in self._entries)

    @property
    def score(self) -> float:
        return max(0.0, self.baseline_score - self.total_points)

    def apply(self, action: RuleAction, config: RuleConfig) -> PenaltyEntry | None:
        if action.kind is not ActionKind.PENALTY or action.requested_points <= 0:
            return None
        already_applied = sum(
            entry.points for entry in self._entries if entry.rule_id == config.rule_id
        )
        remaining = max(0.0, config.maximum_penalty_points - already_applied)
        points = min(action.requested_points, remaining)
        if points <= 0:
            return None
        entry = PenaltyEntry(
            rule_id=action.rule_id,
            event_id=action.event_id,
            at_monotonic=action.at_monotonic,
            points=points,
            observed_violation_seconds=action.observed_violation_seconds,
        )
        self._entries.append(entry)
        return entry


@dataclass(frozen=True, slots=True)
class EngineResult:
    actions: tuple[RuleAction, ...]
    penalties: tuple[PenaltyEntry, ...]
    score: float


class RuleEngine:
    def __init__(self, configs: tuple[RuleConfig, ...], baseline_score: float = 100.0) -> None:
        identifiers = [config.rule_id for config in configs]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("rule ids must be unique")
        self._configs = {config.rule_id: config for config in configs}
        self._rules = {config.rule_id: SustainedRule(config) for config in configs}
        self._next_group_alert_at: dict[str, float] = {}
        self.ledger = PenaltyLedger(baseline_score)

    @property
    def phases(self) -> dict[str, RulePhase]:
        return {rule_id: rule.phase for rule_id, rule in self._rules.items()}

    def process(self, observation: Observation) -> EngineResult:
        actions: list[RuleAction] = []
        penalties: list[PenaltyEntry] = []
        for rule_id, rule in self._rules.items():
            rule_actions = rule.update(observation)
            for action in rule_actions:
                config = self._configs[rule_id]
                if action.kind is ActionKind.ALERT and config.alert_group:
                    next_at = self._next_group_alert_at.get(config.alert_group, 0.0)
                    if action.at_monotonic < next_at:
                        continue
                    self._next_group_alert_at[config.alert_group] = (
                        action.at_monotonic + config.alert_cooldown_seconds
                    )
                actions.append(action)
                entry = self.ledger.apply(action, self._configs[rule_id])
                if entry is not None:
                    penalties.append(entry)
        return EngineResult(tuple(actions), tuple(penalties), self.ledger.score)
