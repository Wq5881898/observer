from __future__ import annotations

from dataclasses import dataclass

from observer.observations import Observation
from observer.rules import ActionKind, RuleAction, RuleConfig, SustainedRule


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
        self.ledger = PenaltyLedger(baseline_score)

    def process(self, observation: Observation) -> EngineResult:
        actions: list[RuleAction] = []
        penalties: list[PenaltyEntry] = []
        for rule_id, rule in self._rules.items():
            rule_actions = rule.update(observation)
            actions.extend(rule_actions)
            for action in rule_actions:
                entry = self.ledger.apply(action, self._configs[rule_id])
                if entry is not None:
                    penalties.append(entry)
        return EngineResult(tuple(actions), tuple(penalties), self.ledger.score)

