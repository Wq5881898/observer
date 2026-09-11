import unittest

from observer.observations import Observation, SignalValue
from observer.rules import ActionKind, RuleConfig, RulePhase, SustainedRule, default_rule_configs
from observer.scoring import RuleEngine


def observation(at: float, signal: SignalValue) -> Observation:
    return Observation(at, {"phone_held": signal})


class SustainedRuleTests(unittest.TestCase):
    def test_soft_defaults_alert_but_never_emit_penalties(self) -> None:
        for config in default_rule_configs():
            if config.rule_id not in ('attention_away', 'posture_deviation', 'ergonomic_risk'):
                continue
            with self.subTest(rule=config.rule_id):
                rule = SustainedRule(config)
                actions = []
                for second in range(181):
                    actions.extend(rule.update(Observation(float(second), {
                        config.signal_name: SignalValue.violation()
                    })))
                self.assertTrue(any(a.kind is ActionKind.ALERT for a in actions))
                self.assertFalse(any(a.kind is ActionKind.PENALTY for a in actions))
                self.assertEqual(sum(a.kind is ActionKind.ALERT for a in actions), 1)
                for second in range(181, 188):
                    actions.extend(rule.update(Observation(float(second), {
                        config.signal_name: SignalValue.compliant()
                    })))
                self.assertIs(rule.phase, RulePhase.CLEAR)

    def setUp(self) -> None:
        self.config = RuleConfig(
            "phone_held",
            "phone_held",
            trigger_after_seconds=3.0,
            recovery_after_seconds=2.0,
            unknown_grace_seconds=2.0,
            alert_cooldown_seconds=10.0,
            initial_penalty_points=5.0,
            repeat_penalty_points=2.0,
            repeat_penalty_interval_seconds=4.0,
            maximum_penalty_points=7.0,
        )

    def test_short_violation_does_not_activate(self) -> None:
        rule = SustainedRule(self.config)
        rule.update(observation(0.0, SignalValue.violation()))
        rule.update(observation(1.0, SignalValue.violation()))
        actions = rule.update(observation(2.0, SignalValue.compliant()))
        self.assertFalse(actions)
        self.assertIs(rule.phase, RulePhase.CLEAR)

    def test_sustained_violation_activates_alerts_and_penalizes(self) -> None:
        rule = SustainedRule(self.config)
        actions = ()
        for second in range(4):
            actions = rule.update(observation(float(second), SignalValue.violation()))
        self.assertEqual(
            [action.kind for action in actions],
            [ActionKind.ACTIVATED, ActionKind.ALERT, ActionKind.PENALTY],
        )
        self.assertIs(rule.phase, RulePhase.ACTIVE)

    def test_unknown_never_builds_a_violation(self) -> None:
        rule = SustainedRule(self.config)
        for second in range(10):
            actions = rule.update(observation(float(second), SignalValue.unknown("dark")))
            self.assertFalse(actions)
        self.assertIs(rule.phase, RulePhase.CLEAR)

    def test_recovery_requires_continuous_compliance(self) -> None:
        rule = SustainedRule(self.config)
        for second in range(4):
            rule.update(observation(float(second), SignalValue.violation()))
        self.assertFalse(rule.update(observation(4.0, SignalValue.compliant())))
        self.assertFalse(rule.update(observation(5.0, SignalValue.compliant())))
        actions = rule.update(observation(6.0, SignalValue.compliant()))
        self.assertEqual(actions[0].kind, ActionKind.RESOLVED)
        self.assertEqual(actions[0].reason, "recovered")

    def test_unknown_resolves_active_event_without_penalty(self) -> None:
        rule = SustainedRule(self.config)
        for second in range(4):
            rule.update(observation(float(second), SignalValue.violation()))
        rule.update(observation(4.0, SignalValue.unknown("camera lost")))
        rule.update(observation(5.0, SignalValue.unknown("camera lost")))
        actions = rule.update(observation(6.0, SignalValue.unknown("camera lost")))
        self.assertEqual(len(actions), 1)
        self.assertIs(actions[0].kind, ActionKind.RESOLVED)
        self.assertEqual(actions[0].reason, "observation_lost")

    def test_engine_applies_rule_cap(self) -> None:
        engine = RuleEngine((self.config,))
        for second in range(12):
            engine.process(observation(float(second), SignalValue.violation()))
        self.assertEqual(engine.ledger.total_points, 7.0)
        self.assertEqual(engine.ledger.score, 93.0)

    def test_ergonomic_event_is_recordable_without_efficiency_penalty(self) -> None:
        config = RuleConfig(
            "ergonomic_risk",
            "ergonomic_risk",
            trigger_after_seconds=3.0,
            initial_penalty_points=0.0,
            repeat_penalty_points=0.0,
            maximum_penalty_points=0.0,
        )
        engine = RuleEngine((config,))
        result = None
        for second in range(4):
            result = engine.process(
                Observation(
                    float(second),
                    {
                        "ergonomic_risk": SignalValue.violation(
                            reason="shoulder_imbalance"
                        )
                    },
                )
            )

        self.assertIsNotNone(result)
        self.assertIn(ActionKind.ACTIVATED, [action.kind for action in result.actions])
        self.assertEqual(engine.ledger.total_points, 0.0)
        self.assertEqual(engine.ledger.score, 100.0)

    def test_large_observation_gap_breaks_pending_continuity(self) -> None:
        rule = SustainedRule(self.config)
        rule.update(observation(0.0, SignalValue.violation()))
        actions = rule.update(observation(10.0, SignalValue.violation()))
        self.assertFalse(actions)
        self.assertIs(rule.phase, RulePhase.PENDING)

    def test_zero_unknown_grace_resolves_immediately(self) -> None:
        config = RuleConfig(
            "phone_held",
            "phone_held",
            trigger_after_seconds=0.0,
            unknown_grace_seconds=0.0,
        )
        rule = SustainedRule(config)
        rule.update(observation(0.0, SignalValue.violation()))
        actions = rule.update(observation(1.0, SignalValue.unknown("camera lost")))
        self.assertEqual(actions[0].reason, "observation_lost")


if __name__ == "__main__":
    unittest.main()
