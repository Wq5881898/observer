import unittest

from observer.live_validation import (
    evaluate_away_event_report,
    evaluate_brief_away_report,
    evaluate_phone_event_report,
)


def segment(scenario_id: str, *, started: float, compliant=0, violation=0, unknown=0):
    return {
        "scenario_id": scenario_id,
        "started_monotonic": started,
        "state_counts": {
            "compliant": compliant,
            "violation": violation,
            "unknown": unknown,
            "exempt": 0,
        },
        "longest_state_run_seconds": {
            "compliant": 4.0 if compliant else 0.0,
            "violation": 4.0 if violation else 0.0,
            "unknown": 4.0 if unknown else 0.0,
            "exempt": 0.0,
        },
    }


class AwayEventReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.segments = [
            segment("event_normal", started=0.0, compliant=80),
            segment("event_brief_away", started=10.0, violation=100),
            segment("event_brief_recovery", started=23.0, compliant=30),
            segment("event_sustained_away", started=30.0, violation=250),
            segment("event_sustained_recovery", started=58.0, compliant=40),
            segment("event_face_covered", started=65.0, compliant=70, unknown=10),
            segment("event_camera_covered", started=76.0, unknown=50),
        ]
        self.actions = [
            {
                "scenario_id": "event_sustained_away",
                "kind": "activated",
                "at_monotonic": 50.1,
                "event_started_monotonic": 30.0,
                "requested_points": 0.0,
            },
            {
                "scenario_id": "event_sustained_away",
                "kind": "alert",
                "at_monotonic": 50.1,
                "event_started_monotonic": 30.0,
                "requested_points": 0.0,
            },
            {
                "scenario_id": "event_sustained_away",
                "kind": "penalty",
                "at_monotonic": 50.1,
                "event_started_monotonic": 30.0,
                "requested_points": 3.0,
            },
            {
                "scenario_id": "event_sustained_recovery",
                "kind": "resolved",
                "at_monotonic": 60.1,
                "event_started_monotonic": 30.0,
                "requested_points": 0.0,
                "reason": "recovered",
            },
        ]

    def test_report_passes_complete_event_sequence(self) -> None:
        report = evaluate_away_event_report(
            self.segments,
            self.actions,
            trigger_after_seconds=20.0,
            recovery_after_seconds=2.0,
            expected_penalty_points=3.0,
            final_score=97.0,
        )
        self.assertTrue(report["passed"])
        self.assertAlmostEqual(report["activation_delay_seconds"], 20.1)
        self.assertAlmostEqual(report["recovery_delay_seconds"], 2.1)

    def test_report_rejects_alert_during_brief_away(self) -> None:
        actions = [
            *self.actions,
            {
                "scenario_id": "event_brief_away",
                "kind": "activated",
                "at_monotonic": 15.0,
                "event_started_monotonic": 10.0,
                "requested_points": 0.0,
            },
        ]
        report = evaluate_away_event_report(
            self.segments,
            actions,
            trigger_after_seconds=20.0,
            recovery_after_seconds=2.0,
            expected_penalty_points=3.0,
            final_score=97.0,
        )
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["brief_away_no_activation"])

    def test_report_rejects_unperformed_brief_away_input(self) -> None:
        self.segments[1] = segment(
            "event_brief_away", started=10.0, compliant=78, violation=51
        )
        report = evaluate_away_event_report(
            self.segments,
            self.actions,
            trigger_after_seconds=20.0,
            recovery_after_seconds=2.0,
            expected_penalty_points=3.0,
            final_score=97.0,
        )
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["brief_away_input_valid"])
        self.assertAlmostEqual(
            report["state_ratios"]["event_brief_away"]["violation"],
            51 / 129,
        )

    def test_focused_brief_away_report_requires_real_input_and_no_penalty(self) -> None:
        segments = self.segments[:3]
        report = evaluate_brief_away_report(segments, [], final_score=100.0)
        self.assertTrue(report["passed"])

        segments[1] = segment(
            "event_brief_away", started=10.0, compliant=30, violation=70
        )
        report = evaluate_brief_away_report(segments, [], final_score=100.0)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["brief_away_input_valid"])


class PhoneEventReportTests(unittest.TestCase):
    def test_complete_phone_event_sequence_passes(self) -> None:
        segments = [
            segment("phone_event_normal", started=0.0, compliant=50),
            segment("phone_event_brief", started=8.0, violation=60),
            segment("phone_event_brief_recovery", started=17.0, compliant=40),
            segment("phone_event_sustained", started=24.0, violation=120),
            segment("phone_event_sustained_recovery", started=39.0, compliant=40),
            segment("phone_event_non_phone", started=46.0, compliant=110, violation=10),
        ]
        actions = [
            {
                "scenario_id": "phone_event_sustained",
                "kind": kind,
                "at_monotonic": 34.1,
                "event_started_monotonic": 24.0,
                "requested_points": 5.0 if kind == "penalty" else 0.0,
            }
            for kind in ("activated", "alert", "penalty")
        ]
        actions.append(
            {
                "scenario_id": "phone_event_sustained_recovery",
                "kind": "resolved",
                "at_monotonic": 41.1,
                "event_started_monotonic": 24.0,
                "requested_points": 0.0,
                "reason": "recovered",
            }
        )
        report = evaluate_phone_event_report(
            segments,
            actions,
            trigger_after_seconds=10.0,
            recovery_after_seconds=2.0,
            expected_penalty_points=5.0,
            final_score=95.0,
        )
        self.assertTrue(report["passed"])
        self.assertAlmostEqual(report["activation_delay_seconds"], 10.1)
        self.assertAlmostEqual(report["recovery_delay_seconds"], 2.1)

    def test_phone_event_accepts_natural_unknown_recovery(self) -> None:
        segments = [
            segment("phone_event_brief", started=0.0, violation=60),
            segment("phone_event_brief_recovery", started=11.0, unknown=40),
            segment("phone_event_sustained", started=18.0, violation=120),
            segment("phone_event_sustained_recovery", started=36.0, unknown=40),
        ]
        actions = [
            {
                "scenario_id": "phone_event_sustained",
                "kind": kind,
                "at_monotonic": 28.0,
                "event_started_monotonic": 18.0,
                "requested_points": 5.0 if kind == "penalty" else 0.0,
            }
            for kind in ("activated", "alert", "penalty")
        ]
        actions.append(
            {
                "scenario_id": "phone_event_sustained_recovery",
                "kind": "resolved",
                "at_monotonic": 39.0,
                "event_started_monotonic": 18.0,
                "requested_points": 0.0,
                "reason": "observation_lost",
            }
        )
        report = evaluate_phone_event_report(
            segments,
            actions,
            trigger_after_seconds=10.0,
            recovery_after_seconds=2.0,
            unknown_grace_seconds=3.0,
            expected_penalty_points=5.0,
            final_score=95.0,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["recovery_reason"], "observation_lost")

    def test_phone_event_rejects_non_phone_activation(self) -> None:
        segments = [
            segment("phone_event_normal", started=0.0, compliant=50),
            segment("phone_event_brief", started=8.0, violation=60),
            segment("phone_event_brief_recovery", started=17.0, compliant=40),
            segment("phone_event_sustained", started=24.0, violation=120),
            segment("phone_event_sustained_recovery", started=39.0, compliant=40),
            segment("phone_event_non_phone", started=46.0, compliant=110, violation=10),
        ]
        actions = [
            {
                "scenario_id": "phone_event_non_phone",
                "kind": "activated",
                "at_monotonic": 56.0,
                "event_started_monotonic": 46.0,
                "requested_points": 0.0,
            }
        ]
        report = evaluate_phone_event_report(
            segments,
            actions,
            trigger_after_seconds=10.0,
            recovery_after_seconds=2.0,
            expected_penalty_points=5.0,
            final_score=95.0,
        )
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["non_phone_no_activation"])


if __name__ == "__main__":
    unittest.main()
