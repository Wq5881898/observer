import unittest

from observer.live_validation import evaluate_away_event_report, evaluate_brief_away_report


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


if __name__ == "__main__":
    unittest.main()
