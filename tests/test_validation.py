import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from observer.live_validation import (
    SCENARIOS,
    Scenario,
    attention_control_action_matches,
    attention_direction_matches,
    _wrap_display_line,
    scenario_accepted_states,
    scenario_acceptance_summary,
    scenario_capture_status,
    scenario_distance_status,
    scenario_record_passes,
    scenario_validation_signal,
)
from observer.observations import SignalState, SignalValue
from observer.perception import FeatureVector
from observer.validation import (
    ErrorEvidenceStore,
    EvidencePolicy,
    TrialRecord,
    read_trials,
    summarize_trials,
    purge_expired_evidence,
    new_batch_manifest,
)


def trial(trial_id: str, truth: str, prediction: str) -> TrialRecord:
    return TrialRecord(
        trial_id=trial_id,
        scenario_id="synthetic",
        signal_name="phone_held",
        dataset_role="holdout",
        truth_state=truth,
        predicted_state=prediction,
        started_monotonic=1.0,
        ended_monotonic=11.0,
        observable_fraction=1.0 if prediction != "unknown" else 0.0,
    )


class ValidationTests(unittest.TestCase):
    def test_attention_boundary_distance_gate_guides_near_and_calibration_positions(self) -> None:
        near = next(s for s in SCENARIOS if s.scenario_id == "attention_boundary_a06_near_left")
        far = next(s for s in SCENARIOS if s.scenario_id == "attention_boundary_a08_far_center")
        probe = SimpleNamespace(
            latest_smoothed_attention_features=FeatureVector(
                "attention", {"face_scale": 0.38}
            )
        )

        self.assertTrue(scenario_distance_status(near, probe, 0.27)[0])
        self.assertFalse(scenario_distance_status(far, probe, 0.27)[0])
        probe.latest_smoothed_attention_features = FeatureVector(
            "attention", {"face_scale": 0.28}
        )
        self.assertFalse(scenario_distance_status(near, probe, 0.27)[0])
        self.assertTrue(scenario_distance_status(far, probe, 0.27)[0])

        probe.latest_smoothed_attention_features = FeatureVector(
            "attention", {"face_scale": 0.49}
        )
        self.assertFalse(scenario_distance_status(near, probe, 0.27, 0.42)[0])
        probe.latest_smoothed_attention_features = FeatureVector(
            "attention", {"face_scale": 0.51}
        )
        self.assertTrue(scenario_distance_status(near, probe, 0.27, 0.42)[0])

    def test_attention_boundary_uses_interactive_single_repetition_scenarios(self) -> None:
        scenarios = [
            scenario
            for scenario in SCENARIOS
            if scenario.scenario_id.startswith("attention_boundary_")
        ]

        self.assertEqual(len(scenarios), 11)
        self.assertTrue(all(scenario.repetitions == 1 for scenario in scenarios))
        self.assertEqual(scenarios[1].duration_seconds, 12.0)
        self.assertEqual(scenarios[5].distance_mode, "near")
        self.assertEqual(scenarios[7].distance_mode, "calibration")

    def test_attention_boundary_shows_baseline_acceptance_standard(self) -> None:
        scenarios = {scenario.scenario_id: scenario for scenario in SCENARIOS}
        self.assertIn(
            "continuous >=8s",
            scenario_acceptance_summary(
                scenarios["attention_boundary_a02_outside_left"]
            ),
        )
        self.assertIn(
            "recover <=2.5s",
            scenario_acceptance_summary(
                scenarios["attention_boundary_a03_recovery"]
            ),
        )
        self.assertIn(
            "violation <=10%",
            scenario_acceptance_summary(
                scenarios["attention_boundary_a06_near_left"]
            ),
        )

    def test_capture_ready_is_independent_of_model_target_prediction(self) -> None:
        scenario = next(
            item
            for item in SCENARIOS
            if item.scenario_id == "attention_compensation_d10_near_right"
        )
        probe = SimpleNamespace(
            latest_quality=SimpleNamespace(observable=True, reasons=()),
            latest_smoothed_attention_features=FeatureVector(
                "attention", {"face_scale": 0.51}
            ),
        )

        ready, status = scenario_capture_status(
            scenario,
            probe,
            distance_matches=True,
            distance_status="DISTANCE: ready",
        )

        self.assertTrue(ready)
        self.assertIn("CAPTURE", status)

    def test_compensation_diagnostic_has_paired_controls_and_positives(self) -> None:
        scenarios = [
            scenario
            for scenario in SCENARIOS
            if scenario.scenario_id.startswith("attention_compensation_")
        ]

        self.assertEqual(len(scenarios), 11)
        self.assertTrue(all(scenario.repetitions == 1 for scenario in scenarios))
        self.assertEqual(
            [scenario.truth_state for scenario in scenarios[1:6]],
            [
                SignalState.COMPLIANT,
                SignalState.VIOLATION,
                SignalState.COMPLIANT,
                SignalState.COMPLIANT,
                SignalState.VIOLATION,
            ],
        )
        self.assertTrue(all(scenario.distance_mode == "near" for scenario in scenarios[7:10]))

    def test_attention_baseline_report_applies_fraction_and_continuity_standard(self) -> None:
        control = next(
            scenario
            for scenario in SCENARIOS
            if scenario.scenario_id == "attention_compensation_d02_head_left_eyes_center"
        )
        positive = next(
            scenario
            for scenario in SCENARIOS
            if scenario.scenario_id == "attention_compensation_d03_outside_left"
        )
        unstable_control = TrialRecord(
            trial_id="control",
            scenario_id=control.scenario_id,
            signal_name="attention_away",
            dataset_role="tuning",
            truth_state="compliant",
            predicted_state="compliant",
            started_monotonic=1.0,
            ended_monotonic=9.0,
            observable_fraction=0.53,
            metrics={
                "compliant_fraction": 0.51,
                "violation_fraction": 0.02,
                "unknown_fraction": 0.47,
                "longest_violation_run_seconds": 0.0,
            },
        )
        interrupted_positive = TrialRecord(
            trial_id="positive",
            scenario_id=positive.scenario_id,
            signal_name="attention_away",
            dataset_role="tuning",
            truth_state="violation",
            predicted_state="violation",
            started_monotonic=1.0,
            ended_monotonic=11.0,
            observable_fraction=1.0,
            metrics={
                "compliant_fraction": 0.05,
                "violation_fraction": 0.95,
                "unknown_fraction": 0.0,
                "longest_violation_run_seconds": 6.0,
            },
        )

        self.assertFalse(scenario_record_passes(control, unstable_control))
        self.assertFalse(scenario_record_passes(positive, interrupted_positive))

    def test_phone_baseline_has_formal_interactive_scenarios(self) -> None:
        phone_scenarios = [
            scenario for scenario in SCENARIOS if scenario.scenario_id.startswith("phone_baseline_b")
        ]
        attention_b08 = next(
            scenario
            for scenario in SCENARIOS
            if scenario.scenario_id == "attention_baseline_b08_look_off_screen"
        )

        self.assertEqual(
            [scenario.scenario_id for scenario in phone_scenarios],
            [
                "phone_baseline_b01_natural_no_phone",
                "phone_baseline_b02_desktop_phone",
                "phone_baseline_b03_non_phone_object",
                "phone_baseline_b04_one_hand_phone",
                "phone_baseline_b05_recovery",
                "phone_baseline_b06_two_hands_low",
                "phone_baseline_b07_recovery",
                "phone_baseline_b08_phone_out_of_frame",
                "phone_baseline_b09_final",
            ],
        )
        self.assertEqual(phone_scenarios[3].duration_seconds, 12.0)
        self.assertEqual(attention_b08.duration_seconds, 12.0)
        self.assertEqual(
            scenario_acceptance_summary(phone_scenarios[3]),
            "STANDARD: violation >=90%; continuous >=10s",
        )

    def test_phone_baseline_applies_positive_negative_and_safe_unknown_rules(self) -> None:
        positive = next(
            scenario
            for scenario in SCENARIOS
            if scenario.scenario_id == "phone_baseline_b04_one_hand_phone"
        )
        negative = next(
            scenario
            for scenario in SCENARIOS
            if scenario.scenario_id == "phone_baseline_b02_desktop_phone"
        )
        low_two_hands = next(
            scenario
            for scenario in SCENARIOS
            if scenario.scenario_id == "phone_baseline_b06_two_hands_low"
        )

        def record(metrics: dict[str, float]) -> TrialRecord:
            return TrialRecord(
                trial_id="sample",
                scenario_id="sample",
                signal_name="phone_held",
                dataset_role="tuning",
                truth_state="violation",
                predicted_state="violation",
                started_monotonic=1.0,
                ended_monotonic=13.0,
                observable_fraction=1.0,
                metrics=metrics,
            )

        self.assertTrue(
            scenario_record_passes(
                positive,
                record(
                    {
                        "violation_fraction": 0.95,
                        "longest_violation_run_seconds": 10.1,
                    }
                ),
            )
        )
        self.assertFalse(
            scenario_record_passes(
                positive,
                record(
                    {
                        "violation_fraction": 0.95,
                        "longest_violation_run_seconds": 9.8,
                    }
                ),
            )
        )
        self.assertFalse(
            scenario_record_passes(
                negative,
                record(
                    {
                        "violation_fraction": 0.05,
                        "longest_violation_run_seconds": 2.2,
                    }
                ),
            )
        )
        self.assertTrue(
            scenario_record_passes(
                low_two_hands,
                record(
                    {
                        "violation_fraction": 0.05,
                        "unknown_fraction": 0.95,
                    }
                ),
            )
        )

    def test_posture_baseline_has_formal_interactive_scenarios(self) -> None:
        posture_scenarios = [
            scenario
            for scenario in SCENARIOS
            if scenario.scenario_id.startswith("posture_ergonomics_c")
        ]
        attention_scenarios = [
            scenario
            for scenario in SCENARIOS
            if scenario.scenario_id.startswith("attention_ergonomics_c")
        ]

        self.assertEqual(
            [scenario.scenario_id for scenario in posture_scenarios],
            [
                "posture_ergonomics_c01_natural",
                "posture_ergonomics_c02_near_natural",
                "posture_ergonomics_c03_forward_slouch",
                "posture_ergonomics_c04_recovery",
                "posture_ergonomics_c05_side_or_tilt",
                "posture_ergonomics_c06_shoulder_unknown",
                "posture_ergonomics_c08_final",
            ],
        )
        self.assertEqual(
            [scenario.scenario_id for scenario in attention_scenarios],
            [
                "attention_ergonomics_c06_attention",
                "attention_ergonomics_c07_small_chin",
            ],
        )
        self.assertTrue(all(scenario.repetitions == 1 for scenario in posture_scenarios))
        self.assertEqual(posture_scenarios[0].duration_seconds, 15.0)
        self.assertEqual(posture_scenarios[5].truth_state, SignalState.UNKNOWN)

    def test_posture_baseline_applies_strict_state_and_run_rules(self) -> None:
        near_natural = next(
            scenario
            for scenario in SCENARIOS
            if scenario.scenario_id == "posture_ergonomics_c02_near_natural"
        )
        shoulder_unknown = next(
            scenario
            for scenario in SCENARIOS
            if scenario.scenario_id == "posture_ergonomics_c06_shoulder_unknown"
        )

        def record(metrics: dict[str, float]) -> TrialRecord:
            return TrialRecord(
                trial_id="sample",
                scenario_id="sample",
                signal_name="posture_deviation",
                dataset_role="tuning",
                truth_state="compliant",
                predicted_state="compliant",
                started_monotonic=1.0,
                ended_monotonic=16.0,
                observable_fraction=1.0,
                metrics=metrics,
            )

        self.assertTrue(
            scenario_record_passes(
                near_natural,
                record(
                    {
                        "compliant_fraction": 1.0,
                        "violation_fraction": 0.0,
                        "unknown_fraction": 0.0,
                        "longest_violation_run_seconds": 0.0,
                    }
                ),
            )
        )
        self.assertFalse(
            scenario_record_passes(
                near_natural,
                record(
                    {
                        "compliant_fraction": 0.95,
                        "violation_fraction": 0.05,
                        "unknown_fraction": 0.0,
                        "longest_violation_run_seconds": 2.1,
                    }
                ),
            )
        )
        self.assertTrue(
            scenario_record_passes(
                shoulder_unknown,
                record(
                    {
                        "compliant_fraction": 0.0,
                        "violation_fraction": 0.0,
                        "unknown_fraction": 1.0,
                    }
                ),
            )
        )
        self.assertFalse(
            scenario_record_passes(
                shoulder_unknown,
                record(
                    {
                        "compliant_fraction": 0.0,
                        "violation_fraction": 0.20,
                        "unknown_fraction": 0.80,
                    }
                ),
            )
        )

    def test_overlay_text_wraps_inside_frame_width(self) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        line = "B08 ATTENTION: repeat hold; keep eyes on the phone off screen"

        rendered = _wrap_display_line(frame, line, 0.46)

        self.assertGreater(len(rendered), 1)
        import cv2

        for item in rendered:
            width = cv2.getTextSize(
                item, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1
            )[0][0]
            self.assertLessEqual(width, 320 - 28)

    def test_scenario_can_accept_safe_unknown_without_accepting_violation(self) -> None:
        scenario = Scenario(
            "natural",
            "phone_held",
            SignalState.COMPLIANT,
            "work naturally",
            accepted_states=(SignalState.COMPLIANT, SignalState.UNKNOWN),
        )
        accepted = scenario_accepted_states(scenario)
        self.assertIn(SignalState.COMPLIANT, accepted)
        self.assertIn(SignalState.UNKNOWN, accepted)
        self.assertNotIn(SignalState.VIOLATION, accepted)

    def test_directional_attention_requires_requested_direction(self) -> None:
        desk = next(
            scenario for scenario in SCENARIOS if scenario.scenario_id == "attention_desk"
        )
        right_only = SignalValue.violation(
            0.9,
            attention_head_yaw_delta=-0.30,
            attention_head_pitch_delta=0.01,
            attention_screen_x=1.2,
            attention_screen_y=0.5,
        )

        self.assertFalse(attention_direction_matches(desk, right_only))
        self.assertIs(
            scenario_validation_signal(desk, right_only).state,
            SignalState.COMPLIANT,
        )

        false_vertical_from_turn = SignalValue.violation(
            0.9,
            attention_head_yaw_delta=-0.50,
            attention_head_pitch_delta=0.05,
            attention_screen_x=0.23,
            attention_screen_y=1.17,
        )
        small_chin_drop = SignalValue.violation(
            0.7,
            attention_head_yaw_delta=-0.03,
            attention_head_pitch_delta=0.09,
            attention_screen_x=0.20,
            attention_screen_y=0.34,
        )

        self.assertFalse(attention_direction_matches(desk, false_vertical_from_turn))
        self.assertFalse(attention_direction_matches(desk, small_chin_drop))

        chin_control = next(
            scenario
            for scenario in SCENARIOS
            if scenario.scenario_id == "attention_desk_control_chin"
        )
        upward_wrong_action = SignalValue.compliant(
            0.9,
            attention_head_yaw_delta=0.01,
            attention_head_pitch_delta=-0.08,
            attention_screen_x=1.65,
            attention_screen_y=0.65,
        )
        self.assertFalse(
            attention_control_action_matches(chin_control, upward_wrong_action)
        )
        self.assertIs(
            scenario_validation_signal(chin_control, upward_wrong_action).state,
            SignalState.UNKNOWN,
        )
        self.assertTrue(
            attention_control_action_matches(chin_control, small_chin_drop)
        )
        self.assertIs(
            scenario_validation_signal(chin_control, small_chin_drop).state,
            SignalState.COMPLIANT,
        )

        side_control = next(
            scenario
            for scenario in SCENARIOS
            if scenario.scenario_id == "attention_desk_control_side"
        )
        self.assertIs(
            scenario_validation_signal(side_control, false_vertical_from_turn).state,
            SignalState.COMPLIANT,
        )
        for scenario_id, yaw in (
            ("attention_desk_control_left", 0.50),
            ("attention_desk_control_right", -0.50),
        ):
            side_control = next(
                scenario
                for scenario in SCENARIOS
                if scenario.scenario_id == scenario_id
            )
            matching_turn = SignalValue.violation(
                0.9,
                attention_head_yaw_delta=yaw,
                attention_head_pitch_delta=0.05,
                attention_screen_x=0.23,
                attention_screen_y=1.17,
            )
            self.assertIs(
                scenario_validation_signal(
                    side_control, matching_turn
                ).state,
                SignalState.COMPLIANT,
            )

        centered = SignalValue.compliant(
            0.9,
            attention_head_yaw_delta=0.0,
            attention_head_pitch_delta=0.0,
            attention_screen_x=0.5,
            attention_screen_y=0.5,
        )
        left_control = next(
            scenario
            for scenario in SCENARIOS
            if scenario.scenario_id == "attention_desk_control_left"
        )
        self.assertFalse(attention_control_action_matches(left_control, centered))
        self.assertIs(
            scenario_validation_signal(left_control, centered).state,
            SignalState.UNKNOWN,
        )

    def test_directional_attention_accepts_matching_head_evidence(self) -> None:
        left = next(
            scenario
            for scenario in SCENARIOS
            if scenario.scenario_id == "attention_outside_left"
        )
        left_turn = SignalValue.violation(
            0.8,
            attention_head_yaw_delta=0.25,
            attention_head_pitch_delta=0.01,
            attention_screen_x=0.4,
            attention_screen_y=0.5,
        )

        self.assertTrue(attention_direction_matches(left, left_turn))
        self.assertIs(
            scenario_validation_signal(left, left_turn).state,
            SignalState.VIOLATION,
        )

    def test_directional_attention_rejects_near_edge_early_trigger(self) -> None:
        right = next(
            scenario
            for scenario in SCENARIOS
            if scenario.scenario_id == "attention_outside_right"
        )
        near_edge = SignalValue.violation(
            0.6,
            attention_head_yaw_delta=-0.18,
            attention_head_pitch_delta=0.01,
            attention_screen_x=1.05,
            attention_screen_y=0.5,
        )

        self.assertFalse(attention_direction_matches(right, near_edge))
        self.assertIs(
            scenario_validation_signal(right, near_edge).state,
            SignalState.COMPLIANT,
        )

    def test_manifest_records_same_session_alignment(self) -> None:
        manifest = new_batch_manifest(
            batch_id="tuning-test",
            dataset_role="tuning",
            git_commit="abc",
            config_sha256="config",
            model_manifest_sha256="models",
            calibration_profile_id="profile",
            environment={"camera_index": 0},
            alignment={"final_context_z": 1.8, "frames_saved": 0},
        )

        self.assertEqual(manifest.alignment["final_context_z"], 1.8)
        self.assertEqual(manifest.alignment["frames_saved"], 0)

    def test_metrics_count_unknown_positive_as_false_negative(self) -> None:
        metrics = summarize_trials(
            [
                trial("tp", "violation", "violation"),
                trial("fn", "violation", "unknown"),
                trial("tn", "compliant", "compliant"),
                trial("fp", "compliant", "violation"),
            ],
            "phone_held",
        )
        self.assertEqual(metrics.true_positive, 1)
        self.assertEqual(metrics.false_negative, 1)
        self.assertEqual(metrics.false_positive, 1)
        self.assertAlmostEqual(metrics.precision, 0.5)
        self.assertAlmostEqual(metrics.recall, 0.5)

    def test_metrics_expose_expected_unknown_safety_failure(self) -> None:
        metrics = summarize_trials(
            [
                trial("covered", "unknown", "violation"),
                trial("dark", "unknown", "unknown"),
            ],
            "phone_held",
        )

        self.assertEqual(metrics.expected_unknown_trials, 2)
        self.assertEqual(metrics.expected_unknown_failures, 1)

    def test_jsonl_rejects_invalid_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trials.jsonl"
            path.write_text('{"truth_state":"made-up"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1"):
                read_trials(path)

    def test_evidence_is_opt_in_and_one_frame_per_trial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            frame = np.zeros((20, 20, 3), dtype=np.uint8)
            disabled = ErrorEvidenceStore(Path(temporary) / "off")
            self.assertIsNone(disabled.save("t1", frame, is_error=True))
            enabled = ErrorEvidenceStore(
                Path(temporary) / "on", EvidencePolicy(enabled=True)
            )
            saved = enabled.save("t1", frame, is_error=True)
            self.assertIsNotNone(saved)
            self.assertIsNone(enabled.save("t1", frame, is_error=True))
            self.assertEqual(len(list((Path(temporary) / "on").glob("*.jpg"))), 1)

    def test_evidence_purge_is_dry_run_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            image = directory / "expired.jpg"
            image.write_bytes(b"test")
            index = directory / "index.jsonl"
            index.write_text(
                '{"relative_path":"expired.jpg","delete_after_utc":"2020-01-01T00:00:00+00:00"}\n',
                encoding="utf-8",
            )
            preview = purge_expired_evidence(
                index, now_utc=datetime(2026, 1, 1, tzinfo=timezone.utc)
            )
            self.assertTrue(image.exists())
            self.assertEqual(preview["expired"], ["expired.jpg"])
            purge_expired_evidence(
                index, apply=True, now_utc=datetime(2026, 1, 1, tzinfo=timezone.utc)
            )
            self.assertFalse(image.exists())


if __name__ == "__main__":
    unittest.main()
