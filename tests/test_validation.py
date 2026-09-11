import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import observer.live_validation as live_validation

from observer.live_validation import (
    PHONE_EDGE_PARAMETER_PROFILES,
    SCENARIOS,
    Scenario,
    _TUNING_DIAGNOSTIC_ONLY_SCENARIOS,
    _git_commit,
    attention_control_action_matches,
    attention_direction_matches,
    _wrap_display_line,
    scenario_accepted_states,
    scenario_acceptance_summary,
    scenario_capture_status,
    scenario_distance_status,
    scenario_record_passes,
    scenario_target_ready,
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
    def test_git_commit_uses_the_source_repository_not_a_machine_path(self) -> None:
        with patch("observer.live_validation.subprocess.run") as run:
            run.return_value = SimpleNamespace(stdout="abc123\n")

            self.assertEqual(_git_commit(), "abc123")

        command = run.call_args.args[0]
        repository = Path(live_validation.__file__).resolve().parents[2]
        self.assertEqual(command[command.index("-C") + 1], str(repository))
        self.assertIn(f"safe.directory={repository.as_posix()}", command)

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

    def test_large_target_badge_resets_when_model_target_is_withdrawn(self) -> None:
        self.assertTrue(
            scenario_target_ready(
                capture_matches=True,
                model_matches=True,
                stable_seconds=1.0,
            )
        )
        self.assertFalse(
            scenario_target_ready(
                capture_matches=True,
                model_matches=False,
                stable_seconds=1.0,
            )
        )

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
                "phone_baseline_b10_one_hand_low",
                "phone_baseline_b10_recovery",
                "phone_baseline_b11_two_hands_normal",
                "phone_baseline_b11_recovery",
                "phone_baseline_b12_one_hand_edge",
                "phone_baseline_b12_recovery",
                "phone_baseline_b13_two_hands_edge",
                "phone_baseline_b13_recovery",
            ],
        )
        self.assertEqual(phone_scenarios[3].duration_seconds, 12.0)
        self.assertEqual(attention_b08.duration_seconds, 12.0)
        self.assertEqual(
            scenario_acceptance_summary(phone_scenarios[3]),
            "STANDARD: violation >=90%; continuous >=10s",
        )

    def test_phone_visibility_matrix_covers_hand_count_and_height(self) -> None:
        scenarios = {scenario.scenario_id: scenario for scenario in SCENARIOS}
        matrix = {
            "phone_baseline_b04_one_hand_phone": ("one", "phone"),
            "phone_baseline_b06_two_hands_low": ("two", "low"),
            "phone_baseline_b10_one_hand_low": ("one", "low"),
            "phone_baseline_b11_two_hands_normal": ("two", "normal"),
            "phone_baseline_b12_one_hand_edge": ("one", "edge"),
            "phone_baseline_b13_two_hands_edge": ("two", "edge"),
        }
        recoveries = {
            "phone_baseline_b04_one_hand_phone": "phone_baseline_b05_recovery",
            "phone_baseline_b06_two_hands_low": "phone_baseline_b07_recovery",
            "phone_baseline_b10_one_hand_low": "phone_baseline_b10_recovery",
            "phone_baseline_b11_two_hands_normal": "phone_baseline_b11_recovery",
            "phone_baseline_b12_one_hand_edge": "phone_baseline_b12_recovery",
            "phone_baseline_b13_two_hands_edge": "phone_baseline_b13_recovery",
        }
        self.assertEqual(set(matrix), set(scenarios) & set(matrix))
        for scenario_id, (hand_count, height) in matrix.items():
            scenario = scenarios[scenario_id]
            self.assertIn(hand_count, scenario.scenario_id)
            self.assertIn(height, scenario.scenario_id)
            self.assertIs(scenario.truth_state, SignalState.VIOLATION)
            self.assertEqual(
                scenario_accepted_states(scenario), (SignalState.VIOLATION,)
            )
            self.assertEqual(
                scenario_acceptance_summary(scenario),
                "STANDARD: violation >=90%; continuous >=10s",
            )
            self.assertIn(recoveries[scenario_id], scenarios)

    def test_all_phone_visibility_positives_use_strict_event_line(self) -> None:
        positive_ids = {
            "phone_baseline_b02_desktop_phone",
            "phone_baseline_b04_one_hand_phone",
            "phone_baseline_b06_two_hands_low",
            "phone_baseline_b10_one_hand_low",
            "phone_baseline_b11_two_hands_normal",
            "phone_baseline_b12_one_hand_edge",
            "phone_baseline_b13_two_hands_edge",
        }
        for scenario_id in positive_ids:
            scenario = next(
                item for item in SCENARIOS if item.scenario_id == scenario_id
            )
            passed = scenario_record_passes(
                scenario,
                TrialRecord(
                    trial_id=scenario_id,
                    scenario_id=scenario_id,
                    signal_name="phone_held",
                    dataset_role="tuning",
                    truth_state=SignalState.VIOLATION,
                    predicted_state=SignalState.VIOLATION,
                    started_monotonic=1.0,
                    ended_monotonic=13.0,
                    observable_fraction=1.0,
                    metrics={
                        "violation_fraction": 0.95,
                        "longest_violation_run_seconds": 10.1,
                    },
                ),
            )
            self.assertTrue(passed, scenario_id)

    def test_extended_phone_matrix_is_explicit_diagnostic_only(self) -> None:
        matrix_ids = {
            "phone_baseline_b10_one_hand_low",
            "phone_baseline_b10_recovery",
            "phone_baseline_b11_two_hands_normal",
            "phone_baseline_b11_recovery",
            "phone_baseline_b12_one_hand_edge",
            "phone_baseline_b12_recovery",
            "phone_baseline_b13_two_hands_edge",
            "phone_baseline_b13_recovery",
        }
        core_ids = {
            "phone_baseline_b01_natural_no_phone",
            "phone_baseline_b02_desktop_phone",
            "phone_baseline_b03_non_phone_object",
            "phone_baseline_b04_one_hand_phone",
            "phone_baseline_b05_recovery",
            "phone_baseline_b06_two_hands_low",
            "phone_baseline_b07_recovery",
            "phone_baseline_b08_phone_out_of_frame",
            "phone_baseline_b09_final",
        }
        self.assertTrue(matrix_ids.issubset(_TUNING_DIAGNOSTIC_ONLY_SCENARIOS))
        self.assertFalse(matrix_ids & core_ids)

    def test_phone_edge_matrix_is_separate_from_existing_baselines(self) -> None:
        edge_ids = {
            "phone_edge_e01_bottom_center",
            "phone_edge_e01_bottom_center_recovery",
            "phone_edge_e02_bottom_left",
            "phone_edge_e02_bottom_left_recovery",
            "phone_edge_e03_bottom_right",
            "phone_edge_e03_bottom_right_recovery",
            "phone_edge_e04_top_edge",
            "phone_edge_e04_top_edge_recovery",
        }
        scenarios = {scenario.scenario_id: scenario for scenario in SCENARIOS}
        self.assertTrue(edge_ids.issubset(scenarios))
        self.assertTrue(edge_ids.issubset(_TUNING_DIAGNOSTIC_ONLY_SCENARIOS))
        self.assertTrue(
            all(scenarios[item].signal_name == "phone_held" for item in edge_ids)
        )
        self.assertTrue(
            all(
                scenarios[item].truth_state is (
                    SignalState.COMPLIANT
                    if item.endswith("_recovery")
                    else SignalState.VIOLATION
                )
                for item in edge_ids
            )
        )
        original_phone_ids = {
            "phone_baseline_b01_natural_no_phone",
            "phone_baseline_b02_desktop_phone",
            "phone_baseline_b03_non_phone_object",
            "phone_baseline_b04_one_hand_phone",
            "phone_baseline_b05_recovery",
            "phone_baseline_b06_two_hands_low",
            "phone_baseline_b07_recovery",
            "phone_baseline_b08_phone_out_of_frame",
            "phone_baseline_b09_final",
        }
        self.assertFalse(edge_ids & original_phone_ids)

    def test_phone_edge_parameter_profiles_are_explicit_and_ordered(self) -> None:
        self.assertEqual(
            [profile["profile_id"] for profile in PHONE_EDGE_PARAMETER_PROFILES],
            [
                "P0_control",
                "P1_medium_recall",
                "P2_low_threshold_continuity",
                "P3_low_threshold_wide_tracking",
                "P4_medium_threshold_wide_tracking",
            ],
        )
        for profile in PHONE_EDGE_PARAMETER_PROFILES:
            self.assertGreaterEqual(profile["phone_score_threshold"], 0.0)
            self.assertLessEqual(profile["phone_score_threshold"], 1.0)
            self.assertGreaterEqual(profile["phone_retain_seconds"], 0.0)
            self.assertGreaterEqual(profile["phone_center_distance"], 0.0)
        self.assertEqual(
            PHONE_EDGE_PARAMETER_PROFILES[0]["phone_score_threshold"], 0.25
        )
        self.assertEqual(
            PHONE_EDGE_PARAMETER_PROFILES[0]["phone_retain_seconds"], 1.5
        )
        self.assertEqual(
            PHONE_EDGE_PARAMETER_PROFILES[0]["phone_center_distance"], 0.18
        )

    def test_phone_powerbank_contrast_is_paired_and_diagnostic_only(self) -> None:
        contrast = [
            scenario
            for scenario in SCENARIOS
            if scenario.scenario_id.startswith("phone_contrast_f")
        ]
        self.assertEqual(
            [scenario.scenario_id for scenario in contrast],
            [
                "phone_contrast_f00_clear",
                "phone_contrast_f01_powerbank_center",
                "phone_contrast_f02_phone_center",
                "phone_contrast_f03_clear",
                "phone_contrast_f04_powerbank_left",
                "phone_contrast_f05_phone_left",
                "phone_contrast_f06_clear",
                "phone_contrast_f07_powerbank_right",
                "phone_contrast_f08_phone_right",
                "phone_contrast_f09_final_clear",
            ],
        )
        self.assertTrue(
            {scenario.scenario_id for scenario in contrast}.issubset(
                _TUNING_DIAGNOSTIC_ONLY_SCENARIOS
            )
        )
        for scenario in contrast:
            self.assertEqual(scenario.signal_name, "phone_held")
            self.assertEqual(scenario.repetitions, 1)
            if "_phone_" in scenario.scenario_id:
                self.assertIs(scenario.truth_state, SignalState.VIOLATION)
                self.assertEqual(
                    scenario_acceptance_summary(scenario),
                    "DIAGNOSTIC: violation >=90%; continuous >=8s",
                )
            else:
                self.assertIs(scenario.truth_state, SignalState.COMPLIANT)
                self.assertEqual(
                    scenario_acceptance_summary(scenario),
                    "DIAGNOSTIC: violation <=10%; violation run <2s",
                )

    def test_phone_edge_scenarios_use_diagnostic_acceptance_only(self) -> None:
        scenario = next(
            item
            for item in SCENARIOS
            if item.scenario_id == "phone_edge_e02_bottom_left"
        )
        recovery = next(
            item
            for item in SCENARIOS
            if item.scenario_id == "phone_edge_e02_bottom_left_recovery"
        )
        self.assertEqual(
            scenario_acceptance_summary(scenario),
            "DIAGNOSTIC: violation >=90%; continuous >=10s",
        )
        self.assertEqual(
            scenario_acceptance_summary(recovery),
            "DIAGNOSTIC: violation <=10%; violation run <2.5s",
        )
        self.assertTrue(
            scenario_record_passes(
                scenario,
                TrialRecord(
                    trial_id="edge-positive",
                    scenario_id=scenario.scenario_id,
                    signal_name="phone_held",
                    dataset_role="tuning",
                    truth_state=SignalState.VIOLATION,
                    predicted_state=SignalState.VIOLATION,
                    started_monotonic=1.0,
                    ended_monotonic=13.0,
                    observable_fraction=1.0,
                    metrics={
                        "violation_fraction": 0.95,
                        "longest_violation_run_seconds": 10.1,
                    },
                ),
            )
        )

    def test_phone_baseline_applies_positive_negative_and_safe_unknown_rules(self) -> None:
        positive = next(
            scenario
            for scenario in SCENARIOS
            if scenario.scenario_id == "phone_baseline_b04_one_hand_phone"
        )
        desktop_phone = next(
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
                desktop_phone,
                record(
                    {
                        "violation_fraction": 0.05,
                        "longest_violation_run_seconds": 2.2,
                    }
                ),
            )
        )
        self.assertFalse(
            scenario_record_passes(
                desktop_phone,
                record(
                    {
                        "violation_fraction": 0.0,
                        "longest_violation_run_seconds": 0.0,
                    }
                ),
            )
        )
        self.assertTrue(
            scenario_record_passes(
                desktop_phone,
                record(
                    {
                        "violation_fraction": 0.95,
                        "longest_violation_run_seconds": 10.1,
                    }
                ),
            )
        )
        self.assertTrue(
            scenario_record_passes(
                low_two_hands,
                record(
                    {
                        "compliant_fraction": 0.0,
                        "unknown_fraction": 0.05,
                        "violation_fraction": 0.95,
                        "longest_violation_run_seconds": 10.1,
                    }
                ),
            )
        )
        self.assertFalse(
            scenario_record_passes(
                low_two_hands,
                record(
                    {
                        "compliant_fraction": 0.30,
                        "unknown_fraction": 0.05,
                        "violation_fraction": 0.65,
                        "longest_violation_run_seconds": 10.1,
                    }
                ),
            )
        )
        self.assertTrue(
            scenario_record_passes(
                low_two_hands,
                record(
                    {
                        "compliant_fraction": 0.0,
                        "unknown_fraction": 0.05,
                        "violation_fraction": 0.95,
                        "longest_violation_run_seconds": 10.1,
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
        # Scenario intent must not rewrite any production state or its metrics.
        for scenario in SCENARIOS:
            for signal in (SignalValue.compliant(), SignalValue.violation(),
                           SignalValue.unknown('unobservable')):
                self.assertIs(scenario_validation_signal(scenario, signal), signal)
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
            SignalState.VIOLATION,
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
            SignalState.COMPLIANT,
        )
        self.assertTrue(
            attention_control_action_matches(chin_control, small_chin_drop)
        )
        self.assertIs(
            scenario_validation_signal(chin_control, small_chin_drop).state,
            SignalState.VIOLATION,
        )

        side_control = next(
            scenario
            for scenario in SCENARIOS
            if scenario.scenario_id == "attention_desk_control_side"
        )
        self.assertIs(
            scenario_validation_signal(side_control, false_vertical_from_turn).state,
            SignalState.VIOLATION,
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
                SignalState.VIOLATION,
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
            SignalState.COMPLIANT,
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
            SignalState.VIOLATION,
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
