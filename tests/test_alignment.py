import unittest
from dataclasses import replace
from tempfile import TemporaryDirectory
from pathlib import Path

from observer.alignment import (
    AlignmentCancelled,
    AlignmentStability,
    _attention_setup_timeout_message,
    _count_feature_observation,
    attention_session_baseline_from_samples,
    assess_alignment,
    guide_attention_session_baseline,
)
from observer.calibration import (
    AttentionCalibration,
    CalibrationProfile,
    EnvironmentFingerprint,
    FeatureStats,
    PostureCalibration,
)
from observer.cli import (
    ATTENTION_BOUNDARY_CUES,
    ATTENTION_MINUTE_CUES,
    C_CLOSEOUT_HOLDOUT_CUES,
    GUIDED_DEBUG_CUES,
    _evidence_capture_kind,
    _finalize_debug_video,
    _finish_violation_intervals,
    _finalize_guided_case_summaries,
    _guided_case_metadata,
    _guided_debug_cue,
    _record_guided_case_observation,
    _track_violation_interval,
    build_parser,
)
from observer.observations import SignalState
from observer.perception import FeatureVector, QualityThresholds


def profile() -> CalibrationProfile:
    posture_stats = {
        "head_offset_x": FeatureStats(0.0, 0.01, -0.02, 0.02, 40),
        "head_offset_y": FeatureStats(-0.75, 0.01, -0.77, -0.73, 40),
        "shoulder_width": FeatureStats(0.4, 0.01, 0.38, 0.42, 40),
    }
    posture = PostureCalibration("observer-c1-v1", posture_stats, 40, 1.0)
    attention = AttentionCalibration(
        feature_version="observer-c1-v1",
        feature_names=("face_center_x", "face_center_y", "face_scale", "gaze_x"),
        feature_means=(0.52, 0.66, 0.30, 0.5),
        feature_scales=(0.01, 0.02, 0.01, 0.2),
        coefficients_x=(0.5, 0.0, 0.0, 0.0, 0.2),
        coefficients_y=(0.5, 0.0, 0.0, 0.0, 0.0),
        training_rmse=0.1,
        validation_rmse=0.15,
        valid_sample_count=119,
        coverage=1.0,
    )
    return CalibrationProfile(
        schema_version=1,
        profile_id="test-profile",
        created_at_utc="2026-08-30T12:00:00+00:00",
        environment=EnvironmentFingerprint(0, 640, 480, False, 1920, 1080, 100),
        quality=QualityThresholds(),
        posture=posture,
        attention=attention,
    )


def neutral_posture() -> FeatureVector:
    return FeatureVector(
        "posture",
        {"head_offset_x": 0.0, "head_offset_y": -0.75, "shoulder_width": 0.4},
    )


class AlignmentTests(unittest.TestCase):
    def test_headless_attention_setup_can_cancel_without_opening_preview(self) -> None:
        with self.assertRaisesRegex(AlignmentCancelled, "session setup cancelled"):
            guide_attention_session_baseline(
                object(),
                object(),
                show_preview=False,
                cancelled=lambda: True,
            )

    def test_attention_setup_timeout_diagnostics_are_safe_and_ranked(self) -> None:
        message = _attention_setup_timeout_message(
            face_runs=12,
            valid_frames=4,
            longest_valid_seconds=1.25,
            invalid_reason_counts={
                "quality:blurred": 3,
                "face_or_eyes_not_observable": 8,
            },
        )

        self.assertIn("face_runs=12", message)
        self.assertIn("valid_frames=4", message)
        self.assertIn("longest_valid=1.25s", message)
        self.assertLess(
            message.index("face_or_eyes_not_observable=8"),
            message.index("quality:blurred=3"),
        )

    def test_feature_diagnostics_distinguish_missing_rejected_and_observable(self) -> None:
        counts = {}
        _count_feature_observation(counts, None, 1)
        _count_feature_observation(
            counts,
            FeatureVector("attention", {}, reasons=("iris missing",)),
            2,
        )
        _count_feature_observation(
            counts,
            FeatureVector("attention", {"face_center_x": 0.5}),
            3,
        )

        self.assertEqual(
            counts,
            {"missing": 1, "iris missing": 2, "observable": 3},
        )

    def test_align_cli_exposes_same_session_verification(self) -> None:
        args = build_parser().parse_args(
            ["align", "--stable-seconds", "4", "--verify-seconds", "10"]
        )

        self.assertEqual(args.command, "align")
        self.assertEqual(args.stable_seconds, 4.0)
        self.assertEqual(args.verify_seconds, 10.0)
        self.assertFalse(args.require_posture)

    def test_vision_smoke_exposes_opt_in_evidence_and_report_paths(self) -> None:
        args = build_parser().parse_args(
            [
                "vision-smoke",
                "--evidence-dir",
                "data/evidence/test",
                "--report",
                "data/reports/test.json",
                "--debug-video",
                "data/debug/test.mp4",
                "--guided-debug-cues",
            ]
        )

        self.assertEqual(args.evidence_dir, Path("data/evidence/test"))
        self.assertEqual(args.report, Path("data/reports/test.json"))
        self.assertEqual(args.debug_video, Path("data/debug/test.mp4"))
        self.assertTrue(args.guided_debug_cues)

    def test_yolo_replay_exposes_uniform_frame_sampling(self) -> None:
        args = build_parser().parse_args(
            [
                "yolo-replay",
                "--video",
                "input.mp4",
                "--model",
                "model.onnx",
                "--output",
                "report.json",
                "--sample-frames",
                "16",
            ]
        )

        self.assertEqual(args.sample_frames, 16)

    def test_guided_debug_cues_have_objective_action_durations(self) -> None:
        self.assertEqual(_guided_debug_cue(219.9)[0], "normal_1")
        cue_id, instruction, remaining = _guided_debug_cue(220.0)
        self.assertEqual(cue_id, "outside_left")
        self.assertIn("LEFT", instruction)
        self.assertEqual(remaining, 12.0)
        self.assertEqual(_guided_debug_cue(231.9)[0], "outside_left")
        self.assertEqual(_guided_debug_cue(232.0)[0], "normal_2")
        self.assertEqual(_guided_debug_cue(245.0)[0], "phone")
        self.assertIn("NO PHONE", _guided_debug_cue(180.0)[1])
        self.assertIn("PHONE", _guided_debug_cue(245.0)[1])
        self.assertNotIn("NO PHONE", _guided_debug_cue(245.0)[1])
        self.assertTrue(all(len(cue[3]) <= 48 for cue in GUIDED_DEBUG_CUES))

    def test_attention_minute_plan_separates_positive_and_negative_controls(self) -> None:
        args = build_parser().parse_args(
            ["vision-smoke", "--guided-debug-plan", "attention-minute"]
        )
        self.assertEqual(args.guided_debug_plan, "attention-minute")
        self.assertEqual(
            _guided_debug_cue(10.0, ATTENTION_MINUTE_CUES)[0],
            "outside_left_down",
        )
        self.assertEqual(
            _guided_debug_cue(35.0, ATTENTION_MINUTE_CUES)[0],
            "near_lower_inside",
        )
        self.assertEqual(_guided_debug_cue(60.0, ATTENTION_MINUTE_CUES)[0], "complete")

    def test_attention_boundary_plan_has_unscored_preparation_and_scored_holds(self) -> None:
        args = build_parser().parse_args(
            ["vision-smoke", "--guided-debug-plan", "attention-boundary"]
        )

        self.assertEqual(args.guided_debug_plan, "attention-boundary")
        self.assertEqual(ATTENTION_BOUNDARY_CUES[-1][1], 151.0)
        self.assertEqual(_guided_debug_cue(0.0, ATTENTION_BOUNDARY_CUES)[0], "prep_a01")
        self.assertEqual(_guided_debug_cue(3.0, ATTENTION_BOUNDARY_CUES)[0], "score_a01")
        self.assertEqual(_guided_debug_cue(68.0, ATTENTION_BOUNDARY_CUES)[0], "score_a06")
        self.assertEqual(_guided_debug_cue(98.0, ATTENTION_BOUNDARY_CUES)[0], "score_a08")
        self.assertEqual(_guided_debug_cue(126.0, ATTENTION_BOUNDARY_CUES)[0], "score_a10")
        self.assertEqual(_guided_debug_cue(151.0, ATTENTION_BOUNDARY_CUES)[0], "complete")
        self.assertFalse(_guided_case_metadata("prep_a02")["scored"])
        self.assertTrue(_guided_case_metadata("score_a02")["scored"])
        self.assertEqual(
            _guided_case_metadata("score_a02")["expected_state"], "violation"
        )

    def test_closeout_holdout_plan_is_fixed_and_contains_no_model_feedback(self) -> None:
        args = build_parser().parse_args(
            ["vision-smoke", "--guided-debug-plan", "c-closeout-holdout"]
        )

        self.assertEqual(args.guided_debug_plan, "c-closeout-holdout")
        self.assertEqual(C_CLOSEOUT_HOLDOUT_CUES[-1][1], 144.0)
        self.assertEqual(_guided_debug_cue(15.0, C_CLOSEOUT_HOLDOUT_CUES)[0], "h02_away")
        self.assertEqual(_guided_debug_cue(60.0, C_CLOSEOUT_HOLDOUT_CUES)[0], "h04_phone")
        self.assertEqual(
            _guided_debug_cue(121.0, C_CLOSEOUT_HOLDOUT_CUES)[0],
            "h08_camera_cover",
        )
        instructions = " ".join(cue[3].upper() for cue in C_CLOSEOUT_HOLDOUT_CUES)
        self.assertNotIn("COMPLIANT", instructions)
        self.assertNotIn("VIOLATION", instructions)
        self.assertNotIn("TARGET READY", instructions)
        self.assertIn("NO FACE, SHOULDER, ARM, OR TORSO", instructions)

    def test_attention_boundary_summary_excludes_preparation_and_keeps_exact_runs(self) -> None:
        summaries = {}
        active = {}
        prep = _guided_case_metadata("prep_a02")
        scored = _guided_case_metadata("score_a02")
        _record_guided_case_observation(summaries, active, prep, 14.0, "compliant")
        _record_guided_case_observation(summaries, active, scored, 16.0, "violation")
        _record_guided_case_observation(summaries, active, scored, 17.0, "violation")
        _record_guided_case_observation(summaries, active, scored, 18.0, "unknown")
        _record_guided_case_observation(summaries, active, scored, 19.0, "violation")

        rendered = _finalize_guided_case_summaries(summaries, active)

        self.assertEqual(len(rendered), 1)
        self.assertEqual(rendered[0]["case_id"], "A02")
        self.assertEqual(rendered[0]["observation_count"], 4)
        self.assertEqual(rendered[0]["state_counts"], {"violation": 3, "unknown": 1})
        self.assertEqual(rendered[0]["longest_exact_violation_seconds"], 2.0)

    def test_debug_video_report_includes_timestamp_index_and_hashes(self) -> None:
        with TemporaryDirectory() as directory:
            video = Path(directory) / "test.mp4"
            video.write_bytes(b"temporary-video")

            result = _finalize_debug_video(video, [0.1, 0.2], 15.0)

            self.assertTrue(result["temporary"])
            self.assertTrue(result["review_then_delete"])
            self.assertEqual(result["frame_count"], 2)
            self.assertTrue(Path(result["timestamps_path"]).exists())
            self.assertEqual(len(result["sha256"]), 64)

    def test_natural_monitor_records_continuous_violation_intervals(self) -> None:
        active = {}
        completed = {}
        _track_violation_interval("attention_away", "violation", 10.0, active, completed)
        _track_violation_interval("attention_away", "compliant", 10.2, active, completed)
        _track_violation_interval("attention_away", "violation", 10.4, active, completed)
        _track_violation_interval("attention_away", "compliant", 11.0, active, completed)
        _finish_violation_intervals(active, completed, 11.0)

        self.assertEqual(len(completed["attention_away"]), 1)
        self.assertEqual(completed["attention_away"][0]["start_seconds"], 10.0)
        self.assertEqual(completed["attention_away"][0]["end_seconds"], 10.4)

    def test_sustained_violation_reserves_evidence_after_regular_cap(self) -> None:
        active = {
            "attention_away": {
                "start_seconds": 10.0,
                "last_violation_seconds": 18.1,
            }
        }

        kind = _evidence_capture_kind(
            "attention_away",
            "violation",
            18.1,
            active,
            {"attention_away": 3},
            {"attention_away": 17.0},
            set(),
            {"attention_away": 8.0},
            100.0,
        )

        self.assertEqual(kind, "sustained_threshold")

    def test_sustained_evidence_is_only_reserved_once(self) -> None:
        kind = _evidence_capture_kind(
            "attention_away",
            "violation",
            20.0,
            {"attention_away": {"start_seconds": 10.0, "last_violation_seconds": 20.0}},
            {"attention_away": 3},
            {"attention_away": 19.0},
            {"attention_away"},
            {"attention_away": 8.0},
            100.0,
        )

        self.assertIsNone(kind)

    def test_matching_face_context_and_posture_are_aligned(self) -> None:
        attention = FeatureVector(
            "attention",
            {"face_center_x": 0.52, "face_center_y": 0.66, "face_scale": 0.30},
        )

        result = assess_alignment(profile(), attention, neutral_posture())

        self.assertTrue(result.aligned)
        self.assertAlmostEqual(result.context_z or 0.0, 2.0)
        self.assertEqual(result.target_x, 0.5)
        self.assertAlmostEqual(result.tolerance_x or 0.0, 0.05)
        self.assertIs(result.posture_state, SignalState.COMPLIANT)

    def test_alignment_does_not_copy_an_off_center_calibration_position(self) -> None:
        attention = FeatureVector(
            "attention",
            {"face_center_x": 0.50, "face_center_y": 0.66, "face_scale": 0.30},
        )

        result = assess_alignment(profile(), attention, neutral_posture())

        self.assertTrue(result.aligned)
        self.assertEqual(result.target_x, 0.5)
        self.assertAlmostEqual(result.tolerance_x or 0.0, 0.05)

    def test_natural_session_alignment_does_not_force_historical_face_scale(self) -> None:
        attention = FeatureVector(
            "attention",
            {"face_center_x": 0.50, "face_center_y": 0.66, "face_scale": 0.42},
        )

        result = assess_alignment(
            profile(),
            attention,
            neutral_posture(),
            require_face_scale=False,
        )

        self.assertTrue(result.aligned)
        self.assertFalse(result.scale_required)

    def test_natural_vertical_shift_with_complete_face_is_accepted(self) -> None:
        calibration = profile()
        calibration = replace(
            calibration,
            attention=replace(
                calibration.attention,
                feature_scales=(0.01, 0.001, 0.01, 0.2),
            ),
        )
        attention = FeatureVector(
            "attention",
            {"face_center_x": 0.50, "face_center_y": 0.715, "face_scale": 0.47},
        )

        result = assess_alignment(
            calibration,
            attention,
            neutral_posture(),
            require_face_scale=False,
        )

        self.assertTrue(result.aligned)
        self.assertAlmostEqual(result.tolerance_y or 0.0, 0.06)

    def test_large_vertical_shift_remains_blocked(self) -> None:
        attention = FeatureVector(
            "attention",
            {"face_center_x": 0.50, "face_center_y": 0.80, "face_scale": 0.47},
        )

        result = assess_alignment(
            profile(),
            attention,
            neutral_posture(),
            require_face_scale=False,
        )

        self.assertFalse(result.aligned)
        self.assertIn("position box", result.reason)

    def test_repeatable_face_shift_is_not_accepted_as_aligned(self) -> None:
        attention = FeatureVector(
            "attention",
            {"face_center_x": 0.70, "face_center_y": 0.66, "face_scale": 0.30},
        )

        result = assess_alignment(profile(), attention, neutral_posture())

        self.assertFalse(result.aligned)
        self.assertGreater(result.context_z or 0.0, 5.0)
        self.assertIn("position box", result.reason)

    def test_face_size_mismatch_has_specific_distance_instruction(self) -> None:
        attention = FeatureVector(
            "attention",
            {"face_center_x": 0.52, "face_center_y": 0.66, "face_scale": 0.40},
        )

        result = assess_alignment(
            profile(), attention, neutral_posture(), require_posture=False
        )

        self.assertFalse(result.aligned)
        self.assertIn("farther", result.reason)
        self.assertAlmostEqual(result.tolerance_scale or 0.0, 0.05)

    def test_unobservable_posture_prevents_alignment(self) -> None:
        attention = FeatureVector(
            "attention",
            {"face_center_x": 0.52, "face_center_y": 0.66, "face_scale": 0.30},
        )
        posture = FeatureVector("posture", {}, reasons=("shoulders missing",))

        result = assess_alignment(profile(), attention, posture)

        self.assertFalse(result.aligned)
        self.assertIs(result.posture_state, SignalState.UNKNOWN)
        self.assertIn("shoulders missing", result.reason)

    def test_attention_only_alignment_records_but_does_not_gate_on_posture(self) -> None:
        attention = FeatureVector(
            "attention",
            {"face_center_x": 0.52, "face_center_y": 0.66, "face_scale": 0.30},
        )
        posture = FeatureVector("posture", {}, reasons=("shoulders missing",))

        result = assess_alignment(
            profile(), attention, posture, require_posture=False
        )

        self.assertTrue(result.aligned)
        self.assertIs(result.posture_state, SignalState.UNKNOWN)
        self.assertIn("not gating", result.reason)

    def test_stability_timer_pauses_for_brief_misalignment(self) -> None:
        stability = AlignmentStability(required_seconds=3.0)

        self.assertEqual(stability.update(10.0, True), (0.0, False))
        self.assertEqual(stability.update(12.0, True), (2.0, False))
        elapsed, complete = stability.update(12.1, False)
        self.assertAlmostEqual(elapsed, 2.1)
        self.assertFalse(complete)
        elapsed, complete = stability.update(12.4, True)
        self.assertAlmostEqual(elapsed, 2.1)
        self.assertFalse(complete)
        elapsed, complete = stability.update(13.4, True)
        self.assertAlmostEqual(elapsed, 3.1)
        self.assertTrue(complete)

    def test_stability_timer_resets_after_sustained_misalignment(self) -> None:
        stability = AlignmentStability(required_seconds=3.0)

        stability.update(10.0, True)
        stability.update(12.0, False)
        self.assertEqual(stability.update(12.6, False), (0.0, False))
        self.assertEqual(stability.update(13.0, True), (0.0, False))

    def test_stability_timer_tolerates_short_indeterminate_dropout(self) -> None:
        stability = AlignmentStability(
            required_seconds=3.0, dropout_grace_seconds=0.5
        )

        stability.update(10.0, True)
        self.assertEqual(stability.update(11.0, None), (1.0, False))
        elapsed, complete = stability.update(11.3, True)
        self.assertAlmostEqual(elapsed, 1.0)
        self.assertFalse(complete)
        elapsed, complete = stability.update(13.1, True)
        self.assertAlmostEqual(elapsed, 2.8)
        self.assertFalse(complete)
        elapsed, complete = stability.update(13.4, True)
        self.assertAlmostEqual(elapsed, 3.1)
        self.assertTrue(complete)

    def test_stability_timer_resets_after_long_indeterminate_dropout(self) -> None:
        stability = AlignmentStability(
            required_seconds=3.0, dropout_grace_seconds=0.5
        )

        stability.update(10.0, True)
        stability.update(11.0, None)
        self.assertEqual(stability.update(11.6, None), (0.0, False))
        self.assertEqual(stability.update(12.0, True), (0.0, False))

    def test_attention_session_baseline_uses_robust_medians(self) -> None:
        samples = [
            FeatureVector(
                "attention",
                {"head_yaw_proxy": yaw, "head_pitch_proxy": pitch},
            )
            for yaw, pitch in ((0.1, 0.2), (0.11, 0.21), (2.0, -3.0))
        ]

        baseline = attention_session_baseline_from_samples(samples)

        self.assertAlmostEqual(baseline["head_yaw_proxy"], 0.11)
        self.assertAlmostEqual(baseline["head_pitch_proxy"], 0.2)

    def test_attention_session_baseline_rejects_incomplete_samples(self) -> None:
        with self.assertRaises(ValueError):
            attention_session_baseline_from_samples(
                [FeatureVector("attention", {"head_yaw_proxy": 0.1})]
            )


if __name__ == "__main__":
    unittest.main()
