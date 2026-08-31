import unittest

from observer.alignment import (
    AlignmentStability,
    _count_feature_observation,
    assess_alignment,
)
from observer.calibration import (
    AttentionCalibration,
    CalibrationProfile,
    EnvironmentFingerprint,
    FeatureStats,
    PostureCalibration,
)
from observer.cli import build_parser
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

    def test_matching_face_context_and_posture_are_aligned(self) -> None:
        attention = FeatureVector(
            "attention",
            {"face_center_x": 0.52, "face_center_y": 0.66, "face_scale": 0.30},
        )

        result = assess_alignment(profile(), attention, neutral_posture())

        self.assertTrue(result.aligned)
        self.assertEqual(result.context_z, 0.0)
        self.assertIs(result.posture_state, SignalState.COMPLIANT)

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


if __name__ == "__main__":
    unittest.main()
