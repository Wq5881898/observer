import tempfile
import unittest
from pathlib import Path

from observer.calibration import (
    ATTENTION_POINTS_9,
    AttentionCalibration,
    AttentionSessionBaseline,
    AttentionSample,
    EnvironmentFingerprint,
    aggregate_attention_samples,
    create_profile,
    fit_attention_calibration,
    fit_posture_calibration,
    load_profile,
    save_profile,
)
from observer.calibration_workflow import (
    ATTENTION_CALIBRATION_POINTS,
    ATTENTION_VALIDATION_POINTS,
    _attention_validation_rmse_by_point,
)
from observer.observations import SignalState
from observer.perception import FeatureVector


def posture_sample(index: int = 0) -> FeatureVector:
    jitter = (index % 5 - 2) * 0.001
    return FeatureVector(
        "posture",
        {
            "shoulder_slope": jitter,
            "head_offset_x": jitter,
            "head_offset_y": -1.0 + jitter,
            "shoulder_width": 0.3 + jitter,
            "torso_lean": jitter,
        },
    )


def attention_samples() -> list[AttentionSample]:
    return [
        AttentionSample(
            FeatureVector("attention", {"gaze_x": x, "gaze_y": y, "scale": 0.4}),
            x,
            y,
            point_id,
        )
        for point_id, x, y in ATTENTION_POINTS_9
    ]


class CalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = EnvironmentFingerprint(0, 640, 480, True, 1920, 1080, 100)

    def test_posture_uses_two_group_vote_or_corroborated_lateral_lean(self) -> None:
        calibration = fit_posture_calibration(
            [posture_sample(index) for index in range(40)], planned_sample_count=40
        )
        compliant = calibration.evaluate(posture_sample())
        one_feature = calibration.evaluate(
            FeatureVector(
                "posture",
                {
                    **posture_sample().values,
                    "shoulder_slope": 1.0,
                },
            )
        )
        corroborated_alignment = calibration.evaluate(
            FeatureVector(
                "posture",
                {
                    **posture_sample().values,
                    "shoulder_slope": 1.0,
                    "head_offset_x": 1.0,
                },
            )
        )
        two_groups = calibration.evaluate(
            FeatureVector(
                "posture",
                {
                    **posture_sample().values,
                    "shoulder_slope": 1.0,
                    "head_offset_y": 0.0,
                },
            )
        )
        self.assertIs(compliant.state, SignalState.COMPLIANT)
        self.assertIs(one_feature.state, SignalState.COMPLIANT)
        self.assertIs(corroborated_alignment.state, SignalState.VIOLATION)
        self.assertIs(two_groups.state, SignalState.VIOLATION)

    def test_attention_ridge_model_classifies_inside_and_outside(self) -> None:
        calibration = fit_attention_calibration(attention_samples())
        inside = calibration.evaluate(
            FeatureVector("attention", {"gaze_x": 0.5, "gaze_y": 0.5, "scale": 0.4})
        )
        outside = calibration.evaluate(
            FeatureVector("attention", {"gaze_x": 1.3, "gaze_y": 0.5, "scale": 0.4})
        )
        self.assertIs(inside.state, SignalState.COMPLIANT)
        self.assertIs(outside.state, SignalState.VIOLATION)

    def test_attention_uses_eye_openness_for_observability_not_regression(self) -> None:
        samples = [
            AttentionSample(
                FeatureVector(
                    "attention",
                    {
                        "gaze_x": x,
                        "gaze_y": y,
                        "eye_a_open": 0.2 + index * 0.03,
                        "eye_b_open": 0.7 - index * 0.04,
                    },
                ),
                x,
                y,
                point_id,
            )
            for index, (point_id, x, y) in enumerate(ATTENTION_POINTS_9)
        ]

        calibration = fit_attention_calibration(samples)

        self.assertNotIn("eye_a_open", calibration.feature_names)
        self.assertNotIn("eye_b_open", calibration.feature_names)
        self.assertEqual(set(calibration.feature_names), {"gaze_x", "gaze_y"})

    def test_attention_validation_diagnostics_report_each_point(self) -> None:
        samples = attention_samples()
        calibration = fit_attention_calibration(samples)

        diagnostics = _attention_validation_rmse_by_point(calibration, samples)

        self.assertEqual(set(diagnostics), {sample.point_id for sample in samples})
        self.assertLess(max(diagnostics.values()), 0.18)

    def test_attention_point_aggregation_uses_feature_medians(self) -> None:
        samples = [
            AttentionSample(
                FeatureVector("attention", {"gaze_x": value, "gaze_y": 0.5}),
                0.5,
                0.5,
                "center",
            )
            for value in (0.48, 0.5, 4.0, 0.52, 0.49)
        ]

        aggregated = aggregate_attention_samples(samples)

        self.assertEqual(len(aggregated), 1)
        self.assertAlmostEqual(aggregated[0].features.values["gaze_x"], 0.5)

    def test_attention_training_and_validation_points_are_disjoint(self) -> None:
        training_coordinates = {(x, y) for _point_id, x, y in ATTENTION_CALIBRATION_POINTS}
        validation_coordinates = {(x, y) for _point_id, x, y in ATTENTION_VALIDATION_POINTS}

        self.assertEqual(len(training_coordinates), 17)
        self.assertEqual(len(validation_coordinates), 8)
        self.assertTrue(training_coordinates.isdisjoint(validation_coordinates))

    def test_attention_rejects_face_context_far_outside_calibration(self) -> None:
        samples = [
            AttentionSample(
                FeatureVector(
                    "attention",
                    {
                        "gaze_x": x,
                        "gaze_y": y,
                        "face_center_x": 0.5 + index * 0.001,
                        "face_center_y": 0.5 - index * 0.001,
                        "face_scale": 0.4 + index * 0.0005,
                    },
                ),
                x,
                y,
                point_id,
            )
            for index, (point_id, x, y) in enumerate(ATTENTION_POINTS_9)
        ]
        calibration = fit_attention_calibration(samples)
        signal = calibration.evaluate(
            FeatureVector(
                "attention",
                {
                    "gaze_x": 0.5,
                    "gaze_y": 0.5,
                    "face_center_x": 0.8,
                    "face_center_y": 0.8,
                    "face_scale": 0.2,
                },
            )
        )
        self.assertIs(signal.state, SignalState.UNKNOWN)
        self.assertIn("outside", signal.reason or "")

    def test_attention_context_guard_has_practical_resitting_floor(self) -> None:
        samples = [
            AttentionSample(
                FeatureVector(
                    "attention",
                    {
                        "gaze_x": x,
                        "gaze_y": y,
                        "face_center_x": 0.52,
                        "face_center_y": 0.66,
                        "face_scale": 0.30,
                    },
                ),
                x,
                y,
                point_id,
            )
            for point_id, x, y in ATTENTION_POINTS_9
        ]
        calibration = fit_attention_calibration(samples)

        signal = calibration.evaluate(
            FeatureVector(
                "attention",
                {
                    "gaze_x": 0.5,
                    "gaze_y": 0.5,
                    "face_center_x": 0.539,
                    "face_center_y": 0.689,
                    "face_scale": 0.309,
                },
            )
        )

        self.assertIs(signal.state, SignalState.COMPLIANT)
        self.assertLessEqual(signal.metrics.get("attention_context_z", 0.0), 5.0)

        practical_scale_shift = calibration.evaluate(
            FeatureVector(
                "attention",
                {
                    "gaze_x": 0.5,
                    "gaze_y": 0.5,
                    "face_center_x": 0.52,
                    "face_center_y": 0.66,
                    "face_scale": 0.324,
                },
            )
        )
        self.assertIs(practical_scale_shift.state, SignalState.COMPLIANT)
        self.assertLessEqual(
            practical_scale_shift.metrics.get("attention_context_z", 0.0), 5.0
        )

    def test_zero_weight_context_aligns_but_does_not_reject_look_away(self) -> None:
        calibration = AttentionCalibration(
            feature_version="observer-c1-v1",
            feature_names=("iris_a_x", "face_center_x", "face_center_y", "face_scale"),
            feature_means=(0.5, 0.5, 0.5, 0.3),
            feature_scales=(0.1, 0.004, 0.006, 0.005),
            coefficients_x=(0.5, 1.0, 0.0, 0.0, 0.0),
            coefficients_y=(0.5, 0.0, 0.0, 0.0, 0.0),
            training_rmse=0.1,
            validation_rmse=0.1,
            valid_sample_count=9,
            coverage=1.0,
        )
        features = FeatureVector(
            "attention",
            {
                "iris_a_x": 0.5,
                "face_center_x": 0.8,
                "face_center_y": 0.8,
                "face_scale": 0.5,
            },
        )

        signal = calibration.evaluate(features)

        self.assertIs(signal.state, SignalState.COMPLIANT)
        self.assertGreater(signal.metrics["attention_context_z"], 5.0)

    def test_session_head_pose_recovers_attention_when_iris_is_unreliable(self) -> None:
        calibration = AttentionCalibration(
            feature_version="observer-c1-v1",
            feature_names=("iris_a_x",),
            feature_means=(0.5,),
            feature_scales=(0.1,),
            coefficients_x=(0.5, 0.1),
            coefficients_y=(0.5, 0.0),
            training_rmse=0.1,
            validation_rmse=0.1,
            valid_sample_count=9,
            coverage=1.0,
        )
        baseline = AttentionSessionBaseline(0.0, 0.2)
        features = FeatureVector(
            "attention",
            {
                "iris_a_x": 0.5,
                "head_yaw_proxy": 0.25,
                "head_pitch_proxy": 0.21,
            },
            reasons=("eyes closed or eye landmarks unreliable",),
        )

        signal = calibration.evaluate_with_session_baseline(features, baseline)

        self.assertIs(signal.state, SignalState.VIOLATION)
        self.assertAlmostEqual(signal.metrics["attention_head_yaw_delta"], 0.25)

    def test_session_head_pose_keeps_small_natural_motion_compliant(self) -> None:
        calibration = AttentionCalibration(
            feature_version="observer-c1-v1",
            feature_names=("iris_a_x",),
            feature_means=(0.5,),
            feature_scales=(0.1,),
            coefficients_x=(0.5, 0.1),
            coefficients_y=(0.5, 0.0),
            training_rmse=0.1,
            validation_rmse=0.1,
            valid_sample_count=9,
            coverage=1.0,
        )
        signal = calibration.evaluate_with_session_baseline(
            FeatureVector(
                "attention",
                {
                    "iris_a_x": 0.5,
                    "head_yaw_proxy": 0.04,
                    "head_pitch_proxy": 0.23,
                },
            ),
            AttentionSessionBaseline(0.0, 0.2),
        )

        self.assertIs(signal.state, SignalState.COMPLIANT)

    def test_down_attention_requires_pitch_and_gaze_agreement(self) -> None:
        calibration = AttentionCalibration(
            feature_version="observer-c1-v1",
            feature_names=("iris_a_y",),
            feature_means=(0.5,),
            feature_scales=(0.1,),
            coefficients_x=(0.5, 0.0),
            coefficients_y=(0.5, 0.1),
            training_rmse=0.1,
            validation_rmse=0.1,
            valid_sample_count=9,
            coverage=1.0,
        )
        baseline = AttentionSessionBaseline(0.0, 0.2)

        horizontal_turn_false_y = calibration.evaluate_with_session_baseline(
            FeatureVector(
                "attention",
                {
                    "iris_a_y": 1.17,
                    "head_yaw_proxy": -0.10,
                    "head_pitch_proxy": 0.25,
                },
            ),
            baseline,
        )
        small_chin_drop = calibration.evaluate_with_session_baseline(
            FeatureVector(
                "attention",
                {
                    "iris_a_y": 0.34,
                    "head_yaw_proxy": -0.03,
                    "head_pitch_proxy": 0.29,
                },
            ),
            baseline,
        )
        real_down = calibration.evaluate_with_session_baseline(
            FeatureVector(
                "attention",
                {
                    "iris_a_y": 1.17,
                    "head_yaw_proxy": -0.03,
                    "head_pitch_proxy": 0.29,
                },
            ),
            baseline,
        )

        self.assertIs(horizontal_turn_false_y.state, SignalState.COMPLIANT)
        self.assertIs(small_chin_drop.state, SignalState.COMPLIANT)
        self.assertIs(real_down.state, SignalState.VIOLATION)

    def test_profile_roundtrip_and_environment_guard(self) -> None:
        posture = fit_posture_calibration(
            [posture_sample(index) for index in range(40)], planned_sample_count=40
        )
        attention = fit_attention_calibration(attention_samples())
        profile = create_profile(self.environment, posture, attention)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            save_profile(profile, path)
            loaded = load_profile(path, self.environment)
            self.assertEqual(loaded.profile_id, profile.profile_id)
            changed = EnvironmentFingerprint(0, 640, 480, True, 2560, 1440, 100)
            with self.assertRaisesRegex(ValueError, "screen_width"):
                load_profile(path, changed)

    def test_unknown_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            path.write_text('{"schema_version": 999}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported calibration schema"):
                load_profile(path)


if __name__ == "__main__":
    unittest.main()
