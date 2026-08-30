import tempfile
import unittest
from pathlib import Path

from observer.calibration import (
    ATTENTION_POINTS_9,
    AttentionSample,
    EnvironmentFingerprint,
    create_profile,
    fit_attention_calibration,
    fit_posture_calibration,
    load_profile,
    save_profile,
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

    def test_posture_uses_two_group_vote(self) -> None:
        calibration = fit_posture_calibration(
            [posture_sample(index) for index in range(40)], planned_sample_count=40
        )
        compliant = calibration.evaluate(posture_sample())
        one_group = calibration.evaluate(
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
        self.assertIs(one_group.state, SignalState.COMPLIANT)
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
