import unittest
from dataclasses import dataclass

import numpy as np

from observer.observations import NormalizedBox, PhoneCandidate
from observer.perception import (
    FeatureMedianSmoother,
    FeatureVector,
    PhoneCandidateTracker,
    PresenceHitTracker,
    QualityThresholds,
    assess_frame_quality,
    extract_attention_features,
    extract_posture_features,
)


@dataclass
class Landmark:
    x: float
    y: float
    visibility: float = 1.0


class PerceptionTests(unittest.TestCase):
    def test_feature_median_smoother_rejects_single_frame_outlier(self) -> None:
        smoother = FeatureMedianSmoother("attention", window_size=7)
        result = None
        for value in (0.4, 0.41, 0.39, 4.0, 0.42, 0.4, 0.38):
            result = smoother.update(
                FeatureVector("attention", {"iris_y": value, "iris_x": 0.5})
            )

        self.assertIsNotNone(result)
        self.assertTrue(result.observable)
        self.assertAlmostEqual(result.values["iris_y"], 0.4)

    def test_feature_median_smoother_tolerates_one_blink_then_resets(self) -> None:
        smoother = FeatureMedianSmoother("attention", window_size=3)
        for value in (0.4, 0.5, 0.6):
            result = smoother.update(
                FeatureVector("attention", {"iris_y": value})
            )
        self.assertTrue(result.observable)

        missing = smoother.update(
            FeatureVector("attention", {}, reasons=("face missing",))
        )
        recovered = smoother.update(
            FeatureVector("attention", {"iris_y": 0.7})
        )

        self.assertFalse(missing.observable)
        self.assertTrue(recovered.observable)

        for _ in range(3):
            smoother.update(
                FeatureVector("attention", {}, reasons=("eyes closed",))
            )
        warming = smoother.update(FeatureVector("attention", {"iris_y": 0.8}))

        self.assertFalse(warming.observable)
        self.assertIn("1/3", warming.reasons[0])

    def test_dark_flat_frame_is_not_observable(self) -> None:
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        quality = assess_frame_quality(frame)
        self.assertFalse(quality.observable)
        self.assertIn("frame too dark", quality.reasons)
        self.assertIn("frame too blurry", quality.reasons)

    def test_textured_mid_brightness_frame_is_observable(self) -> None:
        checker = np.indices((80, 80)).sum(axis=0) % 2
        gray = (checker * 100 + 75).astype(np.uint8)
        frame = np.repeat(gray[:, :, None], 3, axis=2)
        quality = assess_frame_quality(frame, QualityThresholds(blur_variance_min=1.0))
        self.assertTrue(quality.observable)

    def test_warm_colored_lens_cover_is_unobservable_without_being_black(self) -> None:
        frame = np.full((80, 80, 3), (65, 85, 125), dtype=np.uint8)

        quality = assess_frame_quality(frame)

        self.assertGreater(quality.brightness_mean, 35.0)
        self.assertFalse(quality.observable)
        self.assertIn("camera view may be covered", quality.reasons)

    def test_dim_scene_with_real_edges_is_not_treated_as_lens_cover(self) -> None:
        checker = np.indices((80, 80)).sum(axis=0) % 2
        gray = (checker * 50 + 60).astype(np.uint8)
        frame = np.repeat(gray[:, :, None], 3, axis=2)

        quality = assess_frame_quality(frame)

        self.assertTrue(quality.observable)
        self.assertNotIn("camera view may be covered", quality.reasons)

    def test_attention_rejects_missing_iris_landmarks(self) -> None:
        result = extract_attention_features([Landmark(0.5, 0.5)] * 20)
        self.assertFalse(result.observable)

    def test_posture_extracts_independent_feature_groups(self) -> None:
        pose = [Landmark(0.5, 0.5) for _ in range(25)]
        pose[7] = Landmark(0.43, 0.30)
        pose[8] = Landmark(0.57, 0.30)
        pose[11] = Landmark(0.40, 0.50)
        pose[12] = Landmark(0.60, 0.50)
        pose[23] = Landmark(0.44, 0.80)
        pose[24] = Landmark(0.56, 0.80)
        result = extract_posture_features(pose)
        self.assertTrue(result.observable)
        self.assertIn("shoulder_slope", result.values)
        self.assertIn("torso_lean", result.values)

    def test_posture_keeps_face_only_features_when_shoulders_are_cropped(self) -> None:
        face = [Landmark(0.5, 0.5) for _ in range(478)]
        face[0] = Landmark(0.35, 0.35)
        face[10] = Landmark(0.65, 0.65)
        face[33] = Landmark(0.42, 0.45)
        face[263] = Landmark(0.58, 0.45)
        face[1] = Landmark(0.50, 0.53)
        pose = [Landmark(0.5, 0.5) for _ in range(25)]
        pose[11].visibility = 0.1
        pose[12].visibility = 0.1

        result = extract_posture_features(pose, face)

        self.assertTrue(result.observable)
        self.assertIn("face_head_pitch", result.values)
        self.assertIn("face_head_roll", result.values)
        self.assertIn("face_scale", result.values)
        self.assertNotIn("shoulder_slope", result.values)

    def test_phone_tracker_confirms_and_expires_candidate(self) -> None:
        tracker = PhoneCandidateTracker(confirmation_hits=2, retain_seconds=0.2)
        phone = PhoneCandidate(NormalizedBox(0.4, 0.4, 0.1, 0.2), 0.9)
        self.assertEqual(tracker.update(1.0, (phone,)), ())
        self.assertEqual(tracker.update(1.1, (phone,)), (phone,))
        self.assertEqual(tracker.update(1.4, ()), ())

    def test_default_phone_tracker_bridges_short_detector_gaps(self) -> None:
        tracker = PhoneCandidateTracker()
        phone = PhoneCandidate(NormalizedBox(0.4, 0.4, 0.1, 0.2), 0.9)
        self.assertEqual(tracker.update(1.0, (phone,)), ())
        self.assertEqual(tracker.update(1.2, (phone,)), (phone,))
        self.assertEqual(tracker.update(2.3, ()), (phone,))
        self.assertEqual(tracker.update(3.1, ()), (phone,))
        self.assertEqual(tracker.update(3.3, ()), ())

    def test_phone_tracker_does_not_abandon_track_for_remote_false_box(self) -> None:
        tracker = PhoneCandidateTracker(confirmation_hits=2)
        phone = PhoneCandidate(NormalizedBox(0.45, 0.75, 0.25, 0.2), 0.8)
        moved_phone = PhoneCandidate(NormalizedBox(0.47, 0.76, 0.24, 0.19), 0.3)
        remote_false_box = PhoneCandidate(NormalizedBox(0.0, 0.8, 0.1, 0.1), 0.4)

        self.assertEqual(tracker.update(1.0, (phone,)), ())
        self.assertEqual(tracker.update(1.2, (phone,)), (phone,))
        self.assertEqual(
            tracker.update(1.4, (remote_false_box, moved_phone)),
            (moved_phone,),
        )

    def test_person_presence_requires_consecutive_hits(self) -> None:
        tracker = PresenceHitTracker(confirmation_hits=2)
        self.assertEqual(tracker.update(1), 0)
        self.assertEqual(tracker.update(0), 0)
        self.assertEqual(tracker.update(1), 0)
        self.assertEqual(tracker.update(1), 1)
        self.assertEqual(tracker.update(0), 0)


if __name__ == "__main__":
    unittest.main()
