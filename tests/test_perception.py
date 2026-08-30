import unittest
from dataclasses import dataclass

import numpy as np

from observer.observations import NormalizedBox, PhoneCandidate
from observer.perception import (
    PhoneCandidateTracker,
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

    def test_phone_tracker_confirms_and_expires_candidate(self) -> None:
        tracker = PhoneCandidateTracker(confirmation_hits=2, retain_seconds=0.2)
        phone = PhoneCandidate(NormalizedBox(0.4, 0.4, 0.1, 0.2), 0.9)
        self.assertEqual(tracker.update(1.0, (phone,)), ())
        self.assertEqual(tracker.update(1.1, (phone,)), (phone,))
        self.assertEqual(tracker.update(1.4, ()), ())


if __name__ == "__main__":
    unittest.main()
