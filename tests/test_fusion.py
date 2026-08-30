import unittest

from observer.fusion import ObservationFusion
from observer.observations import (
    NormalizedBox,
    NormalizedPoint,
    PhoneCandidate,
    SignalState,
)


class ObservationFusionTests(unittest.TestCase):
    def test_person_present_when_either_face_or_pose_is_fresh(self) -> None:
        fusion = ObservationFusion()
        fusion.update_face(10.0, 1)
        observation = fusion.build(10.1)
        self.assertIs(observation.signal("away").state, SignalState.COMPLIANT)

    def test_absence_requires_both_detectors_to_be_fresh_and_empty(self) -> None:
        fusion = ObservationFusion()
        fusion.update_face(10.0, 0)
        self.assertIs(fusion.build(10.1).signal("away").state, SignalState.UNKNOWN)
        fusion.update_pose(10.1, 0, ())
        self.assertIs(fusion.build(10.2).signal("away").state, SignalState.VIOLATION)

    def test_stale_object_result_is_unknown(self) -> None:
        fusion = ObservationFusion()
        fusion.update_phones(1.0, ())
        signal = fusion.build(10.0).signal("phone_held")
        self.assertIs(signal.state, SignalState.UNKNOWN)

    def test_fresh_phone_and_wrist_produce_held_candidate(self) -> None:
        fusion = ObservationFusion()
        fusion.update_pose(10.0, 1, (NormalizedPoint(0.52, 0.5, 0.9),))
        fusion.update_phones(
            10.0,
            (PhoneCandidate(NormalizedBox(0.4, 0.4, 0.1, 0.2), 0.8),),
        )
        signal = fusion.build(10.1).signal("phone_held")
        self.assertIs(signal.state, SignalState.VIOLATION)

    def test_phone_presence_without_fresh_pose_is_unknown(self) -> None:
        fusion = ObservationFusion()
        fusion.update_phones(
            10.0,
            (PhoneCandidate(NormalizedBox(0.4, 0.4, 0.1, 0.2), 0.8),),
        )
        signal = fusion.build(10.1).signal("phone_held")
        self.assertIs(signal.state, SignalState.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
