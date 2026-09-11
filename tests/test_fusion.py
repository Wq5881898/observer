import unittest

from observer.fusion import ObservationFusion
from observer.observations import (
    NormalizedBox,
    PhoneCandidate,
    SignalState,
    SignalValue,
)


class ObservationFusionTests(unittest.TestCase):
    def test_person_fallback_supports_hidden_face_empty_and_return(self) -> None:
        fusion = ObservationFusion()
        for at, face, pose, person, expected in (
            (10.0, 0, 0, 1, SignalState.COMPLIANT),
            (11.0, 0, 0, 0, SignalState.VIOLATION),
            (12.0, 1, 1, 1, SignalState.COMPLIANT),
        ):
            fusion.update_face(at, face)
            fusion.update_pose(at, pose)
            fusion.update_phones(at, ())
            fusion.update_person_presence(at, person)
            self.assertIs(fusion.build(at + 0.1).signal('away').state, expected)

    def test_phone_updates_do_not_supply_negative_person_evidence(self) -> None:
        fusion = ObservationFusion()
        fusion.update_face(10.0, 0)
        fusion.update_pose(10.0, 0)
        fusion.update_phones(10.0, ())
        self.assertIs(fusion.build(10.1).signal('away').state, SignalState.UNKNOWN)

    def test_phone_updates_do_not_refresh_old_person_result(self) -> None:
        fusion = ObservationFusion()
        fusion.update_person_presence(1.0, 1)
        fusion.update_face(10.0, 0)
        fusion.update_pose(10.0, 0)
        fusion.update_phones(10.0, ())
        self.assertIs(fusion.build(10.1).signal('away').state, SignalState.UNKNOWN)

    def test_person_updates_do_not_refresh_phone_result(self) -> None:
        fusion = ObservationFusion()
        fusion.update_person_presence(10.0, 1)
        self.assertIs(fusion.build(10.1).signal('phone_held').state, SignalState.UNKNOWN)

    def test_person_present_when_either_face_or_pose_is_fresh(self) -> None:
        fusion = ObservationFusion()
        fusion.update_face(10.0, 1)
        observation = fusion.build(10.1)
        self.assertIs(observation.signal("away").state, SignalState.COMPLIANT)

    def test_absence_requires_all_presence_detectors_to_be_fresh_and_empty(self) -> None:
        fusion = ObservationFusion()
        fusion.update_face(10.0, 0)
        self.assertIs(fusion.build(10.1).signal("away").state, SignalState.UNKNOWN)
        fusion.update_pose(10.1, 0)
        self.assertIs(fusion.build(10.2).signal("away").state, SignalState.UNKNOWN)
        fusion.update_person_presence(10.1, 0)
        self.assertIs(fusion.build(10.2).signal("away").state, SignalState.VIOLATION)

    def test_implausible_pose_does_not_count_as_person_presence(self) -> None:
        fusion = ObservationFusion()
        fusion.update_face(10.0, 0)
        fusion.update_pose(10.0, 1, presence_credible=False)
        fusion.update_person_presence(10.0, 0)
        self.assertIs(fusion.build(10.1).signal("away").state, SignalState.VIOLATION)

    def test_person_detector_confirms_presence_without_face_or_pose(self) -> None:
        fusion = ObservationFusion()
        fusion.update_face(10.0, 0)
        fusion.update_pose(10.0, 0)
        fusion.update_person_presence(10.0, 1)
        self.assertIs(fusion.build(10.1).signal("away").state, SignalState.COMPLIANT)

    def test_stale_object_result_is_unknown(self) -> None:
        fusion = ObservationFusion()
        fusion.update_phones(1.0, ())
        signal = fusion.build(10.0).signal("phone_held")
        self.assertIs(signal.state, SignalState.UNKNOWN)

    def test_fresh_visible_phone_is_violation_without_pose(self) -> None:
        fusion = ObservationFusion()
        fusion.update_phones(
            10.0,
            (PhoneCandidate(NormalizedBox(0.4, 0.4, 0.1, 0.2), 0.8),),
        )
        signal = fusion.build(10.1).signal("phone_held")
        self.assertIs(signal.state, SignalState.VIOLATION)

    def test_phone_presence_does_not_require_pose(self) -> None:
        fusion = ObservationFusion()
        fusion.update_phones(
            10.0,
            (PhoneCandidate(NormalizedBox(0.4, 0.4, 0.1, 0.2), 0.8),),
        )
        signal = fusion.build(10.1).signal("phone_held")
        self.assertIs(signal.state, SignalState.VIOLATION)

    def test_no_phone_is_compliant_even_without_hands(self) -> None:
        fusion = ObservationFusion()
        fusion.update_phones(10.0, ())
        self.assertIs(
            fusion.build(10.1).signal("phone_held").state, SignalState.COMPLIANT
        )

    def test_calibrated_signals_flow_through_when_fresh(self) -> None:
        fusion = ObservationFusion()
        fusion.update_quality(10.0, True, None)
        fusion.update_attention(10.0, SignalValue.compliant(confidence=0.9))
        fusion.update_posture(10.0, SignalValue.violation(confidence=0.8))
        observation = fusion.build(10.1)
        self.assertIs(observation.signal("attention_away").state, SignalState.COMPLIANT)
        self.assertIs(observation.signal("posture_deviation").state, SignalState.VIOLATION)

    def test_ergonomic_risk_combines_posture_and_head_gaze_conflict(self) -> None:
        fusion = ObservationFusion()
        fusion.update_quality(10.0, True, None)
        fusion.update_attention(
            10.0, SignalValue.unknown("head pose and iris gaze conflict")
        )
        fusion.update_ergonomic_posture(
            10.0,
            SignalValue.violation(
                confidence=0.8,
                reason="shoulder_imbalance,head_tilt",
            ),
        )

        signal = fusion.build(10.1).signal("ergonomic_risk")

        self.assertIs(signal.state, SignalState.VIOLATION)
        self.assertIn("eye_head_compensation", signal.reason or "")
        self.assertIn("shoulder_imbalance", signal.reason or "")

    def test_ergonomic_risk_can_be_compliant_independently(self) -> None:
        fusion = ObservationFusion()
        fusion.update_quality(10.0, True, None)
        fusion.update_ergonomic_posture(
            10.0, SignalValue.compliant(confidence=0.9)
        )

        signal = fusion.build(10.1).signal("ergonomic_risk")

        self.assertIs(signal.state, SignalState.COMPLIANT)

    def test_failed_quality_gate_suppresses_violations(self) -> None:
        fusion = ObservationFusion()
        fusion.update_quality(10.0, False, "frame too dark")
        fusion.update_face(10.0, 0)
        fusion.update_pose(10.0, 0)
        fusion.update_person_presence(10.0, 0)
        fusion.update_attention(10.0, SignalValue.violation(confidence=1.0))
        fusion.update_posture(10.0, SignalValue.violation(confidence=1.0))
        observation = fusion.build(10.1)
        self.assertTrue(
            all(
                observation.signal(name).state is SignalState.UNKNOWN
                for name in ("away", "attention_away", "posture_deviation")
            )
        )


if __name__ == "__main__":
    unittest.main()
