import unittest

from observer.observations import (
    NormalizedBox,
    NormalizedPoint,
    PhoneCandidate,
    SignalState,
    phone_held_signal,
)


class PhoneHeldSignalTests(unittest.TestCase):
    def test_phone_near_visible_wrist_is_violation(self) -> None:
        signal = phone_held_signal(
            (PhoneCandidate(NormalizedBox(0.4, 0.4, 0.1, 0.2), 0.9),),
            (NormalizedPoint(0.52, 0.5, 0.8),),
        )
        self.assertIs(signal.state, SignalState.VIOLATION)
        self.assertAlmostEqual(signal.confidence or 0.0, 0.8)

    def test_phone_far_from_wrist_is_compliant(self) -> None:
        signal = phone_held_signal(
            (PhoneCandidate(NormalizedBox(0.1, 0.1, 0.1, 0.1), 0.9),),
            (NormalizedPoint(0.9, 0.9, 0.9),),
        )
        self.assertIs(signal.state, SignalState.COMPLIANT)

    def test_phone_without_observable_wrist_is_unknown(self) -> None:
        signal = phone_held_signal(
            (PhoneCandidate(NormalizedBox(0.4, 0.4, 0.1, 0.2), 0.9),),
            (NormalizedPoint(0.5, 0.5, 0.2),),
        )
        self.assertIs(signal.state, SignalState.UNKNOWN)

    def test_no_phone_is_not_a_violation(self) -> None:
        signal = phone_held_signal((), ())
        self.assertIs(signal.state, SignalState.COMPLIANT)


if __name__ == "__main__":
    unittest.main()

