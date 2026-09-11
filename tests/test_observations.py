import unittest

from observer.observations import (
    NormalizedBox,
    PhoneCandidate,
    SignalState,
    phone_visible_signal,
)


class PhoneVisibleSignalTests(unittest.TestCase):
    def test_visible_phone_is_violation(self) -> None:
        signal = phone_visible_signal(
            (PhoneCandidate(NormalizedBox(0.4, 0.4, 0.1, 0.2), 0.9),),
        )
        self.assertIs(signal.state, SignalState.VIOLATION)
        self.assertAlmostEqual(signal.confidence or 0.0, 0.9)

    def test_multiple_visible_phones_are_violation(self) -> None:
        signal = phone_visible_signal(
            (
                PhoneCandidate(NormalizedBox(0.1, 0.1, 0.1, 0.1), 0.5),
                PhoneCandidate(NormalizedBox(0.4, 0.4, 0.1, 0.2), 0.9),
            ),
        )
        self.assertIs(signal.state, SignalState.VIOLATION)
        self.assertAlmostEqual(signal.confidence or 0.0, 0.9)

    def test_low_confidence_confirmed_phone_is_still_violation(self) -> None:
        signal = phone_visible_signal(
            (PhoneCandidate(NormalizedBox(0.4, 0.4, 0.1, 0.2), 0.25),),
        )
        self.assertIs(signal.state, SignalState.VIOLATION)

    def test_no_phone_is_not_a_violation(self) -> None:
        signal = phone_visible_signal(())
        self.assertIs(signal.state, SignalState.COMPLIANT)


if __name__ == "__main__":
    unittest.main()
