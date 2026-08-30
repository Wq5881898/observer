import unittest

from observer.frames import LatestFrameBuffer


class LatestFrameBufferTests(unittest.TestCase):
    def test_new_frame_replaces_old_frame(self) -> None:
        buffer = LatestFrameBuffer()
        first = buffer.publish("first", 1.0)
        second = buffer.publish("second", 2.0)

        latest = buffer.latest()
        self.assertIsNotNone(latest)
        self.assertEqual(latest.frame, "second")
        self.assertEqual(latest.sequence, first.sequence + 1)
        self.assertEqual(second, latest)

    def test_after_sequence_avoids_duplicate_processing(self) -> None:
        buffer = LatestFrameBuffer()
        snapshot = buffer.publish(object(), 1.0)
        self.assertIsNone(buffer.latest(after_sequence=snapshot.sequence))


if __name__ == "__main__":
    unittest.main()

