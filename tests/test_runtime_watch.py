import unittest
from observer.runtime_watch import RuntimeWatch


class RuntimeWatchTests(unittest.TestCase):
    def test_normal_ticks_and_first_sample(self):
        watch = RuntimeWatch()
        self.assertIsNone(watch.check(1, 100))
        self.assertIsNone(watch.check(1.5, 100.5))

    def test_sleep_with_both_clocks_advancing(self):
        watch = RuntimeWatch()
        watch.check(1, 100)
        self.assertIsNotNone(watch.check(61, 160))
        self.assertIsNone(watch.check(61.1, 160.1))

    def test_sleep_when_monotonic_does_not_advance(self):
        watch = RuntimeWatch()
        watch.check(1, 100)
        self.assertIsNotNone(watch.check(1.1, 160))

    def test_wall_clock_backward_jump(self):
        watch = RuntimeWatch()
        watch.check(1, 100)
        self.assertIsNotNone(watch.check(2, 80))
