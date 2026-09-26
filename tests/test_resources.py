import unittest

from observer.config import ResourceBudget
from observer.resources import ResourceSnapshot, SessionResourceTracker, summarize_resources


def sample(at: float, cpu: float, memory: float) -> ResourceSnapshot:
    return ResourceSnapshot(at, cpu, memory, 30.0, 4000.0)


class ResourceSummaryTests(unittest.TestCase):
    def test_summary_within_budget(self) -> None:
        result = summarize_resources(
            [sample(0.0, 5.0, 100.0), sample(900.0, 10.0, 105.0)],
            ResourceBudget(),
        )
        self.assertTrue(result.within_budget)
        self.assertEqual(result.sample_count, 2)
        self.assertEqual(result.process_memory_growth_mb_per_hour, 20.0)

    def test_short_run_does_not_extrapolate_memory_growth(self) -> None:
        result = summarize_resources(
            [sample(0.0, 5.0, 100.0), sample(15.0, 5.0, 110.0)],
            ResourceBudget(),
        )
        self.assertIsNone(result.process_memory_growth_mb_per_hour)

    def test_summary_reports_cpu_and_memory_violations(self) -> None:
        result = summarize_resources(
            [sample(0.0, 30.0, 490.0), sample(900.0, 30.0, 520.0)],
            ResourceBudget(),
        )
        self.assertFalse(result.within_budget)
        self.assertIn("average CPU budget exceeded", result.violations)
        self.assertIn("steady memory budget exceeded", result.violations)

    def test_session_tracker_samples_on_interval_and_at_finish(self) -> None:
        values = iter([
            sample(0.0, 2.0, 100.0),
            sample(5.0, 4.0, 101.0),
            sample(6.0, 6.0, 102.0),
        ])

        class Sampler:
            def sample(self):
                return next(values)

        tracker = SessionResourceTracker(ResourceBudget(), Sampler, interval_seconds=5)
        tracker.begin(0)
        tracker.sample(1)
        tracker.sample(5)
        result = tracker.finish(6)
        self.assertEqual(result['sample_count'], 3)
        self.assertEqual(result['process_cpu_average_percent'], 4)
        self.assertEqual(result['process_memory_peak_mb'], 102)
        self.assertIsNone(result['sampling_error'])

    def test_session_tracker_isolates_sampler_failure(self) -> None:
        def unavailable():
            raise RuntimeError('metrics unavailable')

        tracker = SessionResourceTracker(ResourceBudget(), unavailable)
        tracker.begin(0)
        result = tracker.finish(1)
        self.assertEqual(result['sample_count'], 0)
        self.assertEqual(result['sampling_error'], 'metrics unavailable')


if __name__ == "__main__":
    unittest.main()
