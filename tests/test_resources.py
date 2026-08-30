import unittest

from observer.config import ResourceBudget
from observer.resources import ResourceSnapshot, summarize_resources


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


if __name__ == "__main__":
    unittest.main()
