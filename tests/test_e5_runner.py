"""Gate verdicts must not accept a partial run or slow/native-resource exit."""
import importlib.util
from pathlib import Path
import unittest

_spec = importlib.util.spec_from_file_location(
    'observer_e5_runner_test', Path(__file__).resolve().parents[1] / 'scripts/run_e5_background.py')
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)


class E5VerdictTests(unittest.TestCase):
    def valid(self):
        return dict(requested_seconds=3600, actual_seconds=3600.01, samples=3590,
                    within_cpu_memory_budget=True, product_resources={'within_budget': True},
                    camera_released_after_end=True, camera_thread_stopped=True,
                    runtime_lock_released=True, exit_seconds=.5, session_state='ended')

    def test_complete_gate_passes(self):
        self.assertTrue(_runner.result_passes(self.valid()))

    def test_incomplete_run_or_no_samples_cannot_pass(self):
        for changes in ({'actual_seconds': 3599}, {'samples': 0}):
            self.assertFalse(_runner.result_passes({**self.valid(), **changes}))

    def test_slow_exit_or_unreleased_resource_cannot_pass(self):
        for changes in ({'exit_seconds': 5.01}, {'camera_thread_stopped': False},
                        {'camera_released_after_end': False}, {'runtime_lock_released': False}):
            self.assertFalse(_runner.result_passes({**self.valid(), **changes}))

    def test_budget_sampling_error_or_interruption_cannot_pass(self):
        for changes in ({'within_cpu_memory_budget': False},
                        {'product_resources': {'within_budget': False}},
                        {'product_resources': {'within_budget': True, 'sampling_error': 'unavailable'}},
                        {'session_state': 'interrupted'}):
            self.assertFalse(_runner.result_passes({**self.valid(), **changes}))
