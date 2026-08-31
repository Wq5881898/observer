import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from observer.validation import (
    ErrorEvidenceStore,
    EvidencePolicy,
    TrialRecord,
    read_trials,
    summarize_trials,
    purge_expired_evidence,
    new_batch_manifest,
)


def trial(trial_id: str, truth: str, prediction: str) -> TrialRecord:
    return TrialRecord(
        trial_id=trial_id,
        scenario_id="synthetic",
        signal_name="phone_held",
        dataset_role="holdout",
        truth_state=truth,
        predicted_state=prediction,
        started_monotonic=1.0,
        ended_monotonic=11.0,
        observable_fraction=1.0 if prediction != "unknown" else 0.0,
    )


class ValidationTests(unittest.TestCase):
    def test_manifest_records_same_session_alignment(self) -> None:
        manifest = new_batch_manifest(
            batch_id="tuning-test",
            dataset_role="tuning",
            git_commit="abc",
            config_sha256="config",
            model_manifest_sha256="models",
            calibration_profile_id="profile",
            environment={"camera_index": 0},
            alignment={"final_context_z": 1.8, "frames_saved": 0},
        )

        self.assertEqual(manifest.alignment["final_context_z"], 1.8)
        self.assertEqual(manifest.alignment["frames_saved"], 0)

    def test_metrics_count_unknown_positive_as_false_negative(self) -> None:
        metrics = summarize_trials(
            [
                trial("tp", "violation", "violation"),
                trial("fn", "violation", "unknown"),
                trial("tn", "compliant", "compliant"),
                trial("fp", "compliant", "violation"),
            ],
            "phone_held",
        )
        self.assertEqual(metrics.true_positive, 1)
        self.assertEqual(metrics.false_negative, 1)
        self.assertEqual(metrics.false_positive, 1)
        self.assertAlmostEqual(metrics.precision, 0.5)
        self.assertAlmostEqual(metrics.recall, 0.5)

    def test_metrics_expose_expected_unknown_safety_failure(self) -> None:
        metrics = summarize_trials(
            [
                trial("covered", "unknown", "violation"),
                trial("dark", "unknown", "unknown"),
            ],
            "phone_held",
        )

        self.assertEqual(metrics.expected_unknown_trials, 2)
        self.assertEqual(metrics.expected_unknown_failures, 1)

    def test_jsonl_rejects_invalid_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trials.jsonl"
            path.write_text('{"truth_state":"made-up"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1"):
                read_trials(path)

    def test_evidence_is_opt_in_and_one_frame_per_trial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            frame = np.zeros((20, 20, 3), dtype=np.uint8)
            disabled = ErrorEvidenceStore(Path(temporary) / "off")
            self.assertIsNone(disabled.save("t1", frame, is_error=True))
            enabled = ErrorEvidenceStore(
                Path(temporary) / "on", EvidencePolicy(enabled=True)
            )
            saved = enabled.save("t1", frame, is_error=True)
            self.assertIsNotNone(saved)
            self.assertIsNone(enabled.save("t1", frame, is_error=True))
            self.assertEqual(len(list((Path(temporary) / "on").glob("*.jpg"))), 1)

    def test_evidence_purge_is_dry_run_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            image = directory / "expired.jpg"
            image.write_bytes(b"test")
            index = directory / "index.jsonl"
            index.write_text(
                '{"relative_path":"expired.jpg","delete_after_utc":"2020-01-01T00:00:00+00:00"}\n',
                encoding="utf-8",
            )
            preview = purge_expired_evidence(
                index, now_utc=datetime(2026, 1, 1, tzinfo=timezone.utc)
            )
            self.assertTrue(image.exists())
            self.assertEqual(preview["expired"], ["expired.jpg"])
            purge_expired_evidence(
                index, apply=True, now_utc=datetime(2026, 1, 1, tzinfo=timezone.utc)
            )
            self.assertFalse(image.exists())


if __name__ == "__main__":
    unittest.main()
