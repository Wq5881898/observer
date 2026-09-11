import unittest

from scripts.audit_c_closeout_holdout import CHECKS, evaluate_rows


class CloseoutHoldoutTests(unittest.TestCase):
    def _passing_rows(self):
        rows = []
        cue_starts = {
            "h01_neutral": 0.0,
            "h02_away": 15.0,
            "h03_away_recovery": 45.0,
            "h04_phone": 60.0,
            "h05_phone_recovery": 80.0,
            "h06_face_cover": 95.0,
            "h07_face_recovery": 108.0,
            "h08_camera_cover": 121.0,
            "h09_final": 134.0,
        }
        cue_durations = {
            "h01_neutral": 10,
            "h02_away": 25,
            "h03_away_recovery": 10,
            "h04_phone": 15,
            "h05_phone_recovery": 10,
            "h06_face_cover": 8,
            "h07_face_recovery": 8,
            "h08_camera_cover": 8,
            "h09_final": 10,
        }
        for cue_id, start in cue_starts.items():
            for offset in range(cue_durations[cue_id] + 1):
                away = "compliant"
                phone = "compliant"
                if cue_id == "h02_away":
                    away = "violation"
                elif cue_id == "h08_camera_cover":
                    away = "unknown"
                if cue_id == "h04_phone":
                    phone = "violation"
                rows.append(
                    {
                        "elapsed_seconds": start + offset,
                        "guided_cue": {"cue_id": cue_id},
                        "signals": {
                            "away": {"state": away},
                            "phone_held": {"state": phone},
                        },
                    }
                )
        return rows

    def test_passing_states_still_require_manual_truth_confirmation(self) -> None:
        report = evaluate_rows(self._passing_rows())

        self.assertTrue(report["automated_checks_passed"])
        self.assertTrue(report["requires_manual_truth_confirmation"])
        self.assertFalse(report["eligible_for_acceptance"])
        self.assertEqual(len(report["checks"]), len(CHECKS))

    def test_fragmented_phone_detection_fails_continuity(self) -> None:
        rows = self._passing_rows()
        for row in rows:
            if row["guided_cue"]["cue_id"] == "h04_phone" and int(
                row["elapsed_seconds"]
            ) % 3 == 0:
                row["signals"]["phone_held"]["state"] = "compliant"

        report = evaluate_rows(rows)
        phone = next(item for item in report["checks"] if item["check_id"] == "H04-phone")

        self.assertFalse(phone["passed"])
        self.assertLess(phone["longest_expected_run_seconds"], 10.0)
        self.assertFalse(report["automated_checks_passed"])


if __name__ == "__main__":
    unittest.main()
