"""Audit the fixed Stage C minimal blind holdout trace.

The report verifies saved signal states against the predeclared scored windows.
It never substitutes for the tester's confirmation that the requested actions
were actually performed naturally and for the full scored duration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


CHECKS = (
    ("H01-away", "h01_neutral", "away", "compliant", 0.80, 0.0),
    ("H01-phone", "h01_neutral", "phone_held", "compliant", 0.80, 0.0),
    ("H02-away", "h02_away", "away", "violation", 0.80, 20.0),
    ("H03-away", "h03_away_recovery", "away", "compliant", 0.80, 0.0),
    ("H04-away", "h04_phone", "away", "compliant", 0.80, 0.0),
    ("H04-phone", "h04_phone", "phone_held", "violation", 0.80, 10.0),
    ("H05-phone", "h05_phone_recovery", "phone_held", "compliant", 0.80, 0.0),
    ("H06-away", "h06_face_cover", "away", "compliant", 0.80, 0.0),
    ("H07-away", "h07_face_recovery", "away", "compliant", 0.80, 0.0),
    ("H08-away", "h08_camera_cover", "away", "unknown", 0.80, 0.0),
    ("H09-away", "h09_final", "away", "compliant", 0.80, 0.0),
    ("H09-phone", "h09_final", "phone_held", "compliant", 0.80, 0.0),
)


def _longest_run(rows: list[dict[str, Any]], signal: str, state: str) -> float:
    longest = 0.0
    started: float | None = None
    previous: float | None = None
    for row in rows:
        elapsed = float(row["elapsed_seconds"])
        observed = row.get("signals", {}).get(signal, {}).get("state")
        if observed == state:
            if started is None:
                started = elapsed
            previous = elapsed
        elif started is not None:
            longest = max(longest, max(0.0, (previous or started) - started))
            started = None
            previous = None
    if started is not None:
        longest = max(longest, max(0.0, (previous or started) - started))
    return longest


def evaluate_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        cue = row.get("guided_cue") or {}
        cue_id = cue.get("cue_id")
        if cue_id:
            grouped.setdefault(str(cue_id), []).append(row)

    results: list[dict[str, Any]] = []
    for check_id, cue_id, signal, expected, minimum_ratio, minimum_run in CHECKS:
        selected = grouped.get(cue_id, [])
        counts: dict[str, int] = {}
        for row in selected:
            state = row.get("signals", {}).get(signal, {}).get("state", "missing")
            counts[str(state)] = counts.get(str(state), 0) + 1
        expected_count = counts.get(expected, 0)
        ratio = expected_count / len(selected) if selected else 0.0
        longest = _longest_run(selected, signal, expected)
        passed = bool(selected) and ratio >= minimum_ratio and longest >= minimum_run
        results.append(
            {
                "check_id": check_id,
                "cue_id": cue_id,
                "signal": signal,
                "expected_state": expected,
                "sample_count": len(selected),
                "state_counts": counts,
                "expected_ratio": round(ratio, 6),
                "minimum_ratio": minimum_ratio,
                "longest_expected_run_seconds": round(longest, 3),
                "minimum_run_seconds": minimum_run,
                "passed": passed,
            }
        )

    return {
        "protocol": "observer-stage-c-minimal-blind-holdout-v1",
        "automated_checks_passed": all(item["passed"] for item in results),
        "requires_manual_truth_confirmation": True,
        "eligible_for_acceptance": False,
        "acceptance_note": (
            "Set only after the tester confirms every scored action was natural, "
            "correct, and held for the requested interval."
        ),
        "checks": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.trace.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = evaluate_rows(rows)
    report["trace"] = str(args.trace.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["automated_checks_passed"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
