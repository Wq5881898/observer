from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from observer.calibration import AttentionSessionBaseline, load_profile
from observer.live_validation import SCENARIOS, scenario_validation_signal
from observer.perception import FeatureVector


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    parser.add_argument("batch", type=Path)
    args = parser.parse_args()

    profile = load_profile(args.profile)
    trials = [
        json.loads(line)
        for line in (args.batch / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    scenario_by_trial = {
        str(trial["trial_id"]): str(trial["scenario_id"]) for trial in trials
    }
    scenario_by_id = {scenario.scenario_id: scenario for scenario in SCENARIOS}
    records = [
        json.loads(line)
        for line in (args.batch / "derived_samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    center_records = [
        record
        for record in records
        if scenario_by_trial[str(record["trial_id"])] == "attention_center"
        and {"head_yaw_proxy", "head_pitch_proxy"}.issubset(
            record.get("attention_smoothed") or {}
        )
    ]
    if not center_records:
        raise RuntimeError("batch has no observable attention_center baseline")
    baseline = AttentionSessionBaseline(
        median(
            record["attention_smoothed"]["head_yaw_proxy"]
            for record in center_records
        ),
        median(
            record["attention_smoothed"]["head_pitch_proxy"]
            for record in center_records
        ),
    )

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    metric_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        trial_id = str(record["trial_id"])
        scenario_id = scenario_by_trial[trial_id]
        if scenario_id not in scenario_by_id:
            continue
        values = record.get("attention_smoothed") or {}
        reasons = tuple(record.get("attention_smoothed_reasons") or ())
        features = FeatureVector("attention", values, reasons=reasons)
        raw_signal = profile.attention.evaluate_with_session_baseline(
            features, baseline
        )
        signal = scenario_validation_signal(scenario_by_id[scenario_id], raw_signal)
        counts[scenario_id][signal.state.value] += 1
        for name, value in raw_signal.metrics.items():
            metric_values[scenario_id][name].append(float(value))

    def distribution(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        return {
            "minimum": ordered[0],
            "p10": ordered[int((len(ordered) - 1) * 0.10)],
            "median": median(ordered),
            "maximum": ordered[-1],
        }

    print(
        json.dumps(
            {
                "baseline": {
                    "head_yaw_proxy": baseline.head_yaw_proxy,
                    "head_pitch_proxy": baseline.head_pitch_proxy,
                },
                "scenarios": {
                    scenario_id: {
                        "states": dict(state_counts),
                        "metrics": {
                            name: distribution(values)
                            for name, values in metric_values[scenario_id].items()
                        },
                    }
                    for scenario_id, state_counts in counts.items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
