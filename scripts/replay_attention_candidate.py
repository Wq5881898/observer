from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from observer.calibration import load_profile
from observer.perception import FeatureVector


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    parser.add_argument("samples", type=Path)
    args = parser.parse_args()

    profile = load_profile(args.profile)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    coordinates: dict[str, list[tuple[float, float]]] = defaultdict(list)
    reasons: dict[str, Counter[str]] = defaultdict(Counter)
    observed_features: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with args.samples.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            values = record.get("attention_smoothed") or {}
            feature_reasons = tuple(record.get("attention_smoothed_reasons") or ())
            features = FeatureVector("attention", values, reasons=feature_reasons)
            signal = profile.attention.evaluate(features)
            trial_id = str(record["trial_id"])
            raw_values = record.get("attention") or {}
            for name in (
                "iris_a_x",
                "iris_a_y",
                "iris_b_x",
                "iris_b_y",
                "head_yaw_proxy",
                "head_pitch_proxy",
                "face_center_x",
                "face_center_y",
                "face_scale",
            ):
                if name in raw_values:
                    observed_features[trial_id][name].append(float(raw_values[name]))
            counts[trial_id][signal.state.value] += 1
            if signal.reason:
                reasons[trial_id][signal.reason] += 1
            if "attention_screen_x" in signal.metrics:
                coordinates[trial_id].append(
                    (
                        signal.metrics["attention_screen_x"],
                        signal.metrics["attention_screen_y"],
                    )
                )

    result = {}
    for trial_id, state_counts in counts.items():
        points = coordinates[trial_id]
        result[trial_id] = {
            "states": dict(state_counts),
            "reasons": dict(reasons[trial_id]),
            "median_features": {
                name: median(values)
                for name, values in observed_features[trial_id].items()
            },
            "mean_prediction": (
                [
                    sum(point[0] for point in points) / len(points),
                    sum(point[1] for point in points) / len(points),
                ]
                if points
                else None
            ),
        }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
