from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from observer.calibration import (
    ATTENTION_CONTEXT_SCALE_FLOORS,
    AttentionCalibration,
    AttentionSample,
    aggregate_attention_samples,
    fit_attention_calibration,
    load_profile,
    save_profile,
)
from observer.perception import FeatureVector


FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "iris_only": ("iris_a_x", "iris_a_y", "iris_b_x", "iris_b_y"),
    "iris_head": (
        "iris_a_x",
        "iris_a_y",
        "iris_b_x",
        "iris_b_y",
        "head_yaw_proxy",
        "head_pitch_proxy",
    ),
    "full_current": (
        "iris_a_x",
        "iris_a_y",
        "iris_b_x",
        "iris_b_y",
        "head_yaw_proxy",
        "head_pitch_proxy",
        "face_center_x",
        "face_center_y",
        "face_scale",
    ),
}


def _samples(
    records: list[dict[str, object]], names: tuple[str, ...]
) -> list[AttentionSample]:
    samples = [
        AttentionSample(
            FeatureVector(
                "attention",
                {name: float(record["features"][name]) for name in names},
            ),
            float(record["target_x"]),
            float(record["target_y"]),
            str(record["point_id"]),
        )
        for record in records
    ]
    return aggregate_attention_samples(samples)


def analyze(
    path: Path, probe_values: dict[str, float] | None = None
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, object] = {}
    for label, names in FEATURE_GROUPS.items():
        training = _samples(payload["attention"], names)
        validation = _samples(payload["attention_validation"], names)
        variants: dict[str, object] = {}
        for ridge in (1.0, 3.0, 10.0):
            model = fit_attention_calibration(
                training,
                validation_samples=validation,
                ridge=ridge,
            )
            center = next(sample for sample in training if sample.point_id == "center")
            training_predictions = [model.predict(sample.features) for sample in training]
            validation_predictions = [
                model.predict(sample.features) for sample in validation
            ]
            variants[f"ridge_{ridge:g}"] = {
                "training_rmse": model.training_rmse,
                "validation_rmse": model.validation_rmse,
                "center_prediction": model.predict(center.features),
                "training_prediction_range": {
                    "x": (
                        min(value[0] for value in training_predictions),
                        max(value[0] for value in training_predictions),
                    ),
                    "y": (
                        min(value[1] for value in training_predictions),
                        max(value[1] for value in training_predictions),
                    ),
                },
                "validation_prediction_range": {
                    "x": (
                        min(value[0] for value in validation_predictions),
                        max(value[0] for value in validation_predictions),
                    ),
                    "y": (
                        min(value[1] for value in validation_predictions),
                        max(value[1] for value in validation_predictions),
                    ),
                },
                "probe_prediction": (
                    model.predict(FeatureVector("attention", probe_values))
                    if probe_values is not None
                    and all(name in probe_values for name in names)
                    else None
                ),
            }
        result[label] = {
            "feature_names": names,
            "variants": variants,
        }
    return result


def write_candidate_profile(
    samples_path: Path,
    source_profile_path: Path,
    output_path: Path,
    *,
    feature_group: str,
    ridge: float,
) -> None:
    payload = json.loads(samples_path.read_text(encoding="utf-8"))
    names = FEATURE_GROUPS[feature_group]
    training = _samples(payload["attention"], names)
    validation = _samples(payload["attention_validation"], names)
    source = load_profile(source_profile_path)
    attention = fit_attention_calibration(
        training,
        validation_samples=validation,
        valid_frame_count=source.attention.valid_sample_count,
        planned_sample_count=source.attention.valid_sample_count,
        ridge=ridge,
    )
    # Keep face position/scale in the serialized model so the normal session
    # alignment and context guard still work, but give them zero regression
    # weight. They describe whether the camera setup is still comparable; they
    # must not move the predicted gaze point themselves.
    source_index = {
        name: index for index, name in enumerate(source.attention.feature_names)
    }
    context_names = tuple(
        name
        for name in ATTENTION_CONTEXT_SCALE_FLOORS
        if name not in attention.feature_names and name in source_index
    )
    attention = AttentionCalibration(
        feature_version=attention.feature_version,
        feature_names=(*attention.feature_names, *context_names),
        feature_means=(
            *attention.feature_means,
            *(source.attention.feature_means[source_index[name]] for name in context_names),
        ),
        feature_scales=(
            *attention.feature_scales,
            *(source.attention.feature_scales[source_index[name]] for name in context_names),
        ),
        coefficients_x=(*attention.coefficients_x, *(0.0 for _ in context_names)),
        coefficients_y=(*attention.coefficients_y, *(0.0 for _ in context_names)),
        training_rmse=attention.training_rmse,
        validation_rmse=attention.validation_rmse,
        valid_sample_count=attention.valid_sample_count,
        coverage=attention.coverage,
        allowed_margin=source.attention.allowed_margin,
        maximum_context_z=source.attention.maximum_context_z,
    )
    save_profile(replace(source, attention=attention), output_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "samples",
        type=Path,
        nargs="?",
        default=Path("data/calibration/default.samples.json"),
    )
    parser.add_argument(
        "--probe-feature",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="optional observed feature used for cross-session prediction",
    )
    parser.add_argument("--source-profile", type=Path)
    parser.add_argument("--write-profile", type=Path)
    parser.add_argument(
        "--candidate-group", choices=tuple(FEATURE_GROUPS), default="iris_only"
    )
    parser.add_argument("--candidate-ridge", type=float, default=1.0)
    args = parser.parse_args()
    probe_values = {
        name: float(value)
        for item in args.probe_feature
        for name, value in (item.split("=", 1),)
    }
    if args.write_profile is not None:
        if args.source_profile is None:
            parser.error("--source-profile is required with --write-profile")
        write_candidate_profile(
            args.samples,
            args.source_profile,
            args.write_profile,
            feature_group=args.candidate_group,
            ridge=args.candidate_ridge,
        )
    print(json.dumps(analyze(args.samples, probe_values or None), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
