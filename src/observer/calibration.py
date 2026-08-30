from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np

from observer.observations import SignalValue
from observer.perception import FeatureVector, QualityThresholds


CALIBRATION_SCHEMA_VERSION = 1
IMPLEMENTATION_VERSION = "0.1.0-c1"


@dataclass(frozen=True, slots=True)
class EnvironmentFingerprint:
    camera_index: int
    frame_width: int
    frame_height: int
    mirrored: bool
    screen_width: int
    screen_height: int
    display_scale_percent: int = 100

    def __post_init__(self) -> None:
        if self.camera_index < 0:
            raise ValueError("camera index must be non-negative")
        if min(self.frame_width, self.frame_height, self.screen_width, self.screen_height) <= 0:
            raise ValueError("frame and screen dimensions must be positive")
        if not 50 <= self.display_scale_percent <= 500:
            raise ValueError("display scale must be between 50 and 500 percent")

    def mismatches(self, other: "EnvironmentFingerprint") -> tuple[str, ...]:
        return tuple(
            field
            for field in self.__dataclass_fields__
            if getattr(self, field) != getattr(other, field)
        )


@dataclass(frozen=True, slots=True)
class FeatureStats:
    median: float
    mad: float
    p05: float
    p95: float
    sample_count: int

    def __post_init__(self) -> None:
        values = (self.median, self.mad, self.p05, self.p95)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("posture feature statistics must be finite")
        if self.mad < 0 or self.sample_count < 1 or not self.p05 <= self.median <= self.p95:
            raise ValueError("posture feature statistics are inconsistent")


POSTURE_GROUPS: Mapping[str, tuple[str, ...]] = {
    "alignment": ("shoulder_slope", "head_offset_x"),
    "vertical": ("head_offset_y", "torso_lean"),
    "distance": ("shoulder_width", "face_to_shoulder_scale"),
}

POSTURE_SCALE_FLOORS: Mapping[str, float] = {
    "shoulder_slope": 0.025,
    "head_offset_x": 0.025,
    "head_offset_y": 0.025,
    "torso_lean": 0.025,
    "shoulder_width": 0.008,
    "face_to_shoulder_scale": 0.025,
}


@dataclass(frozen=True, slots=True)
class PostureCalibration:
    feature_version: str
    stats: Mapping[str, FeatureStats]
    valid_sample_count: int
    coverage: float
    z_threshold: float = 4.5
    minimum_groups_outside: int = 2

    def __post_init__(self) -> None:
        if not self.feature_version or not self.stats or self.valid_sample_count < 1:
            raise ValueError("posture calibration is incomplete")
        if not 0 <= self.coverage <= 1 or self.z_threshold <= 0:
            raise ValueError("posture calibration thresholds or coverage are invalid")
        if not 1 <= self.minimum_groups_outside <= len(POSTURE_GROUPS):
            raise ValueError("posture calibration group vote is invalid")

    def evaluate(self, features: FeatureVector) -> SignalValue:
        if features.kind != "posture" or features.version != self.feature_version:
            return SignalValue.unknown("posture feature version does not match calibration")
        if not features.observable:
            return SignalValue.unknown("; ".join(features.reasons))

        group_scores: dict[str, float] = {}
        metrics: dict[str, float] = {}
        for group, names in POSTURE_GROUPS.items():
            scores: list[float] = []
            for name in names:
                if name not in features.values or name not in self.stats:
                    continue
                stat = self.stats[name]
                robust_scale = max(1.4826 * stat.mad, POSTURE_SCALE_FLOORS.get(name, 1e-3))
                score = abs(features.values[name] - stat.median) / robust_scale
                scores.append(score)
                metrics[f"posture_z_{name}"] = score
            if scores:
                group_scores[group] = max(scores)

        if len(group_scores) < self.minimum_groups_outside:
            return SignalValue.unknown("insufficient observable posture feature groups")
        outside = sum(score >= self.z_threshold for score in group_scores.values())
        metrics["posture_groups_observable"] = float(len(group_scores))
        metrics["posture_groups_outside"] = float(outside)
        confidence = min(1.0, max(group_scores.values()) / (self.z_threshold * 1.5))
        if outside >= self.minimum_groups_outside:
            return SignalValue.violation(confidence=confidence, **metrics)
        return SignalValue.compliant(confidence=1.0 - min(confidence, 0.49), **metrics)


@dataclass(frozen=True, slots=True)
class AttentionCalibration:
    feature_version: str
    feature_names: tuple[str, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    coefficients_x: tuple[float, ...]
    coefficients_y: tuple[float, ...]
    training_rmse: float
    validation_rmse: float | None
    valid_sample_count: int
    coverage: float
    allowed_margin: float = 0.08
    maximum_context_z: float = 5.0

    def __post_init__(self) -> None:
        feature_count = len(self.feature_names)
        if not self.feature_version or feature_count < 1 or len(set(self.feature_names)) != feature_count:
            raise ValueError("attention calibration feature definition is invalid")
        if len(self.feature_means) != feature_count or len(self.feature_scales) != feature_count:
            raise ValueError("attention calibration normalization shape is invalid")
        if len(self.coefficients_x) != feature_count + 1 or len(self.coefficients_y) != feature_count + 1:
            raise ValueError("attention calibration coefficient shape is invalid")
        numeric = (
            *self.feature_means,
            *self.feature_scales,
            *self.coefficients_x,
            *self.coefficients_y,
            self.training_rmse,
            self.coverage,
            self.allowed_margin,
            self.maximum_context_z,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("attention calibration values must be finite")
        if any(scale <= 0 for scale in self.feature_scales):
            raise ValueError("attention feature scales must be positive")
        if self.validation_rmse is not None and (
            not math.isfinite(self.validation_rmse) or self.validation_rmse < 0
        ):
            raise ValueError("attention validation error is invalid")
        if (
            self.training_rmse < 0
            or self.valid_sample_count < 1
            or not 0 <= self.coverage <= 1
            or not 0 <= self.allowed_margin <= 0.5
            or self.maximum_context_z <= 0
        ):
            raise ValueError("attention calibration metrics are invalid")

    def predict(self, features: FeatureVector) -> tuple[float, float]:
        if features.kind != "attention" or features.version != self.feature_version:
            raise ValueError("attention feature version does not match calibration")
        if not features.observable:
            raise ValueError("attention features are not observable")
        missing = [name for name in self.feature_names if name not in features.values]
        if missing:
            raise ValueError(f"attention features are missing: {missing}")
        normalized = [
            (features.values[name] - mean) / scale
            for name, mean, scale in zip(
                self.feature_names, self.feature_means, self.feature_scales, strict=True
            )
        ]
        design = (1.0, *normalized)
        predicted_x = sum(a * b for a, b in zip(self.coefficients_x, design, strict=True))
        predicted_y = sum(a * b for a, b in zip(self.coefficients_y, design, strict=True))
        return predicted_x, predicted_y

    def evaluate(self, features: FeatureVector) -> SignalValue:
        if features.kind != "attention" or features.version != self.feature_version:
            return SignalValue.unknown(
                "attention feature version does not match calibration"
            )
        if not features.observable:
            return SignalValue.unknown("; ".join(features.reasons))
        missing = [name for name in self.feature_names if name not in features.values]
        if missing:
            return SignalValue.unknown(f"attention features are missing: {missing}")
        context_scores = []
        for name in ("face_center_x", "face_center_y", "face_scale"):
            if name not in self.feature_names or name not in features.values:
                continue
            index = self.feature_names.index(name)
            context_scores.append(
                abs(features.values[name] - self.feature_means[index])
                / self.feature_scales[index]
            )
        maximum_context_score = max(context_scores, default=0.0)
        if maximum_context_score > self.maximum_context_z:
            return SignalValue.unknown(
                "face position or scale is outside the attention calibration envelope",
                attention_context_z=maximum_context_score,
            )
        x, y = self.predict(features)
        margin = self.allowed_margin
        outside_distance = max(-margin - x, x - (1.0 + margin), -margin - y, y - (1.0 + margin), 0.0)
        metrics = {"attention_screen_x": x, "attention_screen_y": y}
        if outside_distance > 0:
            confidence = min(1.0, 0.55 + outside_distance / max(margin * 2.0, 1e-6))
            return SignalValue.violation(confidence=confidence, **metrics)
        edge_distance = min(x + margin, 1.0 + margin - x, y + margin, 1.0 + margin - y)
        confidence = min(1.0, max(0.51, edge_distance / max(0.5 + margin, 1e-6)))
        return SignalValue.compliant(confidence=confidence, **metrics)


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    schema_version: int
    profile_id: str
    created_at_utc: str
    environment: EnvironmentFingerprint
    quality: QualityThresholds
    posture: PostureCalibration
    attention: AttentionCalibration
    implementation_version: str = IMPLEMENTATION_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CALIBRATION_SCHEMA_VERSION:
            raise ValueError("calibration profile schema does not match this implementation")
        if not self.profile_id or not self.implementation_version:
            raise ValueError("calibration profile identity is incomplete")
        try:
            created = datetime.fromisoformat(self.created_at_utc)
        except ValueError as exc:
            raise ValueError("calibration profile creation time is invalid") from exc
        if created.tzinfo is None:
            raise ValueError("calibration profile creation time must include a timezone")

    def assert_environment(self, current: EnvironmentFingerprint) -> None:
        mismatches = self.environment.mismatches(current)
        if mismatches:
            raise ValueError(f"calibration environment mismatch: {', '.join(mismatches)}")


@dataclass(frozen=True, slots=True)
class AttentionSample:
    features: FeatureVector
    target_x: float
    target_y: float
    point_id: str

    def __post_init__(self) -> None:
        if not 0 <= self.target_x <= 1 or not 0 <= self.target_y <= 1:
            raise ValueError("attention target must be normalized to 0..1")


ATTENTION_POINTS_9: tuple[tuple[str, float, float], ...] = (
    ("center", 0.50, 0.50),
    ("top", 0.50, 0.08),
    ("bottom", 0.50, 0.92),
    ("left", 0.08, 0.50),
    ("right", 0.92, 0.50),
    ("top_left", 0.08, 0.08),
    ("top_right", 0.92, 0.08),
    ("bottom_left", 0.08, 0.92),
    ("bottom_right", 0.92, 0.92),
)


def fit_posture_calibration(
    samples: Sequence[FeatureVector],
    *,
    planned_sample_count: int,
    minimum_valid_samples: int = 30,
) -> PostureCalibration:
    valid = [sample for sample in samples if sample.kind == "posture" and sample.observable]
    if len(valid) < minimum_valid_samples:
        raise ValueError(
            f"posture calibration needs at least {minimum_valid_samples} valid samples"
        )
    versions = {sample.version for sample in valid}
    if len(versions) != 1:
        raise ValueError("posture calibration samples use mixed feature versions")
    common_names = set(valid[0].values)
    for sample in valid[1:]:
        common_names.intersection_update(sample.values)
    if not common_names:
        raise ValueError("posture calibration has no common features")
    stats: dict[str, FeatureStats] = {}
    for name in sorted(common_names):
        values = np.asarray([sample.values[name] for sample in valid], dtype=float)
        median = float(np.median(values))
        stats[name] = FeatureStats(
            median=median,
            mad=float(np.median(np.abs(values - median))),
            p05=float(np.quantile(values, 0.05)),
            p95=float(np.quantile(values, 0.95)),
            sample_count=len(values),
        )
    represented_groups = sum(any(name in stats for name in names) for names in POSTURE_GROUPS.values())
    if represented_groups < 2:
        raise ValueError("posture calibration needs at least two independent feature groups")
    coverage = len(valid) / max(planned_sample_count, 1)
    return PostureCalibration(
        feature_version=versions.pop(),
        stats=stats,
        valid_sample_count=len(valid),
        coverage=coverage,
    )


def fit_attention_calibration(
    samples: Sequence[AttentionSample],
    *,
    validation_samples: Sequence[AttentionSample] = (),
    planned_sample_count: int | None = None,
    valid_frame_count: int | None = None,
    ridge: float = 1e-3,
) -> AttentionCalibration:
    valid = [sample for sample in samples if sample.features.observable]
    if len(valid) < 9:
        raise ValueError("attention calibration needs at least 9 valid samples")
    versions = {sample.features.version for sample in valid}
    if len(versions) != 1:
        raise ValueError("attention calibration samples use mixed feature versions")
    feature_names = tuple(sorted(set.intersection(*(set(item.features.values) for item in valid))))
    if not feature_names:
        raise ValueError("attention calibration has no common features")
    matrix = np.asarray(
        [[sample.features.values[name] for name in feature_names] for sample in valid],
        dtype=float,
    )
    target = np.asarray([[sample.target_x, sample.target_y] for sample in valid], dtype=float)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales < 1e-6] = 1.0
    design = np.column_stack((np.ones(len(matrix)), (matrix - means) / scales))
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    train_prediction = design @ coefficients
    training_rmse = float(np.sqrt(np.mean((train_prediction - target) ** 2)))
    model = AttentionCalibration(
        feature_version=versions.pop(),
        feature_names=feature_names,
        feature_means=tuple(float(value) for value in means),
        feature_scales=tuple(float(value) for value in scales),
        coefficients_x=tuple(float(value) for value in coefficients[:, 0]),
        coefficients_y=tuple(float(value) for value in coefficients[:, 1]),
        training_rmse=training_rmse,
        validation_rmse=None,
        valid_sample_count=valid_frame_count or len(valid),
        coverage=(valid_frame_count or len(valid))
        / max(planned_sample_count or valid_frame_count or len(samples), 1),
    )
    valid_holdout = [sample for sample in validation_samples if sample.features.observable]
    if valid_holdout:
        errors = [
            (model.predict(sample.features)[0] - sample.target_x) ** 2
            + (model.predict(sample.features)[1] - sample.target_y) ** 2
            for sample in valid_holdout
        ]
        model = replace(
            model, validation_rmse=math.sqrt(sum(errors) / (2 * len(errors)))
        )
    return model


def create_profile(
    environment: EnvironmentFingerprint,
    posture: PostureCalibration,
    attention: AttentionCalibration,
    quality: QualityThresholds | None = None,
) -> CalibrationProfile:
    return CalibrationProfile(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        profile_id=str(uuid4()),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        environment=environment,
        quality=quality or QualityThresholds(),
        posture=posture,
        attention=attention,
    )


def save_profile(profile: CalibrationProfile, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(asdict(profile), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(target)


def load_profile(
    path: str | Path, current_environment: EnvironmentFingerprint | None = None
) -> CalibrationProfile:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise ValueError(f"unsupported calibration schema: {raw.get('schema_version')!r}")
    try:
        posture_raw = dict(raw["posture"])
        posture_raw["stats"] = {
            name: FeatureStats(**values) for name, values in posture_raw["stats"].items()
        }
        attention_raw = dict(raw["attention"])
        for name in (
            "feature_names",
            "feature_means",
            "feature_scales",
            "coefficients_x",
            "coefficients_y",
        ):
            attention_raw[name] = tuple(attention_raw[name])
        profile = CalibrationProfile(
            schema_version=int(raw["schema_version"]),
            profile_id=str(raw["profile_id"]),
            created_at_utc=str(raw["created_at_utc"]),
            environment=EnvironmentFingerprint(**raw["environment"]),
            quality=QualityThresholds(**raw["quality"]),
            posture=PostureCalibration(**posture_raw),
            attention=AttentionCalibration(**attention_raw),
            implementation_version=str(raw["implementation_version"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid calibration profile: {exc}") from exc
    if current_environment is not None:
        profile.assert_environment(current_environment)
    return profile


def aggregate_attention_samples(samples: Iterable[AttentionSample]) -> list[AttentionSample]:
    """Reduce repeated frames to one median feature vector per target point."""

    grouped: dict[str, list[AttentionSample]] = {}
    for sample in samples:
        if sample.features.observable:
            grouped.setdefault(sample.point_id, []).append(sample)
    result: list[AttentionSample] = []
    for point_id, group in grouped.items():
        names = set.intersection(*(set(item.features.values) for item in group))
        median_values = {
            name: float(np.median([item.features.values[name] for item in group]))
            for name in names
        }
        first = group[0]
        result.append(
            AttentionSample(
                FeatureVector("attention", median_values, version=first.features.version),
                first.target_x,
                first.target_y,
                point_id,
            )
        )
    return result
