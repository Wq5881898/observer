from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np

from observer.observations import SignalState, SignalValue
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

# Eyelid aperture is useful for rejecting blinks and poorly observed eyes, but it
# is not a stable gaze-coordinate input.  Keeping it in the regression lets a
# short blink or fatigue-related squint move the predicted screen position.
ATTENTION_NON_POSITIONAL_FEATURES = frozenset({"eye_a_open", "eye_b_open"})

# Regression normalization measures how point medians differ inside one session.
# It can be much tighter than a person's attainable position after sitting down
# again, so it must not be used as an unboundedly small context envelope.
# At the default 5 z limit these floors allow 2% frame width, 3% frame height,
# and 1% normalized face scale while still rejecting material session shifts.
ATTENTION_CONTEXT_SCALE_FLOORS: Mapping[str, float] = {
    "face_center_x": 0.004,
    "face_center_y": 0.006,
    # A person cannot reliably reproduce the sub-percent scale variation seen
    # within one calibration capture. At the default 5 z guard this permits
    # about +/-0.025 normalized scale (roughly +/-8.3% around a 0.30 face).
    "face_scale": 0.005,
}

# Head proxies are normalized by facial geometry. They supplement iris gaze
# only after a neutral baseline has been captured in the same camera session.
ATTENTION_HEAD_YAW_DELTA_THRESHOLD = 0.22
ATTENTION_HEAD_PITCH_DOWN_DELTA_THRESHOLD = 0.07
ATTENTION_HEAD_PITCH_STRONG_DOWN_DELTA_THRESHOLD = 0.14
ATTENTION_GAZE_DOWN_BOUNDARY = 1.0


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
        group_feature_scores: dict[str, list[float]] = {}
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
                group_feature_scores[group] = scores

        if len(group_scores) < self.minimum_groups_outside:
            return SignalValue.unknown("insufficient observable posture feature groups")
        outside = sum(score >= self.z_threshold for score in group_scores.values())
        alignment_features_outside = sum(
            score >= self.z_threshold
            for score in group_feature_scores.get("alignment", ())
        )
        metrics["posture_groups_observable"] = float(len(group_scores))
        metrics["posture_groups_outside"] = float(outside)
        metrics["posture_alignment_features_outside"] = float(
            alignment_features_outside
        )
        confidence = min(1.0, max(group_scores.values()) / (self.z_threshold * 1.5))
        # A lateral lean primarily changes one feature group. Require both head
        # offset and shoulder slope to agree so ordinary one-feature jitter does
        # not weaken the normal two-group false-positive guard.
        corroborated_lateral_lean = alignment_features_outside >= 2
        if outside >= self.minimum_groups_outside or corroborated_lateral_lean:
            return SignalValue.violation(confidence=confidence, **metrics)
        return SignalValue.compliant(confidence=1.0 - min(confidence, 0.49), **metrics)


@dataclass(frozen=True, slots=True)
class AttentionSessionBaseline:
    head_yaw_proxy: float
    head_pitch_proxy: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.head_yaw_proxy) or not math.isfinite(
            self.head_pitch_proxy
        ):
            raise ValueError("attention session baseline must be finite")


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

    def context_statistics(self) -> dict[str, tuple[float, float]]:
        """Return mean and practical scale used by the session context guard."""

        result: dict[str, tuple[float, float]] = {}
        for name in ATTENTION_CONTEXT_SCALE_FLOORS:
            if name not in self.feature_names:
                continue
            index = self.feature_names.index(name)
            result[name] = (
                self.feature_means[index],
                max(
                    self.feature_scales[index],
                    ATTENTION_CONTEXT_SCALE_FLOORS[name],
                ),
            )
        return result

    def context_scores(self, features: FeatureVector) -> dict[str, float]:
        return {
            name: abs(features.values[name] - mean) / scale
            for name, (mean, scale) in self.context_statistics().items()
            if name in features.values
        }

    def predictive_context_scores(self, features: FeatureVector) -> dict[str, float]:
        """Return guards only for context inputs that can move the prediction.

        A zero-weight context input remains available to the pre-session
        alignment check, but must not invalidate a deliberate look-away action
        merely because that action moves the face in the camera frame.
        """

        scores = self.context_scores(features)
        return {
            name: score
            for name, score in scores.items()
            if (
                abs(self.coefficients_x[self.feature_names.index(name) + 1]) > 1e-12
                or abs(self.coefficients_y[self.feature_names.index(name) + 1]) > 1e-12
            )
        }

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
        context_scores = self.context_scores(features)
        maximum_context_score = max(context_scores.values(), default=0.0)
        maximum_predictive_context_score = max(
            self.predictive_context_scores(features).values(), default=0.0
        )
        if maximum_predictive_context_score > self.maximum_context_z:
            return SignalValue.unknown(
                "face position or scale is outside the attention calibration envelope",
                attention_context_z=maximum_context_score,
            )
        x, y = self.predict(features)
        margin = self.allowed_margin
        outside_distance = max(-margin - x, x - (1.0 + margin), -margin - y, y - (1.0 + margin), 0.0)
        metrics = {
            "attention_screen_x": x,
            "attention_screen_y": y,
            "attention_context_z": maximum_context_score,
        }
        if outside_distance > 0:
            confidence = min(1.0, 0.55 + outside_distance / max(margin * 2.0, 1e-6))
            return SignalValue.violation(confidence=confidence, **metrics)
        edge_distance = min(x + margin, 1.0 + margin - x, y + margin, 1.0 + margin - y)
        confidence = min(1.0, max(0.51, edge_distance / max(0.5 + margin, 1e-6)))
        return SignalValue.compliant(confidence=confidence, **metrics)

    def evaluate_with_session_baseline(
        self,
        features: FeatureVector,
        baseline: AttentionSessionBaseline | None,
    ) -> SignalValue:
        """Fuse calibrated iris gaze with same-session relative head motion."""

        gaze_signal = self.evaluate(features)
        if baseline is None or not {
            "head_yaw_proxy",
            "head_pitch_proxy",
        }.issubset(features.values):
            return gaze_signal

        yaw_delta = features.values["head_yaw_proxy"] - baseline.head_yaw_proxy
        pitch_delta = features.values["head_pitch_proxy"] - baseline.head_pitch_proxy
        metrics = {
            **gaze_signal.metrics,
            "attention_head_yaw_delta": yaw_delta,
            "attention_head_pitch_delta": pitch_delta,
        }
        yaw_ratio = abs(yaw_delta) / ATTENTION_HEAD_YAW_DELTA_THRESHOLD
        screen_x = gaze_signal.metrics.get("attention_screen_x")
        screen_y = gaze_signal.metrics.get("attention_screen_y")
        horizontal_gaze_away = screen_x is not None and (
            screen_x < -self.allowed_margin or screen_x > 1.0 + self.allowed_margin
        )
        upward_gaze_away = (
            screen_y is not None and screen_y < -self.allowed_margin
        )
        combined_down_away = (
            pitch_delta >= ATTENTION_HEAD_PITCH_DOWN_DELTA_THRESHOLD
            and screen_y is not None
            and screen_y > ATTENTION_GAZE_DOWN_BOUNDARY
        )
        strong_down_away = (
            pitch_delta >= ATTENTION_HEAD_PITCH_STRONG_DOWN_DELTA_THRESHOLD
        )
        head_or_directional_gaze_away = (
            yaw_ratio >= 1.0
            or horizontal_gaze_away
            or upward_gaze_away
            or combined_down_away
            or strong_down_away
        )
        if head_or_directional_gaze_away:
            pitch_ratio = max(
                pitch_delta / ATTENTION_HEAD_PITCH_DOWN_DELTA_THRESHOLD,
                0.0,
            )
            confidence = min(
                1.0, 0.55 + 0.25 * (max(yaw_ratio, pitch_ratio) - 1.0)
            )
            return SignalValue.violation(confidence=confidence, **metrics)
        if gaze_signal.state is SignalState.COMPLIANT:
            return SignalValue.compliant(gaze_signal.confidence, **metrics)
        if gaze_signal.state is SignalState.VIOLATION:
            # A downward-only iris extrapolation is not sufficient: horizontal
            # head turns can create a false vertical prediction. The combined
            # and strong-down cases were handled above.
            return SignalValue.compliant(confidence=0.51, **metrics)
        return SignalValue.unknown(
            gaze_signal.reason or "attention is not observable", **metrics
        )


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
    ridge: float = 3.0,
) -> AttentionCalibration:
    valid = [sample for sample in samples if sample.features.observable]
    if len(valid) < 9:
        raise ValueError("attention calibration needs at least 9 valid samples")
    versions = {sample.features.version for sample in valid}
    if len(versions) != 1:
        raise ValueError("attention calibration samples use mixed feature versions")
    common_feature_names = set.intersection(
        *(set(item.features.values) for item in valid)
    )
    feature_names = tuple(
        sorted(common_feature_names - ATTENTION_NON_POSITIONAL_FEATURES)
    )
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
