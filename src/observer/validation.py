from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from observer.models import sha256_file
from observer.observations import SignalState, SignalValue
from observer.resources import percentile


DATASET_ROLES = {"tuning", "holdout"}


@dataclass(frozen=True, slots=True)
class TrialRecord:
    trial_id: str
    scenario_id: str
    signal_name: str
    dataset_role: str
    truth_state: str
    predicted_state: str
    started_monotonic: float
    ended_monotonic: float
    confidence: float | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)
    unknown_reason: str | None = None
    candidate_started_monotonic: float | None = None
    observable_fraction: float = 0.0
    valid: bool = True
    invalid_reason: str | None = None
    evidence_relative_path: str | None = None
    evidence_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.dataset_role not in DATASET_ROLES:
            raise ValueError("dataset role must be tuning or holdout")
        allowed_states = {state.value for state in SignalState}
        if self.truth_state not in allowed_states or self.predicted_state not in allowed_states:
            raise ValueError("trial states must use SignalState values")
        if self.started_monotonic < 0 or self.ended_monotonic < self.started_monotonic:
            raise ValueError("trial monotonic time range is invalid")
        if not 0 <= self.observable_fraction <= 1:
            raise ValueError("observable fraction must be between 0 and 1")
        if self.valid and self.invalid_reason:
            raise ValueError("a valid trial cannot have an invalid reason")
        if not self.valid and not self.invalid_reason:
            raise ValueError("an invalid trial must have an invalid reason")

    @property
    def duration_seconds(self) -> float:
        return self.ended_monotonic - self.started_monotonic

    @classmethod
    def from_signal(
        cls,
        *,
        trial_id: str,
        scenario_id: str,
        signal_name: str,
        dataset_role: str,
        truth_state: SignalState,
        signal: SignalValue,
        started_monotonic: float,
        ended_monotonic: float,
        observable_fraction: float,
        candidate_started_monotonic: float | None = None,
    ) -> "TrialRecord":
        return cls(
            trial_id=trial_id,
            scenario_id=scenario_id,
            signal_name=signal_name,
            dataset_role=dataset_role,
            truth_state=truth_state.value,
            predicted_state=signal.state.value,
            started_monotonic=started_monotonic,
            ended_monotonic=ended_monotonic,
            confidence=signal.confidence,
            metrics=dict(signal.metrics),
            unknown_reason=signal.reason,
            candidate_started_monotonic=candidate_started_monotonic,
            observable_fraction=observable_fraction,
        )


@dataclass(frozen=True, slots=True)
class BatchManifest:
    batch_id: str
    dataset_role: str
    created_at_utc: str
    git_commit: str
    config_sha256: str
    model_manifest_sha256: str
    calibration_profile_id: str
    environment: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.dataset_role not in DATASET_ROLES:
            raise ValueError("dataset role must be tuning or holdout")


@dataclass(frozen=True, slots=True)
class SignalMetrics:
    signal_name: str
    trials: int
    invalid_trials: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    unknown_predictions: int
    precision: float
    recall: float
    f1: float
    observable_fraction: float
    false_positives_per_hour: float
    timing_error_p50_seconds: float | None
    timing_error_p95_seconds: float | None


class JsonlTrialRecorder:
    def __init__(self, batch_directory: str | Path, manifest: BatchManifest) -> None:
        self.directory = Path(batch_directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.records_path = self.directory / "trials.jsonl"
        manifest_path = self.directory / "manifest.json"
        if manifest_path.exists() or self.records_path.exists():
            raise FileExistsError(f"validation batch already exists: {self.directory}")
        manifest_path.write_text(
            json.dumps(asdict(manifest), indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def append(self, record: TrialRecord) -> None:
        with self.records_path.open("a", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True))
            output.write("\n")


def read_trials(path: str | Path) -> list[TrialRecord]:
    records: list[TrialRecord] = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                records.append(TrialRecord(**json.loads(line)))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid trial JSONL at line {line_number}: {exc}") from exc
    return records


def summarize_trials(records: Sequence[TrialRecord], signal_name: str) -> SignalMetrics:
    selected = [record for record in records if record.signal_name == signal_name]
    valid = [record for record in selected if record.valid]
    positive = SignalState.VIOLATION.value
    negative = SignalState.COMPLIANT.value
    unknown_states = {SignalState.UNKNOWN.value, SignalState.EXEMPT.value}
    tp = sum(record.truth_state == positive and record.predicted_state == positive for record in valid)
    fp = sum(record.truth_state == negative and record.predicted_state == positive for record in valid)
    tn = sum(record.truth_state == negative and record.predicted_state == negative for record in valid)
    fn = sum(
        record.truth_state == positive and record.predicted_state != positive for record in valid
    )
    unknown = sum(record.predicted_state in unknown_states for record in valid)
    precision_value = tp / (tp + fp) if tp + fp else 0.0
    recall_value = tp / (tp + fn) if tp + fn else 0.0
    f1_value = (
        2 * precision_value * recall_value / (precision_value + recall_value)
        if precision_value + recall_value
        else 0.0
    )
    negative_hours = sum(
        record.duration_seconds for record in valid if record.truth_state == negative
    ) / 3600.0
    timing_errors = [
        record.candidate_started_monotonic - record.started_monotonic
        for record in valid
        if record.truth_state == positive
        and record.predicted_state == positive
        and record.candidate_started_monotonic is not None
    ]
    coverage = (
        sum(record.observable_fraction * record.duration_seconds for record in valid)
        / sum(record.duration_seconds for record in valid)
        if sum(record.duration_seconds for record in valid) > 0
        else 0.0
    )
    return SignalMetrics(
        signal_name=signal_name,
        trials=len(valid),
        invalid_trials=len(selected) - len(valid),
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        unknown_predictions=unknown,
        precision=precision_value,
        recall=recall_value,
        f1=f1_value,
        observable_fraction=coverage,
        false_positives_per_hour=fp / negative_hours if negative_hours else 0.0,
        timing_error_p50_seconds=percentile(timing_errors, 0.5) if timing_errors else None,
        timing_error_p95_seconds=percentile(timing_errors, 0.95) if timing_errors else None,
    )


def write_metrics_report(records: Sequence[TrialRecord], path: str | Path) -> dict[str, Any]:
    signal_names = sorted({record.signal_name for record in records})
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "signals": {
            name: asdict(summarize_trials(records, name)) for name in signal_names
        },
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return report


@dataclass(frozen=True, slots=True)
class EvidencePolicy:
    enabled: bool = False
    retention_days: int = 7
    maximum_per_trial: int = 1

    def __post_init__(self) -> None:
        if self.retention_days < 1 or self.maximum_per_trial not in {0, 1}:
            raise ValueError("evidence policy exceeds the Stage C privacy boundary")


class ErrorEvidenceStore:
    """Opt-in store for one error frame; it never records video or compliant trials."""

    def __init__(self, directory: str | Path, policy: EvidencePolicy | None = None) -> None:
        self.directory = Path(directory)
        self.policy = policy or EvidencePolicy()
        self._captured_trials: set[str] = set()

    def save(self, trial_id: str, bgr_frame: Any, *, is_error: bool) -> dict[str, str] | None:
        if (
            not self.policy.enabled
            or not is_error
            or bgr_frame is None
            or self.policy.maximum_per_trial == 0
            or trial_id in self._captured_trials
        ):
            return None
        import cv2

        ok, encoded = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("could not encode evidence frame")
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{uuid4()}.jpg"
        path.write_bytes(encoded.tobytes())
        self._captured_trials.add(trial_id)
        delete_after = datetime.now(timezone.utc) + timedelta(days=self.policy.retention_days)
        result = {
            "relative_path": path.name,
            "sha256": sha256_file(path),
            "delete_after_utc": delete_after.isoformat(),
        }
        with (self.directory / "index.jsonl").open(
            "a", encoding="utf-8", newline="\n"
        ) as output:
            output.write(
                json.dumps(
                    {"trial_id": trial_id, **result}, ensure_ascii=False, sort_keys=True
                )
            )
            output.write("\n")
        return result


def new_batch_manifest(
    *,
    batch_id: str,
    dataset_role: str,
    git_commit: str,
    config_sha256: str,
    model_manifest_sha256: str,
    calibration_profile_id: str,
    environment: Mapping[str, Any],
) -> BatchManifest:
    return BatchManifest(
        batch_id=batch_id,
        dataset_role=dataset_role,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=git_commit,
        config_sha256=config_sha256,
        model_manifest_sha256=model_manifest_sha256,
        calibration_profile_id=calibration_profile_id,
        environment=dict(environment),
    )


def purge_expired_evidence(
    index_path: str | Path,
    *,
    apply: bool = False,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """List or delete only expired files named by one evidence index."""

    index = Path(index_path).resolve()
    evidence_directory = index.parent.resolve()
    now = now_utc or datetime.now(timezone.utc)
    expired: list[str] = []
    deleted: list[str] = []
    missing: list[str] = []
    with index.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                delete_after = datetime.fromisoformat(record["delete_after_utc"])
                name = str(record["relative_path"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid evidence index at line {line_number}: {exc}") from exc
            if delete_after.tzinfo is None:
                raise ValueError(f"evidence index line {line_number} has a naive timestamp")
            if delete_after > now:
                continue
            candidate = (evidence_directory / name).resolve()
            if candidate.parent != evidence_directory:
                raise ValueError(f"evidence index line {line_number} escapes its directory")
            expired.append(name)
            if apply:
                if candidate.is_file():
                    candidate.unlink()
                    deleted.append(name)
                else:
                    missing.append(name)
    return {
        "apply": apply,
        "expired": expired,
        "deleted": deleted,
        "missing": missing,
    }
