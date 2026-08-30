from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import numpy as np

from observer.calibration import load_profile
from observer.calibration_workflow import current_environment
from observer.camera import CameraWorker
from observer.config import AppConfig
from observer.models import ModelFiles, sha256_file
from observer.observations import SignalState, SignalValue
from observer.resources import ResourceSampler, summarize_resources
from observer.validation import (
    ErrorEvidenceStore,
    EvidencePolicy,
    JsonlTrialRecorder,
    TrialRecord,
    new_batch_manifest,
    write_metrics_report,
)
from observer.vision import MediapipeVisionProbe


@dataclass(frozen=True, slots=True)
class Scenario:
    scenario_id: str
    signal_name: str
    truth_state: SignalState
    instruction: str
    duration_seconds: float = 7.0
    repetitions: int = 5
    target: tuple[float, float] | None = None


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("present_normal", "away", SignalState.COMPLIANT, "Sit normally in view"),
    Scenario("away_leave_seat", "away", SignalState.VIOLATION, "Leave the camera view"),
    Scenario("away_partial_face_pose", "away", SignalState.COMPLIANT, "Keep upper body visible; partially hide face", repetitions=3),
    Scenario("away_camera_covered", "away", SignalState.UNKNOWN, "Cover the camera", repetitions=3),
    Scenario("phone_none", "phone_held", SignalState.COMPLIANT, "No phone in view"),
    Scenario("phone_on_desk", "phone_held", SignalState.COMPLIANT, "Phone on desk; hands away"),
    Scenario("phone_one_hand", "phone_held", SignalState.VIOLATION, "Hold phone in one hand"),
    Scenario("phone_two_hands", "phone_held", SignalState.VIOLATION, "Use phone with both hands"),
    Scenario("phone_low_gaze", "phone_held", SignalState.VIOLATION, "Hold phone and look down at it"),
    Scenario("phone_out_of_frame", "phone_held", SignalState.UNKNOWN, "Move phone and hands below frame", repetitions=3),
    Scenario("phone_remote", "phone_held", SignalState.COMPLIANT, "Hold a remote or calculator", repetitions=3),
    Scenario("phone_dark_rectangle", "phone_held", SignalState.COMPLIANT, "Hold a dark notebook", repetitions=3),
    Scenario("posture_neutral", "posture_deviation", SignalState.COMPLIANT, "Use normal neutral posture"),
    Scenario("posture_forward", "posture_deviation", SignalState.VIOLATION, "Lean forward and stay there"),
    Scenario("posture_left", "posture_deviation", SignalState.VIOLATION, "Lean left and stay there"),
    Scenario("posture_right", "posture_deviation", SignalState.VIOLATION, "Lean right and stay there"),
    Scenario("posture_head_forward", "posture_deviation", SignalState.VIOLATION, "Move head/shoulders forward"),
    Scenario("posture_shoulders_hidden", "posture_deviation", SignalState.UNKNOWN, "Hide both shoulders", repetitions=3),
    Scenario("attention_center", "attention_away", SignalState.COMPLIANT, "Look at the dot", target=(0.50, 0.50)),
    Scenario("attention_top_left", "attention_away", SignalState.COMPLIANT, "Look at the dot", target=(0.08, 0.08), repetitions=3),
    Scenario("attention_top_right", "attention_away", SignalState.COMPLIANT, "Look at the dot", target=(0.92, 0.08), repetitions=3),
    Scenario("attention_bottom_left", "attention_away", SignalState.COMPLIANT, "Look at the dot", target=(0.08, 0.92), repetitions=3),
    Scenario("attention_bottom_right", "attention_away", SignalState.COMPLIANT, "Look at the dot", target=(0.92, 0.92), repetitions=3),
    Scenario("attention_outside_left", "attention_away", SignalState.VIOLATION, "Look beyond the left screen edge"),
    Scenario("attention_outside_right", "attention_away", SignalState.VIOLATION, "Look beyond the right screen edge"),
    Scenario("attention_desk", "attention_away", SignalState.VIOLATION, "Look down at the desk"),
    Scenario("attention_eyes_covered", "attention_away", SignalState.UNKNOWN, "Cover or close both eyes", repetitions=3),
)


class ValidationCancelled(RuntimeError):
    pass


def scenario_catalog() -> list[dict[str, Any]]:
    return [
        {
            **asdict(scenario),
            "truth_state": scenario.truth_state.value,
        }
        for scenario in SCENARIOS
    ]


def run_live_validation(
    config: AppConfig,
    models: ModelFiles,
    profile_path: str | Path,
    *,
    dataset_role: str,
    signal_names: Sequence[str],
    output_root: str | Path = "data/validation",
    repetitions_override: int | None = None,
    save_error_evidence: bool = True,
) -> Path:
    import cv2

    selected = [scenario for scenario in SCENARIOS if scenario.signal_name in signal_names]
    unknown_signals = set(signal_names) - {scenario.signal_name for scenario in SCENARIOS}
    if unknown_signals:
        raise ValueError(f"unknown validation signals: {sorted(unknown_signals)}")
    if not selected:
        raise ValueError("at least one validation signal must be selected")
    if repetitions_override is not None and repetitions_override < 1:
        raise ValueError("repetitions override must be positive")

    camera = CameraWorker(config.camera)
    probe: MediapipeVisionProbe | None = None
    window = "Observer Stage C validation"
    sampler = ResourceSampler()
    resource_samples = []
    try:
        if not camera.start():
            raise RuntimeError(camera.stats.last_error or "camera could not be opened")
        first = _wait_for_frame(camera)
        environment = current_environment(config, first.frame.shape)
        profile = load_profile(profile_path, environment)
        probe = MediapipeVisionProbe(models, config.inference, profile)
        batch_id = f"{dataset_role}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
        batch_directory = Path(output_root) / batch_id
        config_hash = _config_hash(config)
        manifest_hash = sha256_file(models.face.parent / "manifest.json")
        manifest = new_batch_manifest(
            batch_id=batch_id,
            dataset_role=dataset_role,
            git_commit=_git_commit(),
            config_sha256=config_hash,
            model_manifest_sha256=manifest_hash,
            calibration_profile_id=profile.profile_id,
            environment=asdict(environment),
        )
        recorder = JsonlTrialRecorder(batch_directory, manifest)
        evidence = ErrorEvidenceStore(
            batch_directory / "evidence", EvidencePolicy(enabled=save_error_evidence)
        )
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        for scenario in selected:
            repetitions = repetitions_override or scenario.repetitions
            for repetition in range(1, repetitions + 1):
                _await_ready(camera, window, scenario, repetition, repetitions)
                trial_id = str(uuid4())
                record, error_frame, derived_records, interrupted = _run_trial(
                    camera,
                    probe,
                    window,
                    scenario,
                    dataset_role,
                    trial_id,
                    sampler,
                    resource_samples,
                )
                is_error = record.valid and record.predicted_state != record.truth_state
                evidence_result = evidence.save(
                    record.trial_id, error_frame, is_error=is_error
                )
                if evidence_result is not None:
                    record = replace(
                        record,
                        evidence_relative_path=f"evidence/{evidence_result['relative_path']}",
                        evidence_sha256=evidence_result["sha256"],
                    )
                recorder.append(record)
                _append_jsonl(batch_directory / "derived_samples.jsonl", derived_records)
                if interrupted:
                    raise ValidationCancelled(
                        f"validation cancelled during trial {record.trial_id}; trial saved as invalid"
                    )
        records_path = batch_directory / "trials.jsonl"
        from observer.validation import read_trials

        write_metrics_report(read_trials(records_path), batch_directory / "report.json")
        resource_report = asdict(summarize_resources(resource_samples, config.resources))
        (batch_directory / "resources.json").write_text(
            json.dumps(resource_report, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        return batch_directory
    finally:
        if probe is not None:
            probe.close()
        camera.stop()
        cv2.destroyAllWindows()


def _await_ready(
    camera: CameraWorker,
    window: str,
    scenario: Scenario,
    repetition: int,
    repetitions: int,
) -> None:
    import cv2

    last_sequence = 0
    while True:
        snapshot = camera.buffer.latest(after_sequence=last_sequence)
        if snapshot is None:
            key = cv2.waitKey(10)
        else:
            last_sequence = snapshot.sequence
            canvas = _scenario_canvas(snapshot.frame, scenario)
            _put_lines(
                canvas,
                (
                    f"{scenario.scenario_id} ({repetition}/{repetitions})",
                    scenario.instruction,
                    "SPACE=start   ESC=cancel batch",
                ),
            )
            cv2.imshow(window, canvas)
            key = cv2.waitKey(1)
        if key & 0xFF == 27:
            raise ValidationCancelled("validation cancelled by user")
        if key & 0xFF == 32:
            return


def _run_trial(
    camera: CameraWorker,
    probe: MediapipeVisionProbe,
    window: str,
    scenario: Scenario,
    dataset_role: str,
    trial_id: str,
    sampler: ResourceSampler,
    resource_samples: list[Any],
) -> tuple[TrialRecord, Any, list[dict[str, Any]], bool]:
    import cv2

    started = time.monotonic()
    last_sequence = 0
    states: list[SignalValue] = []
    candidate_started: float | None = None
    last_resource_sample = 0.0
    error_frame = None
    derived_records: list[dict[str, Any]] = []
    previous_runs = (-1, -1, -1)
    interrupted = False
    while time.monotonic() - started < scenario.duration_seconds:
        snapshot = camera.buffer.latest(after_sequence=last_sequence)
        if snapshot is None:
            if cv2.waitKey(1) & 0xFF == 27:
                interrupted = True
                break
            continue
        last_sequence = snapshot.sequence
        probe.process_due(snapshot.frame, snapshot.captured_monotonic)
        signal = probe.observation(snapshot.captured_monotonic).signal(scenario.signal_name)
        states.append(signal)
        runs = (
            probe.measurements.face.runs,
            probe.measurements.pose.runs,
            probe.measurements.object.runs,
        )
        if runs != previous_runs:
            derived = probe.derived_snapshot()
            derived.update(
                {
                    "trial_id": trial_id,
                    "captured_monotonic": snapshot.captured_monotonic,
                    "signal_name": scenario.signal_name,
                    "signal_state": signal.state.value,
                }
            )
            derived_records.append(derived)
            previous_runs = runs
        if signal.state is SignalState.VIOLATION and candidate_started is None:
            candidate_started = snapshot.captured_monotonic
        error_frame = snapshot.frame.copy()
        canvas = _scenario_canvas(snapshot.frame, scenario)
        remaining = max(0.0, scenario.duration_seconds - (time.monotonic() - started))
        _put_lines(canvas, (scenario.instruction, f"Recording {remaining:0.1f}s"))
        cv2.imshow(window, canvas)
        if cv2.waitKey(1) & 0xFF == 27:
            interrupted = True
            break
        now = time.monotonic()
        if now - last_resource_sample >= 1.0:
            resource_samples.append(sampler.sample())
            last_resource_sample = now
    ended = time.monotonic()
    if interrupted:
        record = TrialRecord(
            trial_id=trial_id,
            scenario_id=scenario.scenario_id,
            signal_name=scenario.signal_name,
            dataset_role=dataset_role,
            truth_state=scenario.truth_state.value,
            predicted_state=SignalState.UNKNOWN.value,
            started_monotonic=started,
            ended_monotonic=ended,
            observable_fraction=0.0,
            valid=False,
            invalid_reason="user cancelled during trial",
        )
        return record, error_frame, derived_records, True
    predicted, confidence, metrics, reason, coverage = _aggregate_states(states)
    record = TrialRecord(
        trial_id=trial_id,
        scenario_id=scenario.scenario_id,
        signal_name=scenario.signal_name,
        dataset_role=dataset_role,
        truth_state=scenario.truth_state.value,
        predicted_state=predicted.value,
        started_monotonic=started,
        ended_monotonic=ended,
        confidence=confidence,
        metrics=metrics,
        unknown_reason=reason,
        candidate_started_monotonic=candidate_started,
        observable_fraction=coverage,
    )
    return record, error_frame, derived_records, False


def _aggregate_states(
    states: Sequence[SignalValue],
) -> tuple[SignalState, float | None, dict[str, float], str | None, float]:
    if not states:
        return SignalState.UNKNOWN, None, {}, "no observations", 0.0
    compliant = [state for state in states if state.state is SignalState.COMPLIANT]
    violation = [state for state in states if state.state is SignalState.VIOLATION]
    observable = compliant + violation
    coverage = len(observable) / len(states)
    metrics = {
        "sample_count": float(len(states)),
        "compliant_fraction": len(compliant) / len(states),
        "violation_fraction": len(violation) / len(states),
        "unknown_fraction": 1.0 - coverage,
    }
    metric_names = sorted({name for state in states for name in state.metrics})
    for name in metric_names:
        values = [state.metrics[name] for state in states if name in state.metrics]
        metrics[f"mean_{name}"] = sum(values) / len(values)
    if not observable or coverage < 0.5:
        reasons = [state.reason for state in states if state.reason]
        reason = max(set(reasons), key=reasons.count) if reasons else "insufficient coverage"
        return SignalState.UNKNOWN, None, metrics, reason, coverage
    predicted = SignalState.VIOLATION if len(violation) >= len(compliant) else SignalState.COMPLIANT
    matching = [state.confidence for state in observable if state.state is predicted and state.confidence]
    confidence = sum(matching) / len(matching) if matching else None
    return predicted, confidence, metrics, None, coverage


def _scenario_canvas(frame: Any, scenario: Scenario) -> Any:
    import cv2

    if scenario.target is None:
        return cv2.flip(frame, 1)
    height, width = frame.shape[:2]
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    point = (int(scenario.target[0] * width), int(scenario.target[1] * height))
    cv2.circle(canvas, point, 14, (30, 80, 220), -1)
    return canvas


def _put_lines(frame: Any, lines: Sequence[str]) -> None:
    import cv2

    height = 28 + len(lines) * 30
    cv2.rectangle(frame, (0, 0), (frame.shape[1], height), (20, 20, 20), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (14, 30 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _wait_for_frame(camera: CameraWorker):
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        snapshot = camera.buffer.latest()
        if snapshot is not None:
            return snapshot
        time.sleep(0.01)
    raise RuntimeError("camera opened but did not produce a frame")


def _config_hash(config: AppConfig) -> str:
    encoded = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest().upper()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "-c", "safe.directory=C:/Qproject/Observer", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _append_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            output.write("\n")
