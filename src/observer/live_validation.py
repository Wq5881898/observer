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

from observer.alignment import guide_alignment
from observer.calibration import (
    ATTENTION_HEAD_PITCH_DOWN_DELTA_THRESHOLD,
    ATTENTION_HEAD_YAW_DELTA_THRESHOLD,
    load_profile,
)
from observer.calibration_workflow import current_environment
from observer.camera import CameraWorker
from observer.config import AppConfig
from observer.models import ModelFiles, sha256_file
from observer.observations import SignalState, SignalValue
from observer.resources import ResourceSampler, summarize_resources
from observer.scoring import RuleEngine
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
    accepted_states: tuple[SignalState, ...] = ()


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("present_normal", "away", SignalState.COMPLIANT, "Sit normally in view"),
    Scenario("away_leave_seat", "away", SignalState.VIOLATION, "Leave the camera view"),
    Scenario("away_partial_face_pose", "away", SignalState.COMPLIANT, "Keep upper body visible; partially hide face", repetitions=3),
    Scenario("away_camera_covered", "away", SignalState.UNKNOWN, "Cover the camera", repetitions=3),
    Scenario("phone_natural_work", "phone_held", SignalState.COMPLIANT, "Work normally; no phone in your hands", duration_seconds=6.0, repetitions=1, accepted_states=(SignalState.COMPLIANT, SignalState.UNKNOWN)),
    Scenario("phone_one_hand", "phone_held", SignalState.VIOLATION, "Use your phone naturally with one hand", duration_seconds=6.0, repetitions=1),
    Scenario("phone_two_hands_low_gaze", "phone_held", SignalState.VIOLATION, "Use your phone with both hands and look down naturally", duration_seconds=6.0, repetitions=1),
    Scenario("phone_non_phone_object", "phone_held", SignalState.COMPLIANT, "Hold one non-phone object: remote, calculator, or dark notebook", duration_seconds=6.0, repetitions=1, accepted_states=(SignalState.COMPLIANT, SignalState.UNKNOWN)),
    Scenario("posture_neutral", "posture_deviation", SignalState.COMPLIANT, "Sit in your normal comfortable upright posture", duration_seconds=6.0, repetitions=1),
    Scenario("posture_forward_slouch", "posture_deviation", SignalState.VIOLATION, "Lean or slouch clearly forward and hold", duration_seconds=6.0, repetitions=1),
    Scenario("posture_side_lean", "posture_deviation", SignalState.VIOLATION, "Lean clearly to either side and hold", duration_seconds=6.0, repetitions=1),
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


AWAY_EVENT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("event_normal", "away", SignalState.COMPLIANT, "Sit normally; no alert expected", duration_seconds=8.0, repetitions=1),
    Scenario("event_brief_away", "away", SignalState.VIOLATION, "Leave camera view for 10s; no alert expected", duration_seconds=10.0, repetitions=1),
    Scenario("event_brief_recovery", "away", SignalState.COMPLIANT, "Return and sit normally", duration_seconds=3.0, repetitions=1),
    Scenario("event_sustained_away", "away", SignalState.VIOLATION, "Leave camera view for 25s; alert expected near 20s", duration_seconds=25.0, repetitions=1),
    Scenario("event_sustained_recovery", "away", SignalState.COMPLIANT, "Return and sit normally; recovery expected", duration_seconds=4.0, repetitions=1),
    Scenario("event_face_covered", "away", SignalState.COMPLIANT, "Cover full face with blank paper; keep torso visible", duration_seconds=8.0, repetitions=1),
    Scenario("event_camera_covered", "away", SignalState.UNKNOWN, "Cover the camera completely", duration_seconds=5.0, repetitions=1),
)

BRIEF_AWAY_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("event_normal", "away", SignalState.COMPLIANT, "Sit normally for the baseline", duration_seconds=5.0, repetitions=1),
    Scenario("event_brief_away", "away", SignalState.VIOLATION, "Leave camera view for 10s; no alert expected", duration_seconds=10.0, repetitions=1),
    Scenario("event_brief_recovery", "away", SignalState.COMPLIANT, "Return and sit normally", duration_seconds=4.0, repetitions=1),
)

PHONE_EVENT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("phone_event_normal", "phone_held", SignalState.COMPLIANT, "Work normally without holding the phone", duration_seconds=5.0, repetitions=1),
    Scenario("phone_event_brief", "phone_held", SignalState.VIOLATION, "Use the phone with one hand for 8s; no alert expected", duration_seconds=8.0, repetitions=1),
    Scenario("phone_event_brief_recovery", "phone_held", SignalState.COMPLIANT, "Put the phone down and work normally", duration_seconds=4.0, repetitions=1),
    Scenario("phone_event_sustained", "phone_held", SignalState.VIOLATION, "Use the phone naturally for 15s; alert expected near 10s", duration_seconds=15.0, repetitions=1),
    Scenario("phone_event_sustained_recovery", "phone_held", SignalState.COMPLIANT, "Put the phone down and work normally", duration_seconds=4.0, repetitions=1),
    Scenario("phone_event_non_phone", "phone_held", SignalState.COMPLIANT, "Hold the same non-phone object for 12s", duration_seconds=12.0, repetitions=1),
)

PHONE_EVENT_POSITIVE_SCENARIOS: tuple[Scenario, ...] = tuple(
    scenario
    for scenario in PHONE_EVENT_SCENARIOS
    if scenario.scenario_id not in {"phone_event_normal", "phone_event_non_phone"}
)


class ValidationCancelled(RuntimeError):
    pass


TRIAL_PREPARE_SECONDS = 1.0
TARGET_READY_SECONDS = 0.6
ATTENTION_DIRECTION_SCREEN_MARGIN = 0.08


def scenario_catalog() -> list[dict[str, Any]]:
    return [
        {
            **asdict(scenario),
            "truth_state": scenario.truth_state.value,
            "accepted_states": [state.value for state in scenario_accepted_states(scenario)],
        }
        for scenario in SCENARIOS
    ]


def scenario_accepted_states(scenario: Scenario) -> tuple[SignalState, ...]:
    return scenario.accepted_states or (scenario.truth_state,)


_DIRECTIONAL_ATTENTION_SCENARIOS = {
    "attention_outside_left",
    "attention_outside_right",
    "attention_desk",
}


def attention_direction_matches(scenario: Scenario, signal: SignalValue) -> bool:
    """Require the requested direction, not merely any attention violation."""

    metrics = signal.metrics
    yaw = metrics.get("attention_head_yaw_delta")
    pitch = metrics.get("attention_head_pitch_delta")
    screen_x = metrics.get("attention_screen_x")
    screen_y = metrics.get("attention_screen_y")
    if scenario.scenario_id == "attention_outside_left":
        return (yaw is not None and yaw >= ATTENTION_HEAD_YAW_DELTA_THRESHOLD) or (
            screen_x is not None
            and screen_x < -ATTENTION_DIRECTION_SCREEN_MARGIN
        )
    if scenario.scenario_id == "attention_outside_right":
        return (yaw is not None and yaw <= -ATTENTION_HEAD_YAW_DELTA_THRESHOLD) or (
            screen_x is not None
            and screen_x > 1.0 + ATTENTION_DIRECTION_SCREEN_MARGIN
        )
    if scenario.scenario_id == "attention_desk":
        return (
            pitch is not None
            and pitch >= ATTENTION_HEAD_PITCH_DOWN_DELTA_THRESHOLD
        ) or (
            screen_y is not None and screen_y > 1.0
        )
    return signal.state in scenario_accepted_states(scenario)


def scenario_validation_signal(scenario: Scenario, signal: SignalValue) -> SignalValue:
    """Convert the binary production signal into a direction-specific test signal."""

    if scenario.scenario_id not in _DIRECTIONAL_ATTENTION_SCENARIOS:
        return signal
    if signal.state is SignalState.UNKNOWN:
        return signal
    if attention_direction_matches(scenario, signal):
        return SignalValue.violation(signal.confidence, **signal.metrics)
    return SignalValue.compliant(signal.confidence, **signal.metrics)


def attention_direction_status(signal: SignalValue) -> str:
    metrics = signal.metrics
    values = []
    for label, name in (
        ("yaw", "attention_head_yaw_delta"),
        ("pitch", "attention_head_pitch_delta"),
        ("x", "attention_screen_x"),
        ("y", "attention_screen_y"),
    ):
        if name in metrics:
            values.append(f"{label}={metrics[name]:+.2f}")
    return "  ".join(values) or "direction evidence unavailable"


def run_live_validation(
    config: AppConfig,
    models: ModelFiles,
    profile_path: str | Path,
    *,
    dataset_role: str,
    signal_names: Sequence[str],
    scenario_ids: Sequence[str] = (),
    output_root: str | Path = "data/validation",
    repetitions_override: int | None = None,
    save_error_evidence: bool = True,
) -> Path:
    import cv2

    requested_scenarios = set(scenario_ids)
    known_scenarios = {scenario.scenario_id for scenario in SCENARIOS}
    unknown_scenarios = requested_scenarios - known_scenarios
    if unknown_scenarios:
        raise ValueError(f"unknown validation scenarios: {sorted(unknown_scenarios)}")
    selected = [
        scenario
        for scenario in SCENARIOS
        if scenario.signal_name in signal_names
        and (not requested_scenarios or scenario.scenario_id in requested_scenarios)
    ]
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
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        if set(signal_names) == {"phone_held"}:
            alignment_summary: dict[str, Any] = {
                "skipped": True,
                "reason": "phone-only validation does not depend on attention alignment",
                "frames_saved": 0,
            }
        else:
            alignment_summary = asdict(
                guide_alignment(
                    camera,
                    probe,
                    profile,
                    window,
                    require_posture="posture_deviation" in signal_names,
                )
            )
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
            alignment=alignment_summary,
        )
        recorder = JsonlTrialRecorder(batch_directory, manifest)
        evidence = ErrorEvidenceStore(
            batch_directory / "evidence", EvidencePolicy(enabled=save_error_evidence)
        )
        for scenario in selected:
            repetitions = repetitions_override or scenario.repetitions
            for repetition in range(1, repetitions + 1):
                _await_ready(camera, probe, window, scenario, repetition, repetitions)
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
                accepted = {state.value for state in scenario_accepted_states(scenario)}
                is_error = record.valid and record.predicted_state not in accepted
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

        records = read_trials(records_path)
        report = write_metrics_report(records, batch_directory / "report.json")
        scenario_by_id = {scenario.scenario_id: scenario for scenario in selected}
        scenario_checks = [
            {
                "trial_id": record.trial_id,
                "scenario_id": record.scenario_id,
                "predicted_state": record.predicted_state,
                "accepted_states": [
                    state.value
                    for state in scenario_accepted_states(scenario_by_id[record.scenario_id])
                ],
                "passed": record.valid
                and record.predicted_state
                in {
                    state.value
                    for state in scenario_accepted_states(scenario_by_id[record.scenario_id])
                },
            }
            for record in records
        ]
        report["scenario_checks"] = scenario_checks
        report["passed"] = all(check["passed"] for check in scenario_checks)
        (batch_directory / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
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


def run_away_event_validation(
    config: AppConfig,
    models: ModelFiles,
    profile_path: str | Path,
    *,
    dataset_role: str,
    output_root: str | Path = "data/validation",
    suite: str = "full",
) -> Path:
    """Exercise the configured away rule against one continuous live session."""

    import cv2

    if suite not in {"full", "brief"}:
        raise ValueError("away event suite must be 'full' or 'brief'")
    scenarios = AWAY_EVENT_SCENARIOS if suite == "full" else BRIEF_AWAY_SCENARIOS

    away_configs = tuple(rule for rule in config.rules if rule.rule_id == "away")
    if len(away_configs) != 1 or away_configs[0].signal_name != "away":
        raise ValueError("event validation requires exactly one configured away rule")
    away_config = away_configs[0]
    if away_config.trigger_after_seconds >= 25.0:
        raise ValueError("sustained event scenario must outlast the configured away trigger")

    camera = CameraWorker(config.camera)
    probe: MediapipeVisionProbe | None = None
    window = "Observer Stage C away event validation"
    sampler = ResourceSampler()
    resource_samples: list[Any] = []
    try:
        if not camera.start():
            raise RuntimeError(camera.stats.last_error or "camera could not be opened")
        first = _wait_for_frame(camera)
        environment = current_environment(config, first.frame.shape)
        profile = load_profile(profile_path, environment)
        probe = MediapipeVisionProbe(models, config.inference, profile)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        alignment = guide_alignment(camera, probe, profile, window)

        batch_id = (
            f"event-{suite}-{dataset_role}-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
            f"{uuid4().hex[:8]}"
        )
        batch_directory = Path(output_root) / batch_id
        batch_directory.mkdir(parents=True, exist_ok=False)
        manifest = new_batch_manifest(
            batch_id=batch_id,
            dataset_role=dataset_role,
            git_commit=_git_commit(),
            config_sha256=_config_hash(config),
            model_manifest_sha256=sha256_file(models.face.parent / "manifest.json"),
            calibration_profile_id=profile.profile_id,
            environment=asdict(environment),
            alignment=asdict(alignment),
        )
        (batch_directory / "manifest.json").write_text(
            json.dumps(asdict(manifest), indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

        engine = RuleEngine((away_config,))
        segments: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        samples_path = batch_directory / "event_samples.jsonl"
        for scenario in scenarios:
            _await_ready(camera, probe, window, scenario, 1, 1)
            segment, segment_actions, derived_records, interrupted = _run_event_segment(
                camera,
                probe,
                engine,
                window,
                scenario,
                sampler,
                resource_samples,
            )
            segments.append(segment)
            actions.extend(segment_actions)
            _append_jsonl(samples_path, derived_records)
            if interrupted:
                raise ValidationCancelled(
                    f"event validation cancelled during {scenario.scenario_id}"
                )

        if suite == "full":
            report = evaluate_away_event_report(
                segments,
                actions,
                trigger_after_seconds=away_config.trigger_after_seconds,
                recovery_after_seconds=away_config.recovery_after_seconds,
                expected_penalty_points=away_config.initial_penalty_points,
                final_score=engine.ledger.score,
            )
        else:
            report = evaluate_brief_away_report(
                segments,
                actions,
                final_score=engine.ledger.score,
            )
        report["suite"] = suite
        (batch_directory / "event_segments.json").write_text(
            json.dumps(segments, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        (batch_directory / "event_actions.json").write_text(
            json.dumps(actions, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        (batch_directory / "event_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        resources = asdict(summarize_resources(resource_samples, config.resources))
        (batch_directory / "resources.json").write_text(
            json.dumps(resources, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        return batch_directory
    finally:
        if probe is not None:
            probe.close()
        camera.stop()
        cv2.destroyAllWindows()


def run_phone_event_validation(
    config: AppConfig,
    models: ModelFiles,
    profile_path: str | Path,
    *,
    dataset_role: str,
    output_root: str | Path = "data/validation",
    suite: str = "full",
) -> Path:
    """Exercise the configured phone rule in one continuous live session."""

    import cv2

    if suite not in {"full", "positive"}:
        raise ValueError("phone event suite must be 'full' or 'positive'")
    scenarios = (
        PHONE_EVENT_SCENARIOS if suite == "full" else PHONE_EVENT_POSITIVE_SCENARIOS
    )

    phone_configs = tuple(rule for rule in config.rules if rule.rule_id == "phone_held")
    if len(phone_configs) != 1 or phone_configs[0].signal_name != "phone_held":
        raise ValueError("event validation requires exactly one configured phone rule")
    phone_config = phone_configs[0]
    if phone_config.trigger_after_seconds >= 15.0:
        raise ValueError("sustained phone scenario must outlast the configured trigger")

    camera = CameraWorker(config.camera)
    probe: MediapipeVisionProbe | None = None
    window = "Observer Stage C phone event validation"
    sampler = ResourceSampler()
    resource_samples: list[Any] = []
    try:
        if not camera.start():
            raise RuntimeError(camera.stats.last_error or "camera could not be opened")
        first = _wait_for_frame(camera)
        environment = current_environment(config, first.frame.shape)
        profile = load_profile(profile_path, environment)
        probe = MediapipeVisionProbe(models, config.inference, profile)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        batch_id = (
            f"event-phone-{suite}-{dataset_role}-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
            f"{uuid4().hex[:8]}"
        )
        batch_directory = Path(output_root) / batch_id
        batch_directory.mkdir(parents=True, exist_ok=False)
        manifest = new_batch_manifest(
            batch_id=batch_id,
            dataset_role=dataset_role,
            git_commit=_git_commit(),
            config_sha256=_config_hash(config),
            model_manifest_sha256=sha256_file(models.face.parent / "manifest.json"),
            calibration_profile_id=profile.profile_id,
            environment=asdict(environment),
            alignment={
                "skipped": True,
                "reason": "phone event validation does not depend on attention alignment",
                "frames_saved": 0,
            },
        )
        (batch_directory / "manifest.json").write_text(
            json.dumps(asdict(manifest), indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

        engine = RuleEngine((phone_config,))
        segments: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        samples_path = batch_directory / "event_samples.jsonl"
        for scenario in scenarios:
            _await_ready(camera, probe, window, scenario, 1, 1)
            segment, segment_actions, derived_records, interrupted = _run_event_segment(
                camera,
                probe,
                engine,
                window,
                scenario,
                sampler,
                resource_samples,
            )
            segments.append(segment)
            actions.extend(segment_actions)
            _append_jsonl(samples_path, derived_records)
            if interrupted:
                raise ValidationCancelled(
                    f"event validation cancelled during {scenario.scenario_id}"
                )

        report = evaluate_phone_event_report(
            segments,
            actions,
            trigger_after_seconds=phone_config.trigger_after_seconds,
            recovery_after_seconds=phone_config.recovery_after_seconds,
            expected_penalty_points=phone_config.initial_penalty_points,
            final_score=engine.ledger.score,
            unknown_grace_seconds=phone_config.unknown_grace_seconds,
        )
        report["suite"] = suite
        for filename, payload in (
            ("event_segments.json", segments),
            ("event_actions.json", actions),
            ("event_report.json", report),
        ):
            (batch_directory / filename).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
        resources = asdict(summarize_resources(resource_samples, config.resources))
        (batch_directory / "resources.json").write_text(
            json.dumps(resources, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        return batch_directory
    finally:
        if probe is not None:
            probe.close()
        camera.stop()
        cv2.destroyAllWindows()


def _run_event_segment(
    camera: CameraWorker,
    probe: MediapipeVisionProbe,
    engine: RuleEngine,
    window: str,
    scenario: Scenario,
    sampler: ResourceSampler,
    resource_samples: list[Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], bool]:
    import cv2

    started = time.monotonic()
    last_sequence = 0
    last_resource_sample = 0.0
    state_counts = {state.value: 0 for state in SignalState}
    longest_state_run_seconds = {state.value: 0.0 for state in SignalState}
    current_state: SignalState | None = None
    current_state_started: float | None = None
    current_state_last: float | None = None
    segment_actions: list[dict[str, Any]] = []
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
        observation = probe.observation(snapshot.captured_monotonic)
        signal = observation.signal(scenario.signal_name)
        state_counts[signal.state.value] += 1
        if signal.state is not current_state:
            if current_state is not None and current_state_started is not None:
                run_seconds = (current_state_last or current_state_started) - current_state_started
                longest_state_run_seconds[current_state.value] = max(
                    longest_state_run_seconds[current_state.value], run_seconds
                )
            current_state = signal.state
            current_state_started = snapshot.captured_monotonic
        current_state_last = snapshot.captured_monotonic
        result = engine.process(observation)
        for action in result.actions:
            action_payload = asdict(action)
            action_payload["kind"] = action.kind.value
            action_payload["scenario_id"] = scenario.scenario_id
            segment_actions.append(action_payload)

        runs = (
            probe.measurements.face.runs,
            probe.measurements.pose.runs,
            probe.measurements.object.runs,
        )
        if runs != previous_runs:
            derived = probe.derived_snapshot()
            derived.update(
                {
                    "scenario_id": scenario.scenario_id,
                    "captured_monotonic": snapshot.captured_monotonic,
                    "signal_state": signal.state.value,
                    "rule_actions": [
                        action.kind.value for action in result.actions
                    ],
                    "score": result.score,
                }
            )
            derived_records.append(derived)
            previous_runs = runs

        elapsed = time.monotonic() - started
        canvas = _scenario_canvas(snapshot.frame, scenario)
        recent = ", ".join(action.kind.value for action in result.actions) or "none"
        _put_lines(
            canvas,
            (
                scenario.instruction,
                f"Recording {max(0.0, scenario.duration_seconds - elapsed):0.1f}s",
                f"{scenario.signal_name}={signal.state.value} actions={recent} score={result.score:0.1f}",
            ),
        )
        cv2.imshow(window, canvas)
        if cv2.waitKey(1) & 0xFF == 27:
            interrupted = True
            break
        now = time.monotonic()
        if now - last_resource_sample >= 1.0:
            resource_samples.append(sampler.sample())
            last_resource_sample = now

    ended = time.monotonic()
    if current_state is not None and current_state_started is not None:
        run_seconds = (current_state_last or current_state_started) - current_state_started
        longest_state_run_seconds[current_state.value] = max(
            longest_state_run_seconds[current_state.value], run_seconds
        )
    return (
        {
            "scenario_id": scenario.scenario_id,
            "instruction": scenario.instruction,
            "expected_state": scenario.truth_state.value,
            "started_monotonic": started,
            "ended_monotonic": ended,
            "duration_seconds": ended - started,
            "state_counts": state_counts,
            "longest_state_run_seconds": longest_state_run_seconds,
            "action_kinds": [action["kind"] for action in segment_actions],
            "interrupted": interrupted,
        },
        segment_actions,
        derived_records,
        interrupted,
    )


def evaluate_away_event_report(
    segments: Sequence[dict[str, Any]],
    actions: Sequence[dict[str, Any]],
    *,
    trigger_after_seconds: float,
    recovery_after_seconds: float,
    expected_penalty_points: float,
    final_score: float,
) -> dict[str, Any]:
    """Evaluate event semantics without hiding temporal failures behind trial F1."""

    by_scenario = {segment["scenario_id"]: segment for segment in segments}
    actions_by_scenario = {
        scenario_id: [
            action for action in actions if action["scenario_id"] == scenario_id
        ]
        for scenario_id in by_scenario
    }

    def kinds(scenario_id: str) -> list[str]:
        return [action["kind"] for action in actions_by_scenario[scenario_id]]

    def state_ratio(scenario_id: str, state: str) -> float:
        counts = by_scenario[scenario_id]["state_counts"]
        total = sum(counts.values())
        return counts.get(state, 0) / total if total else 0.0

    sustained_actions = actions_by_scenario["event_sustained_away"]
    activation = next(
        (action for action in sustained_actions if action["kind"] == "activated"),
        None,
    )
    activation_delay = (
        activation["at_monotonic"] - activation["event_started_monotonic"]
        if activation is not None
        else None
    )
    recovery_actions = actions_by_scenario["event_sustained_recovery"]
    recovery = next(
        (
            action
            for action in recovery_actions
            if action["kind"] == "resolved" and action.get("reason") == "recovered"
        ),
        None,
    )
    recovery_delay = (
        recovery["at_monotonic"]
        - by_scenario["event_sustained_recovery"]["started_monotonic"]
        if recovery is not None
        else None
    )
    face_counts = by_scenario["event_face_covered"]["state_counts"]
    face_total = sum(face_counts.values())
    camera_counts = by_scenario["event_camera_covered"]["state_counts"]
    camera_total = sum(camera_counts.values())
    penalty_points = sum(
        action.get("requested_points", 0.0)
        for action in actions
        if action["kind"] == "penalty"
    )

    checks = {
        "normal_input_valid": state_ratio("event_normal", "compliant") >= 0.8,
        "normal_no_activation": not any(
            kind in {"activated", "penalty"} for kind in kinds("event_normal")
        ),
        "brief_away_input_valid": state_ratio("event_brief_away", "violation")
        >= 0.7,
        "brief_away_no_activation": not any(
            kind in {"activated", "penalty"}
            for kind in kinds("event_brief_away")
        ),
        "sustained_away_activated_once": kinds("event_sustained_away").count(
            "activated"
        )
        == 1,
        "sustained_away_alerted_once": kinds("event_sustained_away").count("alert")
        == 1,
        "sustained_away_penalized_once": kinds("event_sustained_away").count(
            "penalty"
        )
        == 1,
        "sustained_away_input_valid": state_ratio(
            "event_sustained_away", "violation"
        )
        >= 0.8,
        "activation_timing_within_1_5s": activation_delay is not None
        and abs(activation_delay - trigger_after_seconds) <= 1.5,
        "recovery_resolved": recovery is not None,
        "recovery_input_valid": state_ratio(
            "event_sustained_recovery", "compliant"
        )
        >= 0.8,
        "recovery_timing_within_1_5s": recovery_delay is not None
        and abs(recovery_delay - recovery_after_seconds) <= 1.5,
        "face_covered_no_activation": not any(
            kind in {"activated", "penalty"}
            for kind in kinds("event_face_covered")
        ),
        "face_covered_majority_present": face_total > 0
        and face_counts.get("compliant", 0) / face_total >= 0.5,
        "face_covered_input_valid": state_ratio("event_face_covered", "compliant")
        >= 0.8,
        "camera_covered_no_activation": not any(
            kind in {"activated", "penalty"}
            for kind in kinds("event_camera_covered")
        ),
        "camera_covered_majority_unknown": camera_total > 0
        and camera_counts.get("unknown", 0) / camera_total >= 0.5,
        "camera_covered_input_valid": state_ratio("event_camera_covered", "unknown")
        >= 0.8,
        "penalty_points_match": abs(penalty_points - expected_penalty_points) < 1e-6,
        "final_score_match": abs(final_score - (100.0 - expected_penalty_points))
        < 1e-6,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "trigger_after_seconds": trigger_after_seconds,
        "recovery_after_seconds": recovery_after_seconds,
        "activation_delay_seconds": activation_delay,
        "recovery_delay_seconds": recovery_delay,
        "state_ratios": {
            scenario_id: {
                state: state_ratio(scenario_id, state)
                for state in ("compliant", "violation", "unknown", "exempt")
            }
            for scenario_id in by_scenario
        },
        "penalty_points": penalty_points,
        "final_score": final_score,
    }


def evaluate_brief_away_report(
    segments: Sequence[dict[str, Any]],
    actions: Sequence[dict[str, Any]],
    *,
    final_score: float,
) -> dict[str, Any]:
    """Evaluate the focused below-threshold away and reset sequence."""

    by_scenario = {segment["scenario_id"]: segment for segment in segments}

    def state_ratio(scenario_id: str, state: str) -> float:
        counts = by_scenario[scenario_id]["state_counts"]
        total = sum(counts.values())
        return counts.get(state, 0) / total if total else 0.0

    forbidden_actions = [
        action
        for action in actions
        if action["kind"] in {"activated", "alert", "penalty"}
    ]
    ratios = {
        scenario_id: {
            state: state_ratio(scenario_id, state)
            for state in ("compliant", "violation", "unknown", "exempt")
        }
        for scenario_id in by_scenario
    }
    checks = {
        "normal_input_valid": ratios["event_normal"]["compliant"] >= 0.8,
        "brief_away_input_valid": ratios["event_brief_away"]["violation"] >= 0.8,
        "recovery_input_valid": ratios["event_brief_recovery"]["compliant"] >= 0.8,
        "no_activation_alert_or_penalty": not forbidden_actions,
        "final_score_unchanged": abs(final_score - 100.0) < 1e-6,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "state_ratios": ratios,
        "forbidden_actions": forbidden_actions,
        "final_score": final_score,
    }


def evaluate_phone_event_report(
    segments: Sequence[dict[str, Any]],
    actions: Sequence[dict[str, Any]],
    *,
    trigger_after_seconds: float,
    recovery_after_seconds: float,
    expected_penalty_points: float,
    final_score: float,
    unknown_grace_seconds: float = 3.0,
) -> dict[str, Any]:
    """Evaluate brief use, sustained penalty, recovery, and non-phone safety."""

    by_scenario = {segment["scenario_id"]: segment for segment in segments}

    def ratio(scenario_id: str, state: str) -> float:
        counts = by_scenario[scenario_id]["state_counts"]
        total = sum(counts.values())
        return counts.get(state, 0) / total if total else 0.0

    def non_violation_ratio(scenario_id: str) -> float:
        return ratio(scenario_id, "compliant") + ratio(scenario_id, "unknown")

    def scenario_actions(scenario_id: str) -> list[dict[str, Any]]:
        return [action for action in actions if action["scenario_id"] == scenario_id]

    def kinds(scenario_id: str) -> list[str]:
        return [action["kind"] for action in scenario_actions(scenario_id)]

    sustained_actions = scenario_actions("phone_event_sustained")
    activation = next(
        (action for action in sustained_actions if action["kind"] == "activated"),
        None,
    )
    activation_delay = (
        activation["at_monotonic"] - activation["event_started_monotonic"]
        if activation is not None
        else None
    )
    recovery = next(
        (
            action
            for action in scenario_actions("phone_event_sustained_recovery")
            if action["kind"] == "resolved"
            and action.get("reason") in {"recovered", "observation_lost"}
        ),
        None,
    )
    recovery_delay = (
        recovery["at_monotonic"]
        - by_scenario["phone_event_sustained_recovery"]["started_monotonic"]
        if recovery is not None
        else None
    )
    expected_recovery_delay = (
        recovery_after_seconds
        if recovery is not None and recovery.get("reason") == "recovered"
        else unknown_grace_seconds
    )
    penalty_points = sum(
        action.get("requested_points", 0.0)
        for action in actions
        if action["kind"] == "penalty"
    )
    forbidden = {"activated", "alert", "penalty"}
    checks = {
        "brief_phone_input_valid": by_scenario["phone_event_brief"]
        .get("longest_state_run_seconds", {})
        .get("violation", 0.0)
        >= 3.0,
        "brief_phone_no_activation": not forbidden.intersection(kinds("phone_event_brief")),
        "brief_recovery_input_valid": non_violation_ratio("phone_event_brief_recovery") >= 0.8,
        "sustained_phone_input_valid": ratio("phone_event_sustained", "violation") >= 0.8,
        "sustained_activated_once": kinds("phone_event_sustained").count("activated") == 1,
        "sustained_alerted_once": kinds("phone_event_sustained").count("alert") == 1,
        "sustained_penalized_once": kinds("phone_event_sustained").count("penalty") == 1,
        "activation_timing_within_1_5s": activation_delay is not None
        and abs(activation_delay - trigger_after_seconds) <= 1.5,
        "sustained_recovery_input_valid": non_violation_ratio("phone_event_sustained_recovery") >= 0.8,
        "recovery_resolved": recovery is not None,
        "recovery_timing_within_1_5s": recovery_delay is not None
        and abs(recovery_delay - expected_recovery_delay) <= 1.5,
        "penalty_points_match": abs(penalty_points - expected_penalty_points) < 1e-6,
        "final_score_match": abs(final_score - (100.0 - expected_penalty_points)) < 1e-6,
    }
    if "phone_event_normal" in by_scenario:
        checks.update(
            {
                "normal_input_valid": non_violation_ratio("phone_event_normal") >= 0.8,
                "normal_no_activation": not forbidden.intersection(
                    kinds("phone_event_normal")
                ),
            }
        )
    if "phone_event_non_phone" in by_scenario:
        checks.update(
            {
                "non_phone_input_valid": non_violation_ratio("phone_event_non_phone")
                >= 0.8,
                "non_phone_no_activation": not forbidden.intersection(
                    kinds("phone_event_non_phone")
                ),
            }
        )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "trigger_after_seconds": trigger_after_seconds,
        "recovery_after_seconds": recovery_after_seconds,
        "activation_delay_seconds": activation_delay,
        "recovery_delay_seconds": recovery_delay,
        "recovery_reason": recovery.get("reason") if recovery is not None else None,
        "expected_recovery_delay_seconds": expected_recovery_delay,
        "penalty_points": penalty_points,
        "final_score": final_score,
        "state_ratios": {
            scenario_id: {
                state: ratio(scenario_id, state)
                for state in ("compliant", "violation", "unknown", "exempt")
            }
            for scenario_id in by_scenario
        },
    }


def _await_ready(
    camera: CameraWorker,
    probe: MediapipeVisionProbe,
    window: str,
    scenario: Scenario,
    repetition: int,
    repetitions: int,
) -> None:
    import cv2

    last_sequence = 0
    target_since: float | None = None
    while True:
        snapshot = camera.buffer.latest(after_sequence=last_sequence)
        if snapshot is None:
            key = cv2.waitKey(10)
        else:
            last_sequence = snapshot.sequence
            probe.process_due(snapshot.frame, snapshot.captured_monotonic)
            raw_signal = probe.observation(snapshot.captured_monotonic).signal(
                scenario.signal_name
            )
            signal = scenario_validation_signal(scenario, raw_signal)
            current_state = signal.state
            target_matches = current_state in scenario_accepted_states(scenario)
            if target_matches:
                if target_since is None:
                    target_since = snapshot.captured_monotonic
            else:
                target_since = None
            ready_seconds = (
                max(0.0, snapshot.captured_monotonic - target_since)
                if target_since is not None
                else 0.0
            )
            target_ready = target_matches and ready_seconds >= TARGET_READY_SECONDS
            accepted = "/".join(
                state.value for state in scenario_accepted_states(scenario)
            )
            canvas = _scenario_canvas(snapshot.frame, scenario)
            _put_lines(
                canvas,
                (
                    f"{scenario.scenario_id} ({repetition}/{repetitions})",
                    scenario.instruction,
                    f"TARGET: {accepted}   CURRENT: {current_state.value}",
                    attention_direction_status(raw_signal)
                    if scenario.scenario_id in _DIRECTIONAL_ATTENTION_SCENARIOS
                    else "",
                    (
                        "TARGET READY - press SPACE to sample"
                        if target_ready
                        else "NOT READY - adjust/hold; SPACE still records this attempt"
                    ),
                    "SPACE=begin 1s preparation   ESC=cancel batch",
                ),
            )
            _draw_target_badge(canvas, target_ready)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(1)
        if key & 0xFF == 27:
            raise ValidationCancelled("validation cancelled by user")
        if key & 0xFF == 32:
            _run_prepare_countdown(
                camera, probe, window, scenario, TRIAL_PREPARE_SECONDS
            )
            return


def _run_prepare_countdown(
    camera: CameraWorker,
    probe: MediapipeVisionProbe,
    window: str,
    scenario: Scenario,
    duration_seconds: float,
) -> None:
    import cv2

    started = time.monotonic()
    last_sequence = 0
    while time.monotonic() - started < duration_seconds:
        snapshot = camera.buffer.latest(after_sequence=last_sequence)
        if snapshot is None:
            key = cv2.waitKey(10)
        else:
            last_sequence = snapshot.sequence
            probe.process_due(snapshot.frame, snapshot.captured_monotonic)
            raw_signal = probe.observation(snapshot.captured_monotonic).signal(
                scenario.signal_name
            )
            signal = scenario_validation_signal(scenario, raw_signal)
            current_state = signal.state
            target_matches = current_state in scenario_accepted_states(scenario)
            canvas = _scenario_canvas(snapshot.frame, scenario)
            remaining = max(0.0, duration_seconds - (time.monotonic() - started))
            _put_lines(
                canvas,
                (
                    scenario.instruction,
                    f"PREPARE NOW - recording starts in {remaining:0.1f}s",
                    f"CURRENT: {current_state.value}   "
                    f"TARGET: {'MATCHED' if target_matches else 'NOT MATCHED'}",
                    attention_direction_status(raw_signal)
                    if scenario.scenario_id in _DIRECTIONAL_ATTENTION_SCENARIOS
                    else "",
                    f"Complete the action, then hold it for {scenario.duration_seconds:g}s   ESC=cancel",
                ),
            )
            cv2.imshow(window, canvas)
            key = cv2.waitKey(1)
        if key & 0xFF == 27:
            raise ValidationCancelled("validation cancelled during preparation")


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
        raw_signal = probe.observation(snapshot.captured_monotonic).signal(
            scenario.signal_name
        )
        signal = scenario_validation_signal(scenario, raw_signal)
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
                    "raw_signal_state": raw_signal.state.value,
                }
            )
            derived_records.append(derived)
            previous_runs = runs
        if signal.state is SignalState.VIOLATION and candidate_started is None:
            candidate_started = snapshot.captured_monotonic
        error_frame = snapshot.frame.copy()
        canvas = _scenario_canvas(snapshot.frame, scenario)
        remaining = max(0.0, scenario.duration_seconds - (time.monotonic() - started))
        target_matches = signal.state in scenario_accepted_states(scenario)
        _put_lines(
            canvas,
            (
                scenario.instruction,
                f"Sampling {remaining:0.1f}s - no video saved",
                f"CURRENT: {signal.state.value}   "
                f"TARGET: {'MAINTAINED' if target_matches else 'LOST - still recorded'}",
                attention_direction_status(raw_signal)
                if scenario.scenario_id in _DIRECTIONAL_ATTENTION_SCENARIOS
                else "",
            ),
        )
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


def _draw_target_badge(frame: Any, ready: bool) -> None:
    """Draw an obvious readiness mark without depending on font glyph support."""

    import cv2

    height, width = frame.shape[:2]
    color = (55, 200, 55) if ready else (60, 170, 230)
    label = "TARGET READY" if ready else "TARGET NOT READY"
    left = max(12, width - 250)
    top = max(12, height - 76)
    cv2.rectangle(frame, (left, top), (width - 12, height - 12), (20, 20, 20), -1)
    cv2.rectangle(frame, (left, top), (width - 12, height - 12), color, 2)
    if ready:
        cv2.line(frame, (left + 14, top + 34), (left + 27, top + 48), color, 5)
        cv2.line(frame, (left + 27, top + 48), (left + 50, top + 18), color, 5)
    else:
        cv2.line(frame, (left + 17, top + 20), (left + 46, top + 49), color, 4)
        cv2.line(frame, (left + 46, top + 20), (left + 17, top + 49), color, 4)
    cv2.putText(
        frame,
        label,
        (left + 60, top + 41),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        color,
        2,
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
