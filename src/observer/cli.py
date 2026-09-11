from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

from observer.camera import CameraWorker
from observer.config import load_config
from observer.models import sha256_file, verify_model_directory
from observer.object_detection import detector_runtime_report, is_phone_class_name
from observer.resources import ResourceSampler, percentile, summarize_resources


GUIDED_DEBUG_CUES = (
    (0.0, 180.0, "natural", "Work naturally. No staged action."),
    (180.0, 205.0, "near_lower", "Move closer. Look at SCREEN BOTTOM. NO PHONE."),
    (205.0, 220.0, "normal_1", "NORMAL. Look at the screen. NO PHONE."),
    (220.0, 232.0, "outside_left", "Look LEFT beyond the screen. NO PHONE."),
    (232.0, 245.0, "normal_2", "NORMAL. Look at the screen. NO PHONE."),
    (245.0, 260.0, "phone", "Hold and use PHONE visibly in frame."),
    (260.0, 300.0, "natural_end", "Put PHONE DOWN. Work naturally."),
)

ATTENTION_MINUTE_CUES = (
    (0.0, 10.0, "normal_start", "NORMAL. Look at the screen center."),
    (10.0, 25.0, "outside_left_down", "Look LEFT-DOWN beyond the screen."),
    (25.0, 35.0, "normal_middle", "NORMAL. Look at the screen center."),
    (35.0, 50.0, "near_lower_inside", "Move closer. Look INSIDE screen lower-left."),
    (50.0, 60.0, "normal_end", "NORMAL. Look at the screen center."),
)

ATTENTION_BOUNDARY_CUES = (
    (0.0, 3.0, "prep_a01", "PREP A01: natural distance, screen center."),
    (3.0, 13.0, "score_a01", "A01 HOLD: natural distance, screen center."),
    (13.0, 16.0, "prep_a02", "PREP A02: look at fixed outside LEFT-LOW point."),
    (16.0, 28.0, "score_a02", "A02 HOLD: fixed outside LEFT-LOW point."),
    (28.0, 31.0, "prep_a03", "PREP A03: return to screen center."),
    (31.0, 39.0, "score_a03", "A03 HOLD: screen center recovery."),
    (39.0, 42.0, "prep_a04", "PREP A04: look at fixed outside RIGHT-LOW point."),
    (42.0, 54.0, "score_a04", "A04 HOLD: fixed outside RIGHT-LOW point."),
    (54.0, 57.0, "prep_a05", "PREP A05: return to screen center."),
    (57.0, 65.0, "score_a05", "A05 HOLD: screen center recovery."),
    (65.0, 68.0, "prep_a06", "PREP A06: move closer; fixed INSIDE left-low."),
    (68.0, 80.0, "score_a06", "A06 HOLD: closer, fixed INSIDE left-low."),
    (80.0, 83.0, "prep_a07", "PREP A07: stay closer; fixed INSIDE right-low."),
    (83.0, 95.0, "score_a07", "A07 HOLD: closer, fixed INSIDE right-low."),
    (95.0, 98.0, "prep_a08", "PREP A08: move FAR; look at screen center."),
    (98.0, 108.0, "score_a08", "A08 HOLD: far distance, screen center."),
    (108.0, 111.0, "prep_a09", "PREP A09: far; fixed INSIDE left-low."),
    (111.0, 123.0, "score_a09", "A09 HOLD: far, fixed INSIDE left-low."),
    (123.0, 126.0, "prep_a10", "PREP A10: far; fixed INSIDE right-low."),
    (126.0, 138.0, "score_a10", "A10 HOLD: far, fixed INSIDE right-low."),
    (138.0, 141.0, "prep_a11", "PREP A11: natural distance, screen center."),
    (141.0, 151.0, "score_a11", "A11 HOLD: final screen center."),
)

C_CLOSEOUT_HOLDOUT_CUES = (
    (0.0, 10.0, "h01_neutral", "H01: Work normally. Face visible. NO PHONE."),
    (10.0, 15.0, "transition_away", "Prepare H02: leave the camera view."),
    (
        15.0,
        40.0,
        "h02_away",
        "H02: Stay OUT. No face, shoulder, arm, or torso anywhere in frame.",
    ),
    (40.0, 45.0, "transition_away_recovery", "Return and sit naturally."),
    (45.0, 55.0, "h03_away_recovery", "H03: Work normally. Face visible."),
    (55.0, 60.0, "transition_phone", "Prepare H04: pick up a visible phone."),
    (60.0, 75.0, "h04_phone", "H04: Use a clearly visible PHONE naturally."),
    (75.0, 80.0, "transition_phone_recovery", "Put the phone fully out of view."),
    (80.0, 90.0, "h05_phone_recovery", "H05: Work normally. NO PHONE."),
    (90.0, 95.0, "transition_face_cover", "Prepare H06: cover face, keep torso visible."),
    (95.0, 103.0, "h06_face_cover", "H06: Cover FACE with blank paper; torso visible."),
    (103.0, 108.0, "transition_face_recovery", "Remove paper and sit naturally."),
    (108.0, 116.0, "h07_face_recovery", "H07: Work normally. Face visible."),
    (116.0, 121.0, "transition_camera_cover", "Prepare H08: fully cover CAMERA lens."),
    (121.0, 129.0, "h08_camera_cover", "H08: Keep CAMERA lens fully covered."),
    (129.0, 134.0, "transition_final", "Uncover camera and sit naturally."),
    (134.0, 144.0, "h09_final", "H09: Final normal work. Face visible."),
)

ATTENTION_BOUNDARY_CASES = {
    "a01": {"expected_state": "compliant", "label": "natural_center"},
    "a02": {"expected_state": "violation", "label": "outside_left_low"},
    "a03": {"expected_state": "compliant", "label": "center_recovery_1"},
    "a04": {"expected_state": "violation", "label": "outside_right_low"},
    "a05": {"expected_state": "compliant", "label": "center_recovery_2"},
    "a06": {"expected_state": "compliant", "label": "near_inside_left_low"},
    "a07": {"expected_state": "compliant", "label": "near_inside_right_low"},
    "a08": {"expected_state": "compliant", "label": "far_center"},
    "a09": {"expected_state": "compliant", "label": "far_inside_left_low"},
    "a10": {"expected_state": "compliant", "label": "far_inside_right_low"},
    "a11": {"expected_state": "compliant", "label": "final_center"},
}

GUIDED_DEBUG_PLANS = {
    "full": GUIDED_DEBUG_CUES,
    "attention-minute": ATTENTION_MINUTE_CUES,
    "attention-boundary": ATTENTION_BOUNDARY_CUES,
    "c-closeout-holdout": C_CLOSEOUT_HOLDOUT_CUES,
}


def _guided_case_metadata(cue_id: str) -> dict[str, object] | None:
    if cue_id.startswith("prep_a"):
        phase = "preparation"
        case_id = cue_id.removeprefix("prep_")
    elif cue_id.startswith("score_a"):
        phase = "scored_hold"
        case_id = cue_id.removeprefix("score_")
    else:
        return None
    configured = ATTENTION_BOUNDARY_CASES.get(case_id)
    if configured is None:
        return None
    return {
        "case_id": case_id.upper(),
        "phase": phase,
        "scored": phase == "scored_hold",
        **configured,
    }


def _record_guided_case_observation(
    summaries: dict[str, dict[str, object]],
    active_violation_starts: dict[str, float],
    case_metadata: dict[str, object] | None,
    elapsed_seconds: float,
    state: str,
) -> None:
    current_case_id = (
        str(case_metadata["case_id"])
        if case_metadata is not None and bool(case_metadata["scored"])
        else None
    )
    for active_case_id in tuple(active_violation_starts):
        if active_case_id == current_case_id:
            continue
        entry = summaries[active_case_id]
        end_seconds = float(entry["last_elapsed_seconds"])
        start_seconds = active_violation_starts.pop(active_case_id)
        runs = entry["exact_violation_runs"]
        assert isinstance(runs, list)
        runs.append(
            {
                "start_seconds": round(start_seconds, 3),
                "end_seconds": round(end_seconds, 3),
                "duration_seconds": round(max(0.0, end_seconds - start_seconds), 3),
            }
        )
    if current_case_id is None or case_metadata is None:
        return
    entry = summaries.setdefault(
        current_case_id,
        {
            "case_id": current_case_id,
            "label": case_metadata["label"],
            "expected_state": case_metadata["expected_state"],
            "first_elapsed_seconds": round(elapsed_seconds, 3),
            "last_elapsed_seconds": round(elapsed_seconds, 3),
            "state_counts": {},
            "exact_violation_runs": [],
        },
    )
    entry["last_elapsed_seconds"] = round(elapsed_seconds, 3)
    counts = entry["state_counts"]
    assert isinstance(counts, dict)
    counts[state] = counts.get(state, 0) + 1
    if state == "violation":
        active_violation_starts.setdefault(current_case_id, elapsed_seconds)
    elif current_case_id in active_violation_starts:
        start_seconds = active_violation_starts.pop(current_case_id)
        runs = entry["exact_violation_runs"]
        assert isinstance(runs, list)
        runs.append(
            {
                "start_seconds": round(start_seconds, 3),
                "end_seconds": round(elapsed_seconds, 3),
                "duration_seconds": round(max(0.0, elapsed_seconds - start_seconds), 3),
            }
        )


def _finalize_guided_case_summaries(
    summaries: dict[str, dict[str, object]],
    active_violation_starts: dict[str, float],
) -> list[dict[str, object]]:
    _record_guided_case_observation(
        summaries,
        active_violation_starts,
        None,
        0.0,
        "unknown",
    )
    rendered = []
    for case_id in ATTENTION_BOUNDARY_CASES:
        public_case_id = case_id.upper()
        entry = summaries.get(public_case_id)
        if entry is None:
            continue
        counts = entry["state_counts"]
        runs = entry["exact_violation_runs"]
        assert isinstance(counts, dict)
        assert isinstance(runs, list)
        total = sum(int(value) for value in counts.values())
        rendered.append(
            {
                **entry,
                "observation_count": total,
                "state_ratios": {
                    state: round(int(count) / total, 4) if total else 0.0
                    for state, count in counts.items()
                },
                "longest_exact_violation_seconds": round(
                    max(
                        (float(run["duration_seconds"]) for run in runs),
                        default=0.0,
                    ),
                    3,
                ),
            }
        )
    return rendered


def _guided_debug_cue(
    elapsed_seconds: float,
    cues: tuple[tuple[float, float, str, str], ...] = GUIDED_DEBUG_CUES,
) -> tuple[str, str, float]:
    for start, end, cue_id, instruction in cues:
        if start <= elapsed_seconds < end:
            return cue_id, instruction, end - elapsed_seconds
    return "complete", "Test complete", 0.0


def _show_guided_debug_cue(
    frame: object,
    elapsed_seconds: float,
    cues: tuple[tuple[float, float, str, str], ...],
) -> tuple[bool, str]:
    import cv2

    cue_id, instruction, remaining = _guided_debug_cue(elapsed_seconds, cues)
    case_metadata = _guided_case_metadata(cue_id)
    cue_index = next(
        (
            index
            for index, (_, _, configured_id, _) in enumerate(cues, 1)
            if configured_id == cue_id
        ),
        len(cues),
    )
    preview = frame.copy()
    cv2.rectangle(preview, (0, 0), (preview.shape[1], 112), (20, 20, 20), -1)
    cv2.putText(
        preview,
        (
            f"CASE {case_metadata['case_id']}   {str(case_metadata['phase']).upper()}"
            if case_metadata is not None
            else f"STEP {cue_index}/{len(cues)}   {cue_id.upper()}"
        ),
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        instruction,
        (18, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        (
            f"TIME {remaining:04.1f}s   ESC cancels   model result hidden"
            if case_metadata is not None
            else f"HOLD {remaining:04.1f}s   ESC cancels   model result hidden"
        ),
        (18, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("Observer guided debug test", preview)
    return (cv2.waitKey(1) & 0xFF) == 27, cue_id


def _doctor(config_path: Path | None) -> int:
    config = load_config(config_path)
    packages = {
        name: importlib.util.find_spec(name) is not None
        for name in ("cv2", "mediapipe", "numpy", "psutil")
    }
    version_supported = sys.version_info[:2] in {(3, 11), (3, 12)}
    report = {
        "python": platform.python_version(),
        "python_supported": version_supported,
        "platform": platform.platform(),
        "packages": packages,
        "config_valid": True,
        "camera": asdict(config.camera),
        "resource_budget": asdict(config.resources),
        "rules": [asdict(rule) for rule in config.rules],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if version_supported and all(packages.values()) else 1


def _yolo_doctor(model_path: Path) -> int:
    """Check optional YOLO ONNX Runtime and model-file readiness."""

    report = detector_runtime_report(model_path)
    report["model"]["path"] = str(model_path.resolve())
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ready"] else 1


def _yolo_replay(
    video_path: Path,
    model_path: Path,
    output_path: Path,
    score_threshold: float,
    iou_threshold: float,
    input_size: int,
    sample_frames: int | None,
) -> int:
    """Replay one saved video through YOLO without opening the camera."""

    import cv2

    from observer.object_detection import YoloOnnxBackend

    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")
    backend = YoloOnnxBackend(
        model_path,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
        input_size=input_size,
        single_class_name="cell_phone",
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if sample_frames is not None and sample_frames <= 0:
        raise ValueError("sample_frames must be positive")
    sampled_indices: list[int] | None = None
    if sample_frames is not None:
        if total_frame_count <= 0:
            raise RuntimeError("video does not expose a usable frame count for sampling")
        sample_count = min(sample_frames, total_frame_count)
        if sample_count == 1:
            sampled_indices = [total_frame_count // 2]
        else:
            sampled_indices = sorted(
                {
                    round(index * (total_frame_count - 1) / (sample_count - 1))
                    for index in range(sample_count)
                }
            )
    frames: list[dict[str, object]] = []

    def analyze_frame(frame: object, index: int) -> None:
        timestamp_ms = round((index / fps) * 1000) if fps > 0 else index
        detections = backend.detect(frame, timestamp_ms)
        frames.append(
            {
                "frame_index": index,
                "timestamp_seconds": round(index / fps, 4) if fps > 0 else None,
                "latency_ms": round(backend.last_latency_ms, 3),
                "detections": [
                    {
                        "class_name": detection.class_name,
                        "confidence": round(detection.confidence, 6),
                        "box": {
                            "x": round(detection.box.x, 6),
                            "y": round(detection.box.y, 6),
                            "width": round(detection.box.width, 6),
                            "height": round(detection.box.height, 6),
                        },
                    }
                    for detection in detections
                ],
            }
        )

    try:
        if sampled_indices is not None:
            for index in sampled_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"could not read sampled frame {index}: {video_path}")
                analyze_frame(frame, index)
        else:
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                analyze_frame(frame, index)
                index += 1
    finally:
        capture.release()
        backend.close()
    phone_frames = [
        frame
        for frame in frames
        if any(is_phone_class_name(item["class_name"]) for item in frame["detections"])
    ]
    latencies = [float(frame["latency_ms"]) for frame in frames]
    report = {
        "video": str(video_path.resolve()),
        "model": str(model_path.resolve()),
        "frames": frames,
        "summary": {
            "source_frame_count": total_frame_count,
            "frame_count": len(frames),
            "sampling": "uniform" if sampled_indices is not None else "all",
            "fps": fps,
            "phone_positive_frames": len(phone_frames),
            "phone_positive_fraction": round(len(phone_frames) / len(frames), 6)
            if frames
            else 0.0,
            "latency_average_ms": round(sum(latencies) / len(latencies), 3)
            if latencies
            else 0.0,
            "latency_p95_ms": round(percentile(latencies, 0.95), 3) if latencies else 0.0,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"report": str(output_path.resolve()), **report["summary"]}, indent=2))
    return 0


def _camera_smoke(config_path: Path | None, seconds: float) -> int:
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    config = load_config(config_path)
    camera = CameraWorker(config.camera)
    sampler = ResourceSampler()
    samples = []
    start = time.monotonic()
    last_sequence = 0
    observed_frames = 0
    try:
        camera.start()
        startup_stats = camera.stats
        if not startup_stats.opened:
            result = {
                "duration_seconds": round(time.monotonic() - start, 2),
                "observed_latest_frames": 0,
                "camera": asdict(startup_stats),
                "resources": asdict(
                    summarize_resources([sampler.sample()], config.resources)
                ),
            }
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 2
        while time.monotonic() - start < seconds:
            snapshot = camera.buffer.latest(after_sequence=last_sequence)
            if snapshot is not None:
                observed_frames += 1
                last_sequence = snapshot.sequence
            samples.append(sampler.sample())
            time.sleep(1.0)
    finally:
        camera.stop()

    result = {
        "duration_seconds": round(time.monotonic() - start, 2),
        "observed_latest_frames": observed_frames,
        "camera": asdict(camera.stats),
        "resources": asdict(summarize_resources(samples, config.resources)),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if camera.stats.frames_captured > 0 and camera.stats.last_error is None else 2


def _vision_smoke(
    config_path: Path | None,
    models_dir: Path,
    seconds: float,
    profile_path: Path | None = None,
    evidence_dir: Path | None = None,
    report_path: Path | None = None,
    debug_video_path: Path | None = None,
    guided_debug_cues: bool = False,
    guided_debug_plan: str | None = None,
    detector_backend: str = 'efficientdet',
    yolo_model_path: Path | None = None,
) -> int:
    phone_score_threshold = 0.25
    phone_retain_seconds = 2.0
    phone_center_distance = 0.18
    if detector_backend not in ('efficientdet', 'yolo'):
        raise ValueError('unsupported detector backend')
    if detector_backend == 'yolo' and (yolo_model_path is None or not yolo_model_path.is_file()):
        raise ValueError('YOLO requires an existing --yolo-model')
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    if debug_video_path is not None:
        if debug_video_path.suffix.lower() != ".mp4":
            raise ValueError("temporary debug video path must end in .mp4")
        timestamps_path = debug_video_path.with_suffix(
            debug_video_path.suffix + ".timestamps.json"
        )
        trace_path = debug_video_path.with_suffix(debug_video_path.suffix + ".trace.jsonl")
        if debug_video_path.exists() or timestamps_path.exists() or trace_path.exists():
            raise FileExistsError(
                "temporary debug video or timestamp index already exists; choose a new path"
            )
    if guided_debug_cues and guided_debug_plan is not None:
        raise ValueError("choose either --guided-debug-cues or --guided-debug-plan")
    selected_plan = guided_debug_plan or ("full" if guided_debug_cues else None)
    selected_cues = GUIDED_DEBUG_PLANS.get(selected_plan, ())
    if selected_plan is not None and not selected_cues:
        raise ValueError(f"unknown guided debug plan: {selected_plan}")
    guided_debug_enabled = bool(selected_cues)
    if guided_debug_enabled:
        seconds = float(selected_cues[-1][1])
    if guided_debug_enabled and debug_video_path is None:
        raise ValueError("guided debug cues require --debug-video for objective review")
    from observer.vision import MediapipeVisionProbe

    config = load_config(config_path)
    models = verify_model_directory(models_dir)
    camera = CameraWorker(config.camera)
    probe = None
    samples = []
    last_sequence = 0
    next_resource_sample = 0.0
    last_observation = None
    debug_video_writer = None
    debug_video_timestamps: list[float] = []
    debug_trace_handle = None
    last_debug_cue_id: str | None = None
    guided_debug_cancelled = False
    guided_case_summaries: dict[str, dict[str, object]] = {}
    guided_case_active_violations: dict[str, float] = {}
    session_attention_setup: dict[str, object] | None = None
    signal_state_counts: dict[str, dict[str, int]] = {}
    signal_reason_counts: dict[str, dict[str, int]] = {}
    signal_metric_values: dict[str, dict[str, list[float]]] = {}
    feature_metric_values: dict[str, dict[str, list[float]]] = {
        "attention": {},
        "posture": {},
    }
    feature_reason_counts: dict[str, dict[str, int]] = {
        "attention": {},
        "posture": {},
    }
    previous_feature_runs = {"attention": 0, "posture": 0}
    evidence_store = None
    evidence_records: list[dict[str, object]] = []
    evidence_counts: dict[str, int] = {}
    evidence_regular_counts: dict[str, int] = {}
    evidence_last_saved: dict[str, float] = {}
    sustained_evidence_saved: set[str] = set()
    active_violation_intervals: dict[str, dict[str, float]] = {}
    violation_intervals: dict[str, list[dict[str, object]]] = {}
    tracked_evidence_signals = {
        "away",
        "phone_held",
        "attention_away",
        "posture_deviation",
        "ergonomic_risk",
    }
    evidence_trigger_seconds = {
        rule.signal_name: rule.trigger_after_seconds
        for rule in config.rules
        if rule.signal_name in tracked_evidence_signals
    }
    last_elapsed_seconds = 0.0
    if evidence_dir is not None:
        from observer.validation import ErrorEvidenceStore, EvidencePolicy

        evidence_store = ErrorEvidenceStore(
            evidence_dir, EvidencePolicy(enabled=True, retention_days=7)
        )
    try:
        if not camera.start():
            print(json.dumps({"camera": asdict(camera.stats)}, indent=2, ensure_ascii=False))
            return 2
        profile = None
        if profile_path is not None:
            from observer.calibration import load_profile
            from observer.calibration_workflow import current_environment

            first_snapshot = camera.buffer.latest()
            deadline = time.monotonic() + 5.0
            while first_snapshot is None and time.monotonic() < deadline:
                time.sleep(0.01)
                first_snapshot = camera.buffer.latest()
            if first_snapshot is None:
                raise RuntimeError("camera opened but did not produce a frame")
            environment = current_environment(config, first_snapshot.frame.shape)
            profile = load_profile(profile_path, environment)
        object_backend = None
        if detector_backend == 'yolo':
            from observer.object_detection import YoloOnnxBackend
            object_backend = YoloOnnxBackend(yolo_model_path, single_class_name='cell_phone')
        probe = MediapipeVisionProbe(
            models,
            config.inference,
            profile,
            phone_score_threshold=phone_score_threshold,
            phone_retain_seconds=phone_retain_seconds,
            phone_center_distance=phone_center_distance,
            object_backend=object_backend,
        )
        if profile is not None:
            from observer.alignment import (
                AlignmentCancelled,
                guide_attention_session_baseline,
            )

            try:
                session_attention_setup = asdict(
                    guide_attention_session_baseline(camera, probe)
                )
            except (AlignmentCancelled, RuntimeError) as exc:
                print(
                    json.dumps(
                        {
                            "status": "session_attention_setup_failed",
                            "message": str(exc),
                            "frames_saved": 0,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )
                return 2
        sampler = ResourceSampler()
        started = time.monotonic()
        next_resource_sample = started
        while time.monotonic() - started < seconds:
            snapshot = camera.buffer.latest(after_sequence=last_sequence)
            if snapshot is None:
                time.sleep(0.005)
                continue
            last_sequence = snapshot.sequence
            last_elapsed_seconds = snapshot.captured_monotonic - started
            guided_cue_id = "complete"
            guided_case_metadata = None
            if guided_debug_enabled:
                guided_cue_id, _, _ = _guided_debug_cue(
                    last_elapsed_seconds, selected_cues
                )
                guided_case_metadata = _guided_case_metadata(guided_cue_id)
            if debug_video_path is not None:
                if debug_video_writer is None:
                    import cv2

                    debug_video_path.parent.mkdir(parents=True, exist_ok=True)
                    frame_height, frame_width = snapshot.frame.shape[:2]
                    debug_video_writer = cv2.VideoWriter(
                        str(debug_video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        config.camera.capture_fps,
                        (frame_width, frame_height),
                    )
                    if not debug_video_writer.isOpened():
                        raise RuntimeError(
                            f"temporary debug video could not be opened: {debug_video_path}"
                        )
                    trace_path = debug_video_path.with_suffix(
                        debug_video_path.suffix + ".trace.jsonl"
                    )
                    debug_trace_handle = trace_path.open("x", encoding="utf-8")
                debug_video_writer.write(snapshot.frame)
                debug_video_timestamps.append(last_elapsed_seconds)
            probe.process_due(snapshot.frame, snapshot.captured_monotonic)
            last_observation = probe.observation(snapshot.captured_monotonic)
            if debug_trace_handle is not None:
                guided_cue = None
                if guided_debug_enabled:
                    cue_id, _, remaining = _guided_debug_cue(
                        last_elapsed_seconds, selected_cues
                    )
                    guided_cue = {
                        "cue_id": cue_id,
                        "remaining_seconds": round(remaining, 3),
                        "case": guided_case_metadata,
                    }
                debug_trace_handle.write(
                    json.dumps(
                        {
                            "frame_index": len(debug_video_timestamps) - 1,
                            "elapsed_seconds": round(last_elapsed_seconds, 3),
                            "guided_cue": guided_cue,
                            "signals": {
                                signal_name: {
                                    "state": observed.state.value,
                                    "reason": observed.reason,
                                    "metrics": dict(observed.metrics),
                                }
                                for signal_name, observed in last_observation.signals.items()
                            },
                            "derived": probe.derived_snapshot(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if guided_debug_enabled:
                cancelled, cue_id = _show_guided_debug_cue(
                    snapshot.frame, last_elapsed_seconds, selected_cues
                )
                if cue_id != last_debug_cue_id:
                    try:
                        import winsound

                        winsound.MessageBeep(winsound.MB_OK)
                    except (ImportError, RuntimeError):
                        pass
                    last_debug_cue_id = cue_id
                if cancelled:
                    guided_debug_cancelled = True
                    break
            if selected_plan == "attention-boundary":
                attention_signal = last_observation.signals.get("attention_away")
                if attention_signal is not None:
                    _record_guided_case_observation(
                        guided_case_summaries,
                        guided_case_active_violations,
                        guided_case_metadata,
                        last_elapsed_seconds,
                        attention_signal.state.value,
                    )
            for name, signal in last_observation.signals.items():
                counts = signal_state_counts.setdefault(name, {})
                counts[signal.state.value] = counts.get(signal.state.value, 0) + 1
                if signal.reason:
                    reasons = signal_reason_counts.setdefault(name, {})
                    reasons[signal.reason] = reasons.get(signal.reason, 0) + 1
                metrics = signal_metric_values.setdefault(name, {})
                for metric_name, value in signal.metrics.items():
                    metrics.setdefault(metric_name, []).append(value)
                if name in tracked_evidence_signals:
                    _track_violation_interval(
                        name,
                        signal.state.value,
                        last_elapsed_seconds,
                        active_violation_intervals,
                        violation_intervals,
                    )
                capture_kind = _evidence_capture_kind(
                    name,
                    signal.state.value,
                    last_elapsed_seconds,
                    active_violation_intervals,
                    evidence_regular_counts,
                    evidence_last_saved,
                    sustained_evidence_saved,
                    evidence_trigger_seconds,
                    snapshot.captured_monotonic,
                )
                if evidence_store is not None and capture_kind is not None:
                    sequence_number = evidence_counts.get(name, 0) + 1
                    saved = evidence_store.save(
                        f"natural-{name}-{sequence_number}",
                        snapshot.frame,
                        is_error=True,
                    )
                    if saved is not None:
                        absolute_path = (
                            evidence_dir / saved["relative_path"]
                        ).resolve()
                        evidence_records.append(
                            {
                                "signal": name,
                                "sequence": sequence_number,
                                "capture_kind": capture_kind,
                                "elapsed_seconds": round(
                                    snapshot.captured_monotonic - started, 3
                                ),
                                "reason": signal.reason,
                                "metrics": dict(signal.metrics),
                                "all_signals": {
                                    signal_name: {
                                        "state": observed.state.value,
                                        "reason": observed.reason,
                                        "metrics": dict(observed.metrics),
                                    }
                                    for signal_name, observed in last_observation.signals.items()
                                },
                                "path": str(absolute_path),
                                "sha256": saved["sha256"],
                                "delete_after_utc": saved["delete_after_utc"],
                            }
                        )
                        evidence_counts[name] = sequence_number
                        evidence_last_saved[name] = snapshot.captured_monotonic
                        if capture_kind == "sustained_threshold":
                            sustained_evidence_saved.add(name)
                        else:
                            evidence_regular_counts[name] = (
                                evidence_regular_counts.get(name, 0) + 1
                            )
            current_feature_runs = {
                "attention": probe.measurements.face.runs,
                "posture": probe.measurements.pose.runs,
            }
            for kind, feature in (
                ("attention", probe.latest_attention_features),
                ("posture", probe.latest_posture_features),
            ):
                if feature is None or current_feature_runs[kind] == previous_feature_runs[kind]:
                    continue
                for metric_name, value in feature.values.items():
                    feature_metric_values[kind].setdefault(metric_name, []).append(value)
                for reason in feature.reasons:
                    reasons = feature_reason_counts[kind]
                    reasons[reason] = reasons.get(reason, 0) + 1
                previous_feature_runs[kind] = current_feature_runs[kind]
            now = time.monotonic()
            if now >= next_resource_sample:
                samples.append(sampler.sample())
                next_resource_sample = now + 1.0
    finally:
        if probe is not None:
            probe.close()
        camera.stop()
        if debug_video_writer is not None:
            debug_video_writer.release()
        if debug_trace_handle is not None:
            debug_trace_handle.close()
        if guided_debug_enabled:
            import cv2

            try:
                cv2.destroyWindow("Observer guided debug test")
            except cv2.error:
                pass

    _finish_violation_intervals(
        active_violation_intervals,
        violation_intervals,
        last_elapsed_seconds,
    )
    debug_video = _finalize_debug_video(
        debug_video_path,
        debug_video_timestamps,
        config.camera.capture_fps,
    )

    result = {
        "duration_seconds": round(last_elapsed_seconds, 3),
        "session_attention_setup": session_attention_setup,
        "guided_debug": {
            "enabled": guided_debug_enabled,
            "plan": selected_plan,
            "cancelled": guided_debug_cancelled,
            "cues": [
                {
                    "start_seconds": start,
                    "end_seconds": end,
                    "cue_id": cue_id,
                    "instruction": instruction,
                    "case": _guided_case_metadata(cue_id),
                }
                for start, end, cue_id, instruction in selected_cues
            ]
            if guided_debug_enabled
            else [],
            "case_summaries": _finalize_guided_case_summaries(
                guided_case_summaries,
                guided_case_active_violations,
            )
            if selected_plan == "attention-boundary"
            else [],
            "requires_video_truth_review": selected_plan
            in {"attention-boundary", "c-closeout-holdout"},
        },
        "camera": asdict(camera.stats),
        "vision": probe.summary() if probe is not None else None,
        "object_detector": {
            "backend": detector_backend,
            "model_path": str(yolo_model_path.resolve()) if yolo_model_path else None,
            "sha256": sha256_file(yolo_model_path) if yolo_model_path else None,
            "phone_score_threshold": phone_score_threshold,
            "phone_retain_seconds": phone_retain_seconds,
            "phone_center_distance": phone_center_distance,
        },
        "last_signals": (
            {
                name: {
                    "state": value.state.value,
                    "confidence": value.confidence,
                    "reason": value.reason,
                    "metrics": dict(value.metrics),
                }
                for name, value in last_observation.signals.items()
            }
            if last_observation is not None
            else None
        ),
        "signal_state_counts": signal_state_counts,
        "signal_reason_counts": signal_reason_counts,
        "signal_metric_summary": _numeric_summary(signal_metric_values),
        "feature_reason_counts": feature_reason_counts,
        "feature_metric_summary": _numeric_summary(feature_metric_values),
        "evidence": {
            "enabled": evidence_store is not None,
            "maximum_regular_per_signal": 3 if evidence_store is not None else 0,
            "maximum_sustained_per_signal": 1 if evidence_store is not None else 0,
            "minimum_interval_seconds": 30.0,
            "records": evidence_records,
        },
        "temporary_debug_video": debug_video,
        "violation_intervals": violation_intervals,
        "resources": asdict(summarize_resources(samples, config.resources)),
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered, encoding="utf-8")
    print(rendered)
    if probe is None:
        return 3
    runs = probe.measurements
    return 0 if runs.face.runs and runs.pose.runs and runs.object.runs else 4


def _finalize_debug_video(
    video_path: Path | None,
    frame_elapsed_seconds: list[float],
    nominal_fps: float,
) -> dict[str, object]:
    if video_path is None:
        return {"enabled": False}
    if not video_path.exists():
        return {"enabled": True, "written": False, "path": str(video_path.resolve())}
    timestamps_path = video_path.with_suffix(video_path.suffix + ".timestamps.json")
    trace_path = video_path.with_suffix(video_path.suffix + ".trace.jsonl")
    timestamp_document = {
        "video_path": str(video_path.resolve()),
        "nominal_fps": nominal_fps,
        "frame_count": len(frame_elapsed_seconds),
        "frame_elapsed_seconds": [round(value, 3) for value in frame_elapsed_seconds],
    }
    timestamps_path.write_text(
        json.dumps(timestamp_document, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    delete_after = datetime.now(timezone.utc) + timedelta(days=1)
    return {
        "enabled": True,
        "written": True,
        "temporary": True,
        "path": str(video_path.resolve()),
        "timestamps_path": str(timestamps_path.resolve()),
        "trace_path": str(trace_path.resolve()) if trace_path.exists() else None,
        "frame_count": len(frame_elapsed_seconds),
        "nominal_fps": nominal_fps,
        "size_bytes": video_path.stat().st_size,
        "sha256": sha256_file(video_path),
        "timestamps_sha256": sha256_file(timestamps_path),
        "trace_sha256": sha256_file(trace_path) if trace_path.exists() else None,
        "review_then_delete": True,
        "suggested_delete_after_utc": delete_after.isoformat(),
    }


def _numeric_summary(
    grouped: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        group: {
            name: {
                "minimum": min(values),
                "p05": percentile(values, 0.05),
                "median": percentile(values, 0.5),
                "p95": percentile(values, 0.95),
                "maximum": max(values),
                "average": sum(values) / len(values),
            }
            for name, values in metrics.items()
            if values
        }
        for group, metrics in grouped.items()
    }


def _evidence_capture_kind(
    signal_name: str,
    state: str,
    elapsed_seconds: float,
    active: dict[str, dict[str, float]],
    regular_counts: dict[str, int],
    last_saved: dict[str, float],
    sustained_saved: set[str],
    trigger_seconds: dict[str, float],
    captured_monotonic: float,
) -> str | None:
    if state != "violation":
        return None
    current = active.get(signal_name)
    threshold = trigger_seconds.get(signal_name)
    if (
        current is not None
        and threshold is not None
        and signal_name not in sustained_saved
        and elapsed_seconds - current["start_seconds"] >= threshold
    ):
        return "sustained_threshold"
    if (
        regular_counts.get(signal_name, 0) < 3
        and captured_monotonic - last_saved.get(signal_name, float("-inf")) >= 30.0
    ):
        return "representative"
    return None


def _track_violation_interval(
    signal_name: str,
    state: str,
    elapsed_seconds: float,
    active: dict[str, dict[str, float]],
    completed: dict[str, list[dict[str, object]]],
    *,
    gap_tolerance_seconds: float = 0.5,
) -> None:
    current = active.get(signal_name)
    if state == "violation":
        if current is None:
            active[signal_name] = {
                "start_seconds": elapsed_seconds,
                "last_violation_seconds": elapsed_seconds,
            }
        else:
            current["last_violation_seconds"] = elapsed_seconds
        return
    if current is None:
        return
    if elapsed_seconds - current["last_violation_seconds"] <= gap_tolerance_seconds:
        return
    _complete_violation_interval(signal_name, active, completed)


def _complete_violation_interval(
    signal_name: str,
    active: dict[str, dict[str, float]],
    completed: dict[str, list[dict[str, object]]],
) -> None:
    current = active.pop(signal_name, None)
    if current is None:
        return
    start = current["start_seconds"]
    end = current["last_violation_seconds"]
    completed.setdefault(signal_name, []).append(
        {
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "duration_seconds": round(max(0.0, end - start), 3),
        }
    )


def _finish_violation_intervals(
    active: dict[str, dict[str, float]],
    completed: dict[str, list[dict[str, object]]],
    elapsed_seconds: float,
) -> None:
    for signal_name, current in active.items():
        current["last_violation_seconds"] = min(
            elapsed_seconds, current["last_violation_seconds"]
        )
    for signal_name in tuple(active):
        _complete_violation_interval(signal_name, active, completed)


def _calibrate(config_path: Path | None, models_dir: Path, output_path: Path) -> int:
    from observer.calibration_workflow import CalibrationInputError, run_guided_calibration

    config = load_config(config_path)
    models = verify_model_directory(models_dir)
    try:
        profile = run_guided_calibration(config, models, output_path)
    except CalibrationInputError as exc:
        print(
            json.dumps(
                {
                    "status": "camera_setup_failed",
                    "message": str(exc),
                    "profile_preserved": output_path.exists(),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 2
    result = {
        "profile_path": str(output_path.resolve()),
        "profile_id": profile.profile_id,
        "schema_version": profile.schema_version,
        "posture_valid_samples": profile.posture.valid_sample_count,
        "posture_coverage": profile.posture.coverage,
        "attention_valid_samples": profile.attention.valid_sample_count,
        "attention_coverage": profile.attention.coverage,
        "attention_training_rmse": profile.attention.training_rmse,
        "attention_validation_rmse": profile.attention.validation_rmse,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _align(
    config_path: Path | None,
    models_dir: Path,
    profile_path: Path,
    stable_seconds: float,
    timeout_seconds: float,
    verify_seconds: float,
    require_posture: bool,
) -> int:
    from observer.alignment import AlignmentCancelled, AlignmentTimeout, run_alignment

    def assessment_payload(value):
        if value is None:
            return None
        payload = asdict(value)
        payload["posture_state"] = value.posture_state.value
        return payload

    config = load_config(config_path)
    models = verify_model_directory(models_dir)
    try:
        result = run_alignment(
            config,
            models,
            profile_path,
            stable_seconds=stable_seconds,
            timeout_seconds=timeout_seconds,
            verify_seconds=verify_seconds,
            require_posture=require_posture,
        )
    except AlignmentCancelled as exc:
        print(json.dumps({"aligned": False, "reason": str(exc)}, indent=2))
        return 2
    except AlignmentTimeout as exc:
        print(
            json.dumps(
                {
                    "aligned": False,
                    "reason": str(exc),
                    "longest_stable_seconds": round(exc.longest_stable_seconds, 3),
                    "best_assessment": assessment_payload(exc.best_assessment),
                    "last_assessment": assessment_payload(exc.last_assessment),
                    "diagnostics": exc.diagnostics,
                    "frames_saved": 0,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 3
    print(json.dumps({"aligned": True, **asdict(result)}, indent=2, ensure_ascii=False))
    return 0


def _profile_check(profile_path: Path) -> int:
    from observer.calibration import load_profile

    profile = load_profile(profile_path)
    result = {
        "valid": True,
        "path": str(profile_path.resolve()),
        "sha256": sha256_file(profile_path),
        "profile_id": profile.profile_id,
        "schema_version": profile.schema_version,
        "feature_version": profile.posture.feature_version,
        "environment": asdict(profile.environment),
        "posture_coverage": profile.posture.coverage,
        "attention_coverage": profile.attention.coverage,
        "attention_validation_rmse": profile.attention.validation_rmse,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _validation_report(trials_path: Path, output_path: Path) -> int:
    from observer.validation import read_trials, write_metrics_report

    records = read_trials(trials_path)
    report = write_metrics_report(records, output_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _validation_scenarios() -> int:
    from observer.live_validation import scenario_catalog

    print(json.dumps(scenario_catalog(), indent=2, ensure_ascii=False))
    return 0


def _validate_live(
    config_path: Path | None,
    models_dir: Path,
    profile_path: Path,
    dataset_role: str,
    signals: str,
    scenarios: str | None,
    output_root: Path,
    repetitions: int | None,
    save_error_evidence: bool,
    save_trial_videos: bool,
    phone_score_threshold: float,
    phone_retain_seconds: float,
    phone_center_distance: float,
    detector_backend: str = "efficientdet",
    yolo_model_path: Path | None = None,
) -> int:
    from observer.alignment import AlignmentCancelled, AlignmentTimeout
    from observer.live_validation import run_live_validation

    def assessment_payload(value):
        if value is None:
            return None
        payload = asdict(value)
        payload["posture_state"] = value.posture_state.value
        return payload

    config = load_config(config_path)
    models = verify_model_directory(models_dir)
    signal_names = tuple(name.strip() for name in signals.split(",") if name.strip())
    scenario_ids = tuple(
        name.strip() for name in (scenarios or "").split(",") if name.strip()
    )
    try:
        batch_directory = run_live_validation(
            config,
            models,
            profile_path,
            dataset_role=dataset_role,
            signal_names=signal_names,
            scenario_ids=scenario_ids,
            output_root=output_root,
            repetitions_override=repetitions,
            save_error_evidence=save_error_evidence,
            save_trial_videos=save_trial_videos,
            phone_score_threshold=phone_score_threshold,
            phone_retain_seconds=phone_retain_seconds,
            phone_center_distance=phone_center_distance,
            detector_backend=detector_backend,
            yolo_model_path=yolo_model_path,
        )
    except AlignmentCancelled as exc:
        print(json.dumps({"aligned": False, "reason": str(exc)}, indent=2))
        return 2
    except AlignmentTimeout as exc:
        print(
            json.dumps(
                {
                    "aligned": False,
                    "reason": str(exc),
                    "longest_stable_seconds": round(
                        exc.longest_stable_seconds, 3
                    ),
                    "best_assessment": assessment_payload(exc.best_assessment),
                    "last_assessment": assessment_payload(exc.last_assessment),
                    "diagnostics": exc.diagnostics,
                    "frames_saved": 0,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 3
    print(
        json.dumps(
            {
                "batch_directory": str(batch_directory.resolve()),
                "report": str((batch_directory / "report.json").resolve()),
                "evidence_enabled": save_error_evidence,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _validate_away_events(
    config_path: Path | None,
    models_dir: Path,
    profile_path: Path,
    dataset_role: str,
    output_root: Path,
    suite: str,
    detector_backend: str = "efficientdet",
    yolo_model_path: Path | None = None,
) -> int:
    from observer.live_validation import run_away_event_validation

    config = load_config(config_path)
    models = verify_model_directory(models_dir)
    batch_directory = run_away_event_validation(
        config,
        models,
        profile_path,
        dataset_role=dataset_role,
        output_root=output_root,
        suite=suite,
        detector_backend=detector_backend,
        yolo_model_path=yolo_model_path,
    )
    report_path = batch_directory / "event_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "batch_directory": str(batch_directory.resolve()),
                "report": str(report_path.resolve()),
                "passed": report["passed"],
                "frames_saved": 0,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if report["passed"] else 5


def _validate_phone_events(
    config_path: Path | None,
    models_dir: Path,
    profile_path: Path,
    dataset_role: str,
    output_root: Path,
    suite: str,
    detector_backend: str = "efficientdet",
    yolo_model_path: Path | None = None,
    phone_score_threshold: float = 0.25,
    phone_retain_seconds: float = 2.0,
    phone_center_distance: float = 0.18,
) -> int:
    from observer.live_validation import run_phone_event_validation

    config = load_config(config_path)
    models = verify_model_directory(models_dir)
    batch_directory = run_phone_event_validation(
        config,
        models,
        profile_path,
        dataset_role=dataset_role,
        output_root=output_root,
        suite=suite,
        detector_backend=detector_backend,
        yolo_model_path=yolo_model_path,
        phone_score_threshold=phone_score_threshold,
        phone_retain_seconds=phone_retain_seconds,
        phone_center_distance=phone_center_distance,
    )
    report_path = batch_directory / "event_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "batch_directory": str(batch_directory.resolve()),
                "report": str(report_path.resolve()),
                "passed": report["passed"],
                "frames_saved": 0,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if report["passed"] else 5


def _evidence_purge(index_path: Path, apply: bool) -> int:
    from observer.validation import purge_expired_evidence

    report = purge_expired_evidence(index_path, apply=apply)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="observer")
    parser.add_argument("--config", type=Path, help="optional JSON configuration path")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="validate runtime, packages, and configuration")
    yolo_doctor_parser = subparsers.add_parser(
        "yolo-doctor", help="check optional YOLO ONNX Runtime and model readiness"
    )
    yolo_doctor_parser.add_argument(
        "--model", type=Path, default=Path("models/yolo/yolo11n.onnx")
    )
    yolo_replay_parser = subparsers.add_parser(
        "yolo-replay", help="replay one saved video through YOLO without a camera"
    )
    yolo_replay_parser.add_argument("--video", type=Path, required=True)
    yolo_replay_parser.add_argument("--model", type=Path, required=True)
    yolo_replay_parser.add_argument("--output", type=Path, required=True)
    yolo_replay_parser.add_argument("--score-threshold", type=float, default=0.25)
    yolo_replay_parser.add_argument("--iou-threshold", type=float, default=0.45)
    yolo_replay_parser.add_argument("--input-size", type=int, default=640)
    yolo_replay_parser.add_argument(
        "--sample-frames",
        type=int,
        help="uniformly sample this many frames instead of replaying every frame",
    )
    camera_parser = subparsers.add_parser(
        "camera-smoke", help="capture without preview or recording and report resources"
    )
    camera_parser.add_argument("--seconds", type=float, default=15.0)
    vision_parser = subparsers.add_parser(
        "vision-smoke", help="run rate-limited Face, Pose, and Object models without recording"
    )
    vision_parser.add_argument("--seconds", type=float, default=30.0)
    vision_parser.add_argument("--models-dir", type=Path, default=Path("models"))
    vision_parser.add_argument("--profile", type=Path)
    vision_parser.add_argument('--detector-backend', choices=('efficientdet', 'yolo'), default='efficientdet')
    vision_parser.add_argument('--yolo-model', type=Path)
    vision_parser.add_argument(
        "--evidence-dir",
        type=Path,
        help="opt in to at most three local violation frames per signal",
    )
    vision_parser.add_argument(
        "--report",
        type=Path,
        help="write the complete JSON summary to this path",
    )
    vision_parser.add_argument(
        "--debug-video",
        type=Path,
        help="explicitly opt in to a temporary local MP4 plus timestamp and signal trace",
    )
    vision_parser.add_argument(
        "--guided-debug-cues",
        action="store_true",
        help="show the legacy five-minute timed action plan",
    )
    vision_parser.add_argument(
        "--guided-debug-plan",
        choices=tuple(GUIDED_DEBUG_PLANS),
        help="show a named timed action plan without exposing model results",
    )
    calibrate_parser = subparsers.add_parser(
        "calibrate", help="run guided posture and 17-point attention calibration"
    )
    calibrate_parser.add_argument("--models-dir", type=Path, default=Path("models"))
    calibrate_parser.add_argument(
        "--output", type=Path, default=Path("data/calibration/default.json")
    )
    align_parser = subparsers.add_parser(
        "align", help="return to the saved calibration position without recording"
    )
    align_parser.add_argument(
        "--profile", type=Path, default=Path("data/calibration/default.json")
    )
    align_parser.add_argument("--models-dir", type=Path, default=Path("models"))
    align_parser.add_argument("--stable-seconds", type=float, default=3.0)
    align_parser.add_argument("--timeout-seconds", type=float, default=120.0)
    align_parser.add_argument(
        "--verify-seconds",
        type=float,
        default=0.0,
        help="measure an immediate same-session baseline after alignment",
    )
    align_parser.add_argument(
        "--require-posture",
        action="store_true",
        help="also require the calibrated neutral posture before completing alignment",
    )
    profile_parser = subparsers.add_parser(
        "profile-check", help="validate a calibration profile without opening the camera"
    )
    profile_parser.add_argument("profile", type=Path)
    report_parser = subparsers.add_parser(
        "validation-report", help="summarize a Stage C trials JSONL file"
    )
    report_parser.add_argument("trials", type=Path)
    report_parser.add_argument("--output", type=Path, default=Path("data/validation/report.json"))
    subparsers.add_parser(
        "validation-scenarios", help="list built-in Stage C real validation scenarios"
    )
    live_parser = subparsers.add_parser(
        "validate-live", help="run a new isolated tuning or holdout camera batch"
    )
    live_parser.add_argument("--profile", type=Path, required=True)
    live_parser.add_argument("--models-dir", type=Path, default=Path("models"))
    live_parser.add_argument("--role", choices=("tuning", "holdout"), required=True)
    live_parser.add_argument(
        "--signals",
        required=True,
        help="comma-separated: away,phone_held,posture_deviation,attention_away",
    )
    live_parser.add_argument(
        "--scenarios",
        help="optional comma-separated scenario IDs for a focused rerun",
    )
    live_parser.add_argument("--output-root", type=Path, default=Path("data/validation"))
    live_parser.add_argument(
        "--repetitions", type=int, help="override only for dry runs; omit for protocol counts"
    )
    live_parser.set_defaults(save_error_evidence=True)
    live_parser.add_argument(
        "--no-error-evidence",
        action="store_false",
        dest="save_error_evidence",
        help="disable the approved one-error-frame validation policy",
    )
    live_parser.add_argument(
        "--debug-trial-videos",
        action="store_true",
        help="opt in to temporary videos of scored trial intervals only",
    )
    live_parser.add_argument(
        "--phone-score-threshold",
        type=float,
        default=0.25,
        help="experimental phone detector score threshold for tuning only",
    )
    live_parser.add_argument(
        "--phone-retain-seconds",
        type=float,
        default=2.0,
        help="experimental phone candidate retention seconds for tuning only",
    )
    live_parser.add_argument(
        "--phone-center-distance",
        type=float,
        default=0.18,
        help="experimental maximum phone candidate center distance for tuning only",
    )
    live_parser.add_argument(
        "--detector-backend",
        choices=("efficientdet", "yolo"),
        default="efficientdet",
        help="object detector for this isolated batch; default remains EfficientDet",
    )
    live_parser.add_argument(
        "--yolo-model",
        type=Path,
        help="YOLO ONNX model path (required with --detector-backend yolo)",
    )
    event_parser = subparsers.add_parser(
        "validate-away-events",
        help="run continuous Stage C away alert, penalty, and recovery scenarios",
    )
    event_parser.add_argument("--profile", type=Path, required=True)
    event_parser.add_argument("--models-dir", type=Path, default=Path("models"))
    event_parser.add_argument("--role", choices=("tuning", "holdout"), required=True)
    event_parser.add_argument(
        "--suite",
        choices=("full", "brief"),
        default="full",
        help="full event chain or focused below-threshold brief-away test",
    )
    event_parser.add_argument(
        "--output-root", type=Path, default=Path("data/validation")
    )
    event_parser.add_argument(
        "--detector-backend", choices=("efficientdet", "yolo"), default="efficientdet"
    )
    event_parser.add_argument("--yolo-model", type=Path)
    phone_event_parser = subparsers.add_parser(
        "validate-phone-events",
        help="run continuous Stage C phone alert, penalty, recovery, and false-positive scenarios",
    )
    phone_event_parser.add_argument("--profile", type=Path, required=True)
    phone_event_parser.add_argument("--models-dir", type=Path, default=Path("models"))
    phone_event_parser.add_argument(
        "--role", choices=("tuning", "holdout"), required=True
    )
    phone_event_parser.add_argument(
        "--suite",
        choices=("full", "positive"),
        default="full",
        help="full safety chain or focused phone-positive and recovery retry",
    )
    phone_event_parser.add_argument(
        "--output-root", type=Path, default=Path("data/validation")
    )
    phone_event_parser.add_argument(
        "--detector-backend", choices=("efficientdet", "yolo"), default="efficientdet"
    )
    phone_event_parser.add_argument("--yolo-model", type=Path)
    phone_event_parser.add_argument("--phone-score-threshold", type=float, default=0.25)
    phone_event_parser.add_argument("--phone-retain-seconds", type=float, default=2.0)
    phone_event_parser.add_argument("--phone-center-distance", type=float, default=0.18)
    purge_parser = subparsers.add_parser(
        "evidence-purge", help="dry-run or delete expired Stage C error evidence"
    )
    purge_parser.add_argument("index", type=Path)
    purge_parser.add_argument(
        "--apply", action="store_true", help="delete listed expired image files"
    )
    from observer.product_cli import register_parser
    register_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "product":
        from observer.product_cli import dispatch
        return dispatch(args)
    if args.command == "doctor":
        return _doctor(args.config)
    if args.command == "yolo-doctor":
        return _yolo_doctor(args.model)
    if args.command == "yolo-replay":
        return _yolo_replay(
            args.video,
            args.model,
            args.output,
            args.score_threshold,
            args.iou_threshold,
            args.input_size,
            args.sample_frames,
        )
    if args.command == "camera-smoke":
        return _camera_smoke(args.config, args.seconds)
    if args.command == "vision-smoke":
        return _vision_smoke(
            args.config,
            args.models_dir,
            args.seconds,
            args.profile,
            args.evidence_dir,
            args.report,
            args.debug_video,
            args.guided_debug_cues,
            args.guided_debug_plan,
            args.detector_backend,
            args.yolo_model,
        )
    if args.command == "calibrate":
        return _calibrate(args.config, args.models_dir, args.output)
    if args.command == "align":
        return _align(
            args.config,
            args.models_dir,
            args.profile,
            args.stable_seconds,
            args.timeout_seconds,
            args.verify_seconds,
            args.require_posture,
        )
    if args.command == "profile-check":
        return _profile_check(args.profile)
    if args.command == "validation-report":
        return _validation_report(args.trials, args.output)
    if args.command == "validation-scenarios":
        return _validation_scenarios()
    if args.command == "validate-live":
        return _validate_live(
            args.config,
            args.models_dir,
            args.profile,
            args.role,
            args.signals,
            args.scenarios,
            args.output_root,
            args.repetitions,
            args.save_error_evidence,
            args.debug_trial_videos,
            args.phone_score_threshold,
            args.phone_retain_seconds,
            args.phone_center_distance,
            args.detector_backend,
            args.yolo_model,
        )
    if args.command == "validate-away-events":
        return _validate_away_events(
            args.config,
            args.models_dir,
            args.profile,
            args.role,
            args.output_root,
            args.suite,
            args.detector_backend,
            args.yolo_model,
        )
    if args.command == "validate-phone-events":
        return _validate_phone_events(
            args.config,
            args.models_dir,
            args.profile,
            args.role,
            args.output_root,
            args.suite,
            args.detector_backend,
            args.yolo_model,
            args.phone_score_threshold,
            args.phone_retain_seconds,
            args.phone_center_distance,
        )
    if args.command == "evidence-purge":
        return _evidence_purge(args.index, args.apply)
    raise AssertionError(f"unknown command: {args.command}")
