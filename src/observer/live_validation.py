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
    ATTENTION_GAZE_DOWN_BOUNDARY,
    ATTENTION_HEAD_PITCH_STRONG_DOWN_DELTA_THRESHOLD,
    ATTENTION_HEAD_YAW_DELTA_THRESHOLD,
    attention_down_evidence_matches,
    load_profile,
)
from observer.calibration_workflow import current_environment
from observer.camera import CameraWorker
from observer.config import AppConfig
from observer.models import ModelFiles, sha256_file
from observer.object_detection import YoloOnnxBackend
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
    distance_mode: str | None = None


SCENARIOS: tuple[Scenario, ...] = (
    Scenario('presence_yolo_01_seated', 'away', SignalState.COMPLIANT, 'Sit naturally in view', duration_seconds=8.0, repetitions=1),
    Scenario('presence_yolo_02_face_hidden', 'away', SignalState.COMPLIANT, 'Hide face with hands; keep upper body visible; do not cover camera', duration_seconds=8.0, repetitions=1),
    Scenario('presence_yolo_03_empty', 'away', SignalState.VIOLATION, 'Leave camera view completely; keep lens uncovered', duration_seconds=8.0, repetitions=1),
    Scenario('presence_yolo_04_return', 'away', SignalState.COMPLIANT, 'Return and sit naturally in view', duration_seconds=8.0, repetitions=1),
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
    Scenario("attention_bottom_center", "attention_away", SignalState.COMPLIANT, "Look at the lower screen dot without lowering your head toward the desk", target=(0.50, 0.92), duration_seconds=6.0, repetitions=1),
    Scenario("attention_top_left", "attention_away", SignalState.COMPLIANT, "Look at the dot", target=(0.08, 0.08), repetitions=3),
    Scenario("attention_top_right", "attention_away", SignalState.COMPLIANT, "Look at the dot", target=(0.92, 0.08), repetitions=3),
    Scenario("attention_bottom_left", "attention_away", SignalState.COMPLIANT, "Look at the dot", target=(0.08, 0.92), repetitions=3),
    Scenario("attention_bottom_right", "attention_away", SignalState.COMPLIANT, "Look at the dot", target=(0.92, 0.92), repetitions=3),
    Scenario("attention_outside_left", "attention_away", SignalState.VIOLATION, "Look beyond the left screen edge"),
    Scenario("attention_outside_right", "attention_away", SignalState.VIOLATION, "Look beyond the right screen edge"),
    Scenario("attention_desk", "attention_away", SignalState.VIOLATION, "Look down at the desk"),
    Scenario("attention_desk_control_side", "attention_away", SignalState.COMPLIANT, "Turn clearly to either side but keep your gaze level", duration_seconds=6.0, repetitions=1),
    Scenario("attention_desk_control_left", "attention_away", SignalState.COMPLIANT, "Turn clearly left and keep your gaze level; this must not count as looking down", duration_seconds=6.0, repetitions=1),
    Scenario("attention_desk_control_right", "attention_away", SignalState.COMPLIANT, "Turn clearly right and keep your gaze level; this must not count as looking down", duration_seconds=6.0, repetitions=1),
    Scenario("attention_desk_control_chin", "attention_away", SignalState.COMPLIANT, "Lower your chin slightly but keep looking at the screen", duration_seconds=6.0, repetitions=1),
    Scenario("attention_eyes_covered", "attention_away", SignalState.UNKNOWN, "Cover or close both eyes", repetitions=3),
    Scenario("attention_boundary_a01_center", "attention_away", SignalState.COMPLIANT, "A01: natural distance; look at screen center", target=(0.50, 0.50), duration_seconds=10.0, repetitions=1),
    Scenario("attention_boundary_a02_outside_left", "attention_away", SignalState.VIOLATION, "A02: natural distance; look at the fixed outside LEFT-LOW point", duration_seconds=12.0, repetitions=1),
    Scenario("attention_boundary_a03_recovery", "attention_away", SignalState.COMPLIANT, "A03: return to natural distance and screen center", target=(0.50, 0.50), duration_seconds=8.0, repetitions=1),
    Scenario("attention_boundary_a04_outside_right", "attention_away", SignalState.VIOLATION, "A04: natural distance; look at the fixed outside RIGHT-LOW point", duration_seconds=12.0, repetitions=1),
    Scenario("attention_boundary_a05_recovery", "attention_away", SignalState.COMPLIANT, "A05: return to natural distance and screen center", target=(0.50, 0.50), duration_seconds=8.0, repetitions=1),
    Scenario("attention_boundary_a06_near_left", "attention_away", SignalState.COMPLIANT, "A06: move close; look at the fixed INSIDE left-low point", target=(0.08, 0.92), duration_seconds=12.0, repetitions=1, distance_mode="near"),
    Scenario("attention_boundary_a07_near_right", "attention_away", SignalState.COMPLIANT, "A07: stay close; look at the fixed INSIDE right-low point", target=(0.92, 0.92), duration_seconds=12.0, repetitions=1, distance_mode="near"),
    Scenario("attention_boundary_a08_far_center", "attention_away", SignalState.COMPLIANT, "A08: calibration distance; look at screen center", target=(0.50, 0.50), duration_seconds=10.0, repetitions=1, distance_mode="calibration"),
    Scenario("attention_boundary_a09_far_left", "attention_away", SignalState.COMPLIANT, "A09: stay at calibration distance; look INSIDE left-low", target=(0.08, 0.92), duration_seconds=12.0, repetitions=1, distance_mode="calibration"),
    Scenario("attention_boundary_a10_far_right", "attention_away", SignalState.COMPLIANT, "A10: stay at calibration distance; look INSIDE right-low", target=(0.92, 0.92), duration_seconds=12.0, repetitions=1, distance_mode="calibration"),
    Scenario("attention_boundary_a11_final", "attention_away", SignalState.COMPLIANT, "A11: natural distance; final screen center", target=(0.50, 0.50), duration_seconds=10.0, repetitions=1),
    Scenario("attention_compensation_d01_center", "attention_away", SignalState.COMPLIANT, "D01: natural distance; face forward and look at center dot", target=(0.50, 0.50), duration_seconds=8.0, repetitions=1),
    Scenario("attention_compensation_d02_head_left_eyes_center", "attention_away", SignalState.COMPLIANT, "D02: turn head left moderately; keep EYES on center dot", target=(0.50, 0.50), duration_seconds=8.0, repetitions=1),
    Scenario("attention_compensation_d03_outside_left", "attention_away", SignalState.VIOLATION, "D03: turn HEAD + EYES left together; keep gaze BEYOND left edge; do NOT look back at center", duration_seconds=10.0, repetitions=1),
    Scenario("attention_compensation_d04_recovery", "attention_away", SignalState.COMPLIANT, "D04: face forward and look at center dot", target=(0.50, 0.50), duration_seconds=8.0, repetitions=1),
    Scenario("attention_compensation_d05_head_right_eyes_center", "attention_away", SignalState.COMPLIANT, "D05: turn head right moderately; keep EYES on center dot", target=(0.50, 0.50), duration_seconds=8.0, repetitions=1),
    Scenario("attention_compensation_d06_outside_right", "attention_away", SignalState.VIOLATION, "D06: turn HEAD + EYES right together; keep gaze BEYOND right edge; do NOT look back at center", duration_seconds=10.0, repetitions=1),
    Scenario("attention_compensation_d07_recovery", "attention_away", SignalState.COMPLIANT, "D07: face forward and look at center dot", target=(0.50, 0.50), duration_seconds=8.0, repetitions=1),
    Scenario("attention_compensation_d08_near_center", "attention_away", SignalState.COMPLIANT, "D08: move close; look at center dot", target=(0.50, 0.50), duration_seconds=8.0, repetitions=1, distance_mode="near"),
    Scenario("attention_compensation_d09_near_left", "attention_away", SignalState.COMPLIANT, "D09: stay close; look at INSIDE left-low dot", target=(0.08, 0.92), duration_seconds=8.0, repetitions=1, distance_mode="near"),
    Scenario("attention_compensation_d10_near_right", "attention_away", SignalState.COMPLIANT, "D10: stay close; look at INSIDE right-low dot", target=(0.92, 0.92), duration_seconds=8.0, repetitions=1, distance_mode="near"),
    Scenario("attention_compensation_d11_final", "attention_away", SignalState.COMPLIANT, "D11: return to natural distance and center dot", target=(0.50, 0.50), duration_seconds=8.0, repetitions=1),
    Scenario("phone_baseline_b01_natural_no_phone", "phone_held", SignalState.COMPLIANT, "B01: work naturally; no phone visible", duration_seconds=10.0, repetitions=1, accepted_states=(SignalState.COMPLIANT, SignalState.UNKNOWN)),
    Scenario("phone_baseline_b02_desktop_phone", "phone_held", SignalState.VIOLATION, "B02: phone visible ANYWHERE in camera view", duration_seconds=12.0, repetitions=1),
    Scenario("phone_baseline_b03_non_phone_object", "phone_held", SignalState.COMPLIANT, "B03: hold a remote or calculator; it is not a phone", duration_seconds=12.0, repetitions=1, accepted_states=(SignalState.COMPLIANT, SignalState.UNKNOWN)),
    Scenario("phone_baseline_b04_one_hand_phone", "phone_held", SignalState.VIOLATION, "B04: hold and use the phone naturally with ONE hand", duration_seconds=12.0, repetitions=1),
    Scenario("phone_baseline_b05_recovery", "phone_held", SignalState.COMPLIANT, "B05: put the phone out of view; work and look at screen", duration_seconds=8.0, repetitions=1, accepted_states=(SignalState.COMPLIANT, SignalState.UNKNOWN)),
    Scenario("phone_baseline_b06_two_hands_low", "phone_held", SignalState.VIOLATION, "B06: use phone with BOTH hands; low position is allowed", duration_seconds=12.0, repetitions=1),
    Scenario("phone_baseline_b07_recovery", "phone_held", SignalState.COMPLIANT, "B07: put the phone out of view; work and look at screen", duration_seconds=8.0, repetitions=1, accepted_states=(SignalState.COMPLIANT, SignalState.UNKNOWN)),
    Scenario("phone_baseline_b08_phone_out_of_frame", "phone_held", SignalState.COMPLIANT, "B08 PHONE: hold phone outside camera frame; no phone is visible", duration_seconds=12.0, repetitions=1, accepted_states=(SignalState.COMPLIANT, SignalState.UNKNOWN)),
    Scenario("attention_baseline_b08_look_off_screen", "attention_away", SignalState.VIOLATION, "B08 ATTENTION: repeat hold; keep eyes on the phone off screen", duration_seconds=12.0, repetitions=1),
    Scenario("phone_baseline_b09_final", "phone_held", SignalState.COMPLIANT, "B09: final natural work; no phone visible", duration_seconds=10.0, repetitions=1, accepted_states=(SignalState.COMPLIANT, SignalState.UNKNOWN)),
    Scenario("phone_baseline_b10_one_hand_low", "phone_held", SignalState.VIOLATION, "B10: hold and use the phone with ONE hand in a LOW position; keep part of it visible", duration_seconds=12.0, repetitions=1),
    Scenario("phone_baseline_b10_recovery", "phone_held", SignalState.COMPLIANT, "B10 RECOVERY: put the phone out of view; work and look at screen", duration_seconds=8.0, repetitions=1, accepted_states=(SignalState.COMPLIANT, SignalState.UNKNOWN)),
    Scenario("phone_baseline_b11_two_hands_normal", "phone_held", SignalState.VIOLATION, "B11: hold and use the phone with BOTH hands at a NORMAL position", duration_seconds=12.0, repetitions=1),
    Scenario("phone_baseline_b11_recovery", "phone_held", SignalState.COMPLIANT, "B11 RECOVERY: put the phone out of view; work and look at screen", duration_seconds=8.0, repetitions=1, accepted_states=(SignalState.COMPLIANT, SignalState.UNKNOWN)),
    Scenario("phone_baseline_b12_one_hand_edge", "phone_held", SignalState.VIOLATION, "B12: hold the phone with ONE hand VERY LOW; keep only a small edge visible", duration_seconds=12.0, repetitions=1),
    Scenario("phone_baseline_b12_recovery", "phone_held", SignalState.COMPLIANT, "B12 RECOVERY: put the phone out of view; work and look at screen", duration_seconds=8.0, repetitions=1, accepted_states=(SignalState.COMPLIANT, SignalState.UNKNOWN)),
    Scenario("phone_baseline_b13_two_hands_edge", "phone_held", SignalState.VIOLATION, "B13: hold the phone with BOTH hands VERY LOW; keep only a small edge visible", duration_seconds=12.0, repetitions=1),
    Scenario("phone_baseline_b13_recovery", "phone_held", SignalState.COMPLIANT, "B13 RECOVERY: put the phone out of view; work and look at screen", duration_seconds=8.0, repetitions=1, accepted_states=(SignalState.COMPLIANT, SignalState.UNKNOWN)),
    # Independent phone-edge diagnostics. These are deliberately separate from
    # the B01-B13 baselines so parameter experiments cannot silently change the
    # original acceptance set or its historical evidence.
    Scenario("phone_edge_e01_bottom_center", "phone_held", SignalState.VIOLATION, "E01: keep the face fixed; hold the phone at the bottom-center edge with only a small strip visible", duration_seconds=12.0, repetitions=1),
    Scenario("phone_edge_e01_bottom_center_recovery", "phone_held", SignalState.COMPLIANT, "E01 RECOVERY: remove the phone completely from view; keep the face fixed", duration_seconds=8.0, repetitions=1, accepted_states=(SignalState.COMPLIANT, SignalState.UNKNOWN)),
    Scenario("phone_edge_e02_bottom_left", "phone_held", SignalState.VIOLATION, "E02: keep the face fixed; hold the phone at the bottom-left edge with the same small strip visible as E01", duration_seconds=12.0, repetitions=1),
    Scenario("phone_edge_e02_bottom_left_recovery", "phone_held", SignalState.COMPLIANT, "E02 RECOVERY: remove the phone completely from view; keep the face fixed", duration_seconds=8.0, repetitions=1, accepted_states=(SignalState.COMPLIANT, SignalState.UNKNOWN)),
    Scenario("phone_edge_e03_bottom_right", "phone_held", SignalState.VIOLATION, "E03: keep the face fixed; hold the phone at the bottom-right edge with the same small strip visible as E01", duration_seconds=12.0, repetitions=1),
    Scenario("phone_edge_e03_bottom_right_recovery", "phone_held", SignalState.COMPLIANT, "E03 RECOVERY: remove the phone completely from view; keep the face fixed", duration_seconds=8.0, repetitions=1, accepted_states=(SignalState.COMPLIANT, SignalState.UNKNOWN)),
    Scenario("phone_edge_e04_top_edge", "phone_held", SignalState.VIOLATION, "E04: keep the face fixed; hold the phone at the top screen edge with only a small strip visible", duration_seconds=12.0, repetitions=1),
    Scenario("phone_edge_e04_top_edge_recovery", "phone_held", SignalState.COMPLIANT, "E04 RECOVERY: remove the phone completely from view; keep the face fixed", duration_seconds=8.0, repetitions=1, accepted_states=(SignalState.COMPLIANT, SignalState.UNKNOWN)),
    # Paired phone/power-bank contrast collection. Each negative is recorded
    # before its matching phone positive so phone retention cannot contaminate
    # the start of the negative trial. Keep both objects at the same position,
    # orientation and visible fraction; this tests object identity rather than
    # allowing the detector to learn the background or location.
    Scenario("phone_contrast_f00_clear", "phone_held", SignalState.COMPLIANT, "F00: empty hands; keep phone and power bank fully out of view", duration_seconds=5.0, repetitions=1, accepted_states=(SignalState.COMPLIANT, SignalState.UNKNOWN)),
    Scenario("phone_contrast_f01_powerbank_center", "phone_held", SignalState.COMPLIANT, "F01: hold the POWER BANK at center; show at least one third; use one hand", duration_seconds=10.0, repetitions=1, accepted_states=(SignalState.COMPLIANT, SignalState.UNKNOWN)),
    Scenario("phone_contrast_f02_phone_center", "phone_held", SignalState.VIOLATION, "F02: replace it with the PHONE at the SAME center position, angle and visible fraction", duration_seconds=10.0, repetitions=1),
    Scenario("phone_contrast_f03_clear", "phone_held", SignalState.COMPLIANT, "F03: remove both objects completely; keep face and body position fixed", duration_seconds=5.0, repetitions=1, accepted_states=(SignalState.COMPLIANT, SignalState.UNKNOWN)),
    Scenario("phone_contrast_f04_powerbank_left", "phone_held", SignalState.COMPLIANT, "F04: hold the POWER BANK at lower-left; show at least one third; use one hand", duration_seconds=10.0, repetitions=1, accepted_states=(SignalState.COMPLIANT, SignalState.UNKNOWN)),
    Scenario("phone_contrast_f05_phone_left", "phone_held", SignalState.VIOLATION, "F05: replace it with the PHONE at the SAME lower-left position, angle and visible fraction", duration_seconds=10.0, repetitions=1),
    Scenario("phone_contrast_f06_clear", "phone_held", SignalState.COMPLIANT, "F06: remove both objects completely; keep face and body position fixed", duration_seconds=5.0, repetitions=1, accepted_states=(SignalState.COMPLIANT, SignalState.UNKNOWN)),
    Scenario("phone_contrast_f07_powerbank_right", "phone_held", SignalState.COMPLIANT, "F07: hold the POWER BANK at lower-right; show at least one third; use one hand", duration_seconds=10.0, repetitions=1, accepted_states=(SignalState.COMPLIANT, SignalState.UNKNOWN)),
    Scenario("phone_contrast_f08_phone_right", "phone_held", SignalState.VIOLATION, "F08: replace it with the PHONE at the SAME lower-right position, angle and visible fraction", duration_seconds=10.0, repetitions=1),
    Scenario("phone_contrast_f09_final_clear", "phone_held", SignalState.COMPLIANT, "F09: final empty-hands recovery; remove both objects completely", duration_seconds=5.0, repetitions=1, accepted_states=(SignalState.COMPLIANT, SignalState.UNKNOWN)),
    Scenario("posture_ergonomics_c01_natural", "posture_deviation", SignalState.COMPLIANT, "C01: sit in your natural comfortable posture", duration_seconds=15.0, repetitions=1),
    Scenario("posture_ergonomics_c02_near_natural", "posture_deviation", SignalState.COMPLIANT, "C02: move closer naturally; do NOT deliberately slouch", duration_seconds=15.0, repetitions=1),
    Scenario("posture_ergonomics_c03_forward_slouch", "posture_deviation", SignalState.VIOLATION, "C03: lean/slouch clearly forward; stay safe and comfortable", duration_seconds=12.0, repetitions=1),
    Scenario("posture_ergonomics_c04_recovery", "posture_deviation", SignalState.COMPLIANT, "C04: return to your natural comfortable posture", duration_seconds=8.0, repetitions=1),
    Scenario("posture_ergonomics_c05_side_or_tilt", "posture_deviation", SignalState.VIOLATION, "C05: lean or tilt clearly to one side; stay safe", duration_seconds=12.0, repetitions=1),
    Scenario("posture_ergonomics_c06_shoulder_unknown", "posture_deviation", SignalState.UNKNOWN, "C06 POSTURE: keep one shoulder partly outside frame", duration_seconds=12.0, repetitions=1),
    Scenario("attention_ergonomics_c06_attention", "attention_away", SignalState.COMPLIANT, "C06 ATTENTION: repeat hold; keep eyes on screen center", target=(0.50, 0.50), duration_seconds=12.0, repetitions=1),
    Scenario("attention_ergonomics_c07_small_chin", "attention_away", SignalState.COMPLIANT, "C07: lower chin slightly; keep eyes on screen center", target=(0.50, 0.50), duration_seconds=12.0, repetitions=1),
    Scenario("posture_ergonomics_c08_final", "posture_deviation", SignalState.COMPLIANT, "C08: return to your natural comfortable posture", duration_seconds=10.0, repetitions=1),
)


# Parameter profiles for the independent phone-edge diagnostic matrix. The
# production baseline is included only as a control; none of these profiles
# changes the defaults used by the existing B01-B13 scenarios.
PHONE_EDGE_PARAMETER_PROFILES: tuple[dict[str, float | str], ...] = (
    {
        "profile_id": "P0_control",
        "phone_score_threshold": 0.25,
        "phone_retain_seconds": 1.5,
        "phone_center_distance": 0.18,
    },
    {
        "profile_id": "P1_medium_recall",
        "phone_score_threshold": 0.15,
        "phone_retain_seconds": 1.5,
        "phone_center_distance": 0.18,
    },
    {
        "profile_id": "P2_low_threshold_continuity",
        "phone_score_threshold": 0.10,
        "phone_retain_seconds": 2.2,
        "phone_center_distance": 0.18,
    },
    {
        "profile_id": "P3_low_threshold_wide_tracking",
        "phone_score_threshold": 0.10,
        "phone_retain_seconds": 2.2,
        "phone_center_distance": 0.30,
    },
    {
        "profile_id": "P4_medium_threshold_wide_tracking",
        "phone_score_threshold": 0.15,
        "phone_retain_seconds": 2.2,
        "phone_center_distance": 0.30,
    },
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
ATTENTION_CHIN_CONTROL_MIN_DELTA = 0.03


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


def scenario_acceptance_summary(scenario: Scenario) -> str:
    """Return the short, user-visible baseline criterion for one scenario."""
    if scenario.scenario_id.startswith("phone_contrast_"):
        if "_phone_" in scenario.scenario_id:
            return "DIAGNOSTIC: violation >=90%; continuous >=8s"
        return "DIAGNOSTIC: violation <=10%; violation run <2s"
    if scenario.scenario_id.startswith("phone_edge_"):
        if scenario.scenario_id.endswith("_recovery"):
            return "DIAGNOSTIC: violation <=10%; violation run <2.5s"
        return "DIAGNOSTIC: violation >=90%; continuous >=10s"
    if scenario.scenario_id.startswith("phone_baseline_"):
        if scenario.scenario_id in {
            "phone_baseline_b01_natural_no_phone",
            "phone_baseline_b03_non_phone_object",
            "phone_baseline_b09_final",
            "phone_baseline_b08_phone_out_of_frame",
        }:
            return "STANDARD: no visible-phone violation; error run <2s"
        if scenario.scenario_id in {
            "phone_baseline_b05_recovery",
            "phone_baseline_b07_recovery",
            "phone_baseline_b10_recovery",
            "phone_baseline_b11_recovery",
            "phone_baseline_b12_recovery",
            "phone_baseline_b13_recovery",
        }:
            return "STANDARD: violation <=10%; violation run <2.5s"
        return "STANDARD: violation >=90%; continuous >=10s"
    if scenario.scenario_id.startswith("posture_ergonomics_c"):
        if scenario.scenario_id in {
            "posture_ergonomics_c03_forward_slouch",
            "posture_ergonomics_c05_side_or_tilt",
        }:
            return "STANDARD: violation >=90%"
        if scenario.scenario_id == "posture_ergonomics_c06_shoulder_unknown":
            return "STANDARD: unknown >=90%; no false posture violation"
        if scenario.scenario_id == "posture_ergonomics_c04_recovery":
            return "STANDARD: compliant >=90%; unknown <=10%; no 2.5s error"
        return "STANDARD: compliant >=90%; unknown <=10%; no 2s error"
    if scenario.scenario_id == "attention_baseline_b08_look_off_screen":
        return "STANDARD: violation >=90%; continuous >=8s"
    if scenario.scenario_id.startswith("attention_ergonomics_c"):
        return "STANDARD: violation <=10%; unknown <=10%; no 8s event"
    if scenario.scenario_id in {
        "attention_boundary_a02_outside_left",
        "attention_boundary_a04_outside_right",
        "attention_compensation_d03_outside_left",
        "attention_compensation_d06_outside_right",
    }:
        return "STANDARD: violation >=90%; continuous >=8s"
    if scenario.scenario_id in {
        "attention_boundary_a03_recovery",
        "attention_boundary_a05_recovery",
        "attention_compensation_d04_recovery",
        "attention_compensation_d07_recovery",
    }:
        return "STANDARD: compliant >=90%; recover <=2.5s"
    if scenario.scenario_id.startswith(("attention_boundary_", "attention_compensation_")):
        return "STANDARD: violation <=10%; unknown <=10%; no 8s event"
    return ""


def scenario_record_passes(scenario: Scenario, record: TrialRecord) -> bool:
    """Apply the same strict baseline standard shown in the interactive UI."""

    if not record.valid:
        return False
    metrics = record.metrics
    if scenario.scenario_id.startswith("phone_contrast_"):
        violation_fraction = metrics.get("violation_fraction", 1.0)
        violation_run = metrics.get("longest_violation_run_seconds", 999.0)
        if "_phone_" in scenario.scenario_id:
            return violation_fraction >= 0.90 and violation_run >= 8.0
        return violation_fraction <= 0.10 and violation_run < 2.0
    if scenario.scenario_id.startswith("phone_edge_"):
        violation_fraction = metrics.get("violation_fraction", 1.0)
        violation_run = metrics.get("longest_violation_run_seconds", 999.0)
        if scenario.scenario_id.endswith("_recovery"):
            return violation_fraction <= 0.10 and violation_run < 2.5
        return violation_fraction >= 0.90 and violation_run >= 10.0
    if scenario.scenario_id.startswith("phone_baseline_"):
        violation_fraction = metrics.get("violation_fraction", 1.0)
        violation_run = metrics.get("longest_violation_run_seconds", 999.0)
        if scenario.scenario_id in {
            "phone_baseline_b05_recovery",
            "phone_baseline_b07_recovery",
            "phone_baseline_b10_recovery",
            "phone_baseline_b11_recovery",
            "phone_baseline_b12_recovery",
            "phone_baseline_b13_recovery",
        }:
            return violation_fraction <= 0.10 and violation_run < 2.5
        if scenario.scenario_id in {
            "phone_baseline_b02_desktop_phone",
            "phone_baseline_b06_two_hands_low",
            "phone_baseline_b04_one_hand_phone",
            "phone_baseline_b10_one_hand_low",
            "phone_baseline_b11_two_hands_normal",
            "phone_baseline_b12_one_hand_edge",
            "phone_baseline_b13_two_hands_edge",
        }:
            return violation_fraction >= 0.90 and violation_run >= 10.0
        return violation_fraction <= 0.10 and violation_run < 2.0
    if scenario.scenario_id.startswith("posture_ergonomics_c"):
        compliant_fraction = metrics.get("compliant_fraction", 0.0)
        violation_fraction = metrics.get("violation_fraction", 1.0)
        unknown_fraction = metrics.get("unknown_fraction", 1.0)
        violation_run = metrics.get("longest_violation_run_seconds", 999.0)
        if scenario.scenario_id in {
            "posture_ergonomics_c03_forward_slouch",
            "posture_ergonomics_c05_side_or_tilt",
        }:
            return violation_fraction >= 0.90
        if scenario.scenario_id == "posture_ergonomics_c06_shoulder_unknown":
            return unknown_fraction >= 0.90 and violation_fraction <= 0.10
        maximum_error_run = 2.5 if scenario.scenario_id.endswith("recovery") else 2.0
        return (
            compliant_fraction >= 0.90
            and unknown_fraction <= 0.10
            and violation_run < maximum_error_run
        )
    if scenario.scenario_id == "attention_baseline_b08_look_off_screen":
        return (
            metrics.get("violation_fraction", 0.0) >= 0.90
            and metrics.get("longest_violation_run_seconds", 0.0) >= 8.0
        )
    if scenario.scenario_id.startswith("attention_ergonomics_c"):
        return (
            metrics.get("violation_fraction", 1.0) <= 0.10
            and metrics.get("unknown_fraction", 1.0) <= 0.10
            and metrics.get("longest_violation_run_seconds", 8.0) < 8.0
        )
    if scenario.scenario_id in {
        "attention_boundary_a02_outside_left",
        "attention_boundary_a04_outside_right",
        "attention_compensation_d03_outside_left",
        "attention_compensation_d06_outside_right",
    }:
        return (
            metrics.get("violation_fraction", 0.0) >= 0.90
            and metrics.get("longest_violation_run_seconds", 0.0) >= 8.0
        )
    if scenario.scenario_id in {
        "attention_boundary_a03_recovery",
        "attention_boundary_a05_recovery",
        "attention_compensation_d04_recovery",
        "attention_compensation_d07_recovery",
    }:
        return (
            metrics.get("compliant_fraction", 0.0) >= 0.90
            and metrics.get("unknown_fraction", 1.0) <= 0.10
        )
    if scenario.scenario_id.startswith(
        ("attention_boundary_", "attention_compensation_")
    ):
        return (
            metrics.get("violation_fraction", 1.0) <= 0.10
            and metrics.get("unknown_fraction", 1.0) <= 0.10
            and metrics.get("longest_violation_run_seconds", 8.0) < 8.0
        )
    return record.predicted_state in {
        state.value for state in scenario_accepted_states(scenario)
    }


_DIRECTIONAL_ATTENTION_SCENARIOS = {
    "attention_outside_left",
    "attention_outside_right",
    "attention_desk",
    "attention_desk_control_side",
    "attention_desk_control_left",
    "attention_desk_control_right",
    "attention_desk_control_chin",
    "attention_boundary_a02_outside_left",
    "attention_boundary_a04_outside_right",
    "attention_compensation_d03_outside_left",
    "attention_compensation_d06_outside_right",
}

_ATTENTION_ACTION_GATED_SCENARIOS = {
    "attention_desk_control_side",
    "attention_desk_control_left",
    "attention_desk_control_right",
    "attention_desk_control_chin",
}

# Retained only so historical tuning batches remain replayable. These prompts
# proved ambiguous or redundant and must not enter the default protocol or any
# holdout acceptance batch.
_TUNING_DIAGNOSTIC_ONLY_SCENARIOS = {
    "attention_desk_control_side",
    "attention_desk_control_chin",
    "attention_boundary_a01_center",
    "attention_boundary_a02_outside_left",
    "attention_boundary_a03_recovery",
    "attention_boundary_a04_outside_right",
    "attention_boundary_a05_recovery",
    "attention_boundary_a06_near_left",
    "attention_boundary_a07_near_right",
    "attention_boundary_a08_far_center",
    "attention_boundary_a09_far_left",
    "attention_boundary_a10_far_right",
    "attention_boundary_a11_final",
    "attention_compensation_d01_center",
    "attention_compensation_d02_head_left_eyes_center",
    "attention_compensation_d03_outside_left",
    "attention_compensation_d04_recovery",
    "attention_compensation_d05_head_right_eyes_center",
    "attention_compensation_d06_outside_right",
    "attention_compensation_d07_recovery",
    "attention_compensation_d08_near_center",
    "attention_compensation_d09_near_left",
    "attention_compensation_d10_near_right",
    "attention_compensation_d11_final",
    "phone_baseline_b10_one_hand_low",
    "phone_baseline_b10_recovery",
    "phone_baseline_b11_two_hands_normal",
    "phone_baseline_b11_recovery",
    "phone_baseline_b12_one_hand_edge",
    "phone_baseline_b12_recovery",
    "phone_baseline_b13_two_hands_edge",
    "phone_baseline_b13_recovery",
    "phone_edge_e01_bottom_center",
    "phone_edge_e01_bottom_center_recovery",
    "phone_edge_e02_bottom_left",
    "phone_edge_e02_bottom_left_recovery",
    "phone_edge_e03_bottom_right",
    "phone_edge_e03_bottom_right_recovery",
    "phone_edge_e04_top_edge",
    "phone_edge_e04_top_edge_recovery",
    "phone_contrast_f00_clear",
    "phone_contrast_f01_powerbank_center",
    "phone_contrast_f02_phone_center",
    "phone_contrast_f03_clear",
    "phone_contrast_f04_powerbank_left",
    "phone_contrast_f05_phone_left",
    "phone_contrast_f06_clear",
    "phone_contrast_f07_powerbank_right",
    "phone_contrast_f08_phone_right",
    "phone_contrast_f09_final_clear",
}


def attention_control_action_matches(
    scenario: Scenario, signal: SignalValue
) -> bool:
    """Confirm that a negative-control action was actually performed."""

    return attention_control_action_blocker(scenario, signal) is None


def attention_control_action_blocker(
    scenario: Scenario, signal: SignalValue
) -> str | None:
    """Return a user-facing reason when a control action has not formed."""

    metrics = signal.metrics
    yaw = metrics.get("attention_head_yaw_delta")
    pitch = metrics.get("attention_head_pitch_delta")
    screen_y = metrics.get("attention_screen_y")
    if scenario.scenario_id == "attention_desk_control_left":
        if yaw is None:
            return "yaw unavailable"
        if yaw < ATTENTION_HEAD_YAW_DELTA_THRESHOLD:
            return f"turn farther left (yaw={yaw:+.2f})"
        return None
    if scenario.scenario_id == "attention_desk_control_right":
        if yaw is None:
            return "yaw unavailable"
        if yaw > -ATTENTION_HEAD_YAW_DELTA_THRESHOLD:
            return f"turn farther right (yaw={yaw:+.2f})"
        return None
    if scenario.scenario_id == "attention_desk_control_side":
        if yaw is None:
            return "yaw unavailable"
        if abs(yaw) < ATTENTION_HEAD_YAW_DELTA_THRESHOLD:
            return f"turn farther to one side (yaw={yaw:+.2f})"
        return None
    if scenario.scenario_id == "attention_desk_control_chin":
        if yaw is None or pitch is None or screen_y is None:
            return "direction evidence unavailable"
        if abs(yaw) >= ATTENTION_HEAD_YAW_DELTA_THRESHOLD:
            return f"face the screen more directly (yaw={yaw:+.2f})"
        if pitch < ATTENTION_CHIN_CONTROL_MIN_DELTA:
            return f"chin change too small (pitch={pitch:+.2f})"
        if pitch >= ATTENTION_HEAD_PITCH_STRONG_DOWN_DELTA_THRESHOLD:
            return f"chin is in strong-down range (pitch={pitch:+.2f})"
        if screen_y > ATTENTION_GAZE_DOWN_BOUNDARY:
            return f"gaze predicts below screen (y={screen_y:+.2f})"
        return None
    return None


def attention_direction_matches(scenario: Scenario, signal: SignalValue) -> bool:
    """Require the requested direction, not merely any attention violation."""

    metrics = signal.metrics
    yaw = metrics.get("attention_head_yaw_delta")
    pitch = metrics.get("attention_head_pitch_delta")
    screen_x = metrics.get("attention_screen_x")
    screen_y = metrics.get("attention_screen_y")
    if scenario.scenario_id.endswith("outside_left"):
        return (yaw is not None and yaw >= ATTENTION_HEAD_YAW_DELTA_THRESHOLD) or (
            screen_x is not None
            and screen_x < -ATTENTION_DIRECTION_SCREEN_MARGIN
        )
    if scenario.scenario_id.endswith("outside_right"):
        return (yaw is not None and yaw <= -ATTENTION_HEAD_YAW_DELTA_THRESHOLD) or (
            screen_x is not None
            and screen_x > 1.0 + ATTENTION_DIRECTION_SCREEN_MARGIN
        )
    if scenario.scenario_id.startswith("attention_desk"):
        return (
            yaw is not None
            and pitch is not None
            and attention_down_evidence_matches(
                yaw_delta=yaw,
                pitch_delta=pitch,
                screen_y=screen_y,
            )
        )
    return signal.state in scenario_accepted_states(scenario)


def scenario_validation_signal(scenario: Scenario, signal: SignalValue) -> SignalValue:
    """Score the production signal unchanged; action checks are separate metadata."""
    return signal


def attention_action_status(scenario: Scenario, signal: SignalValue) -> str:
    if scenario.scenario_id == "attention_bottom_center":
        return "ACTION: USER CONFIRM - look at the lower dot before pressing SPACE/ENTER"
    if scenario.scenario_id not in _ATTENTION_ACTION_GATED_SCENARIOS:
        return ""
    blocker = attention_control_action_blocker(scenario, signal)
    return "ACTION: MATCHED" if blocker is None else f"ACTION: NOT FORMED - {blocker}"


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


def scenario_distance_status(
    scenario: Scenario,
    probe: MediapipeVisionProbe,
    calibration_face_scale: float | None,
    session_face_scale: float | None = None,
) -> tuple[bool, str]:
    if scenario.distance_mode is None:
        return True, ""
    feature = probe.latest_smoothed_attention_features
    current = (
        feature.values.get("face_scale")
        if feature is not None and not feature.reasons
        else None
    )
    if current is None or calibration_face_scale is None:
        return False, "DISTANCE: waiting for observable face scale"
    if scenario.distance_mode == "near":
        reference = (
            session_face_scale
            if session_face_scale is not None
            else calibration_face_scale
        )
        minimum = reference + (0.08 if session_face_scale is not None else 0.10)
        return (
            current >= minimum,
            f"DISTANCE: face={current:.3f}  need >= {minimum:.3f} (closer)",
        )
    if scenario.distance_mode == "calibration":
        tolerance = 0.03
        lower = calibration_face_scale - tolerance
        upper = calibration_face_scale + tolerance
        return (
            lower <= current <= upper,
            f"DISTANCE: face={current:.3f}  target {lower:.3f}-{upper:.3f}",
        )
    raise ValueError(f"unknown scenario distance mode: {scenario.distance_mode}")


def scenario_capture_status(
    scenario: Scenario,
    probe: MediapipeVisionProbe,
    *,
    distance_matches: bool,
    distance_status: str,
) -> tuple[bool, str]:
    """Check capture quality without using the model-under-test as ground truth."""
    if not distance_matches:
        return False, distance_status or "CAPTURE: distance not ready"
    quality = probe.latest_quality
    if quality is None or not quality.observable:
        reason = "; ".join(quality.reasons) if quality is not None else "waiting for image"
        return False, f"CAPTURE: {reason}"
    if scenario.signal_name == "attention_away":
        feature = probe.latest_smoothed_attention_features
        if feature is None or not feature.observable:
            reason = "; ".join(feature.reasons) if feature is not None else "waiting for face and eyes"
            return False, f"CAPTURE: {reason}"
    elif scenario.signal_name == "posture_deviation":
        feature = probe.latest_posture_features
        if feature is None or not feature.observable:
            reason = "; ".join(feature.reasons) if feature is not None else "waiting for posture features"
            return False, f"CAPTURE: {reason}"
    return True, "CAPTURE: image/face/eyes/distance ready"


def scenario_target_ready(
    *,
    capture_matches: bool,
    model_matches: bool,
    stable_seconds: float,
    require_model_match: bool = True,
) -> bool:
    """Gate a target badge while preserving quality-only legacy protocols.

    Recording is still allowed while false so model errors remain collectable.
    """

    return (
        capture_matches
        and (model_matches or not require_model_match)
        and stable_seconds >= TARGET_READY_SECONDS
    )


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
    save_trial_videos: bool = False,
    phone_score_threshold: float = 0.25,
    phone_retain_seconds: float = 2.0,
    phone_center_distance: float = 0.18,
    detector_backend: str = "efficientdet",
    yolo_model_path: str | Path | None = None,
) -> Path:
    import cv2

    requested_scenarios = set(scenario_ids)
    known_scenarios = {scenario.scenario_id for scenario in SCENARIOS}
    unknown_scenarios = requested_scenarios - known_scenarios
    if unknown_scenarios:
        raise ValueError(f"unknown validation scenarios: {sorted(unknown_scenarios)}")
    requested_diagnostics = requested_scenarios & _TUNING_DIAGNOSTIC_ONLY_SCENARIOS
    if dataset_role == "holdout" and requested_diagnostics:
        raise ValueError(
            "retired tuning diagnostics cannot be used for holdout: "
            f"{sorted(requested_diagnostics)}"
        )
    selected = [
        scenario
        for scenario in SCENARIOS
        if scenario.signal_name in signal_names
        and (not requested_scenarios or scenario.scenario_id in requested_scenarios)
        and (
            bool(requested_scenarios)
            or scenario.scenario_id not in _TUNING_DIAGNOSTIC_ONLY_SCENARIOS
        )
    ]
    unknown_signals = set(signal_names) - {scenario.signal_name for scenario in SCENARIOS}
    if unknown_signals:
        raise ValueError(f"unknown validation signals: {sorted(unknown_signals)}")
    if not selected:
        raise ValueError("at least one validation signal must be selected")
    if repetitions_override is not None and repetitions_override < 1:
        raise ValueError("repetitions override must be positive")
    if detector_backend not in {"efficientdet", "yolo"}:
        raise ValueError("detector_backend must be 'efficientdet' or 'yolo'")
    if detector_backend == "yolo" and yolo_model_path is None:
        raise ValueError("yolo_model_path is required when detector_backend='yolo'")

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
        object_backend = (
            YoloOnnxBackend(
                yolo_model_path,
                score_threshold=phone_score_threshold,
                single_class_name="cell_phone",
            )
            if detector_backend == "yolo"
            else None
        )
        probe = MediapipeVisionProbe(
            models,
            config.inference,
            profile,
            phone_score_threshold=phone_score_threshold,
            phone_retain_seconds=phone_retain_seconds,
            phone_center_distance=phone_center_distance,
            object_backend=object_backend,
        )
        calibration_face_scale = profile.attention.context_statistics().get(
            "face_scale", (None, None)
        )[0]
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        if set(signal_names) == {"phone_held"}:
            alignment_summary: dict[str, Any] = {
                "skipped": True,
                "reason": "phone-only validation does not depend on attention alignment",
                "frames_saved": 0,
            }
        else:
            alignment_result = guide_alignment(
                camera,
                probe,
                profile,
                window,
                require_posture="posture_deviation" in signal_names,
                require_face_scale=False,
            )
            alignment_summary = asdict(alignment_result)
        session_face_scale = alignment_summary.get("session_face_scale")
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
                _await_ready(
                    camera,
                    probe,
                    window,
                    scenario,
                    repetition,
                    repetitions,
                    calibration_face_scale,
                    session_face_scale,
                )
                trial_id = str(uuid4())
                trial_video_path = (
                    batch_directory / "trial-videos" / f"{scenario.scenario_id}-{trial_id}.mp4"
                    if save_trial_videos
                    else None
                )
                record, error_frame, derived_records, interrupted = _run_trial(
                    camera,
                    probe,
                    window,
                    scenario,
                    dataset_role,
                    trial_id,
                    sampler,
                    resource_samples,
                    trial_video_path=trial_video_path,
                    nominal_fps=config.camera.capture_fps,
                )
                if trial_video_path is not None and trial_video_path.exists():
                    _append_jsonl(
                        batch_directory / "trial-videos.jsonl",
                        [
                            {
                                "trial_id": trial_id,
                                "scenario_id": scenario.scenario_id,
                                "path": str(trial_video_path.resolve()),
                                "sha256": sha256_file(trial_video_path),
                                "temporary": True,
                                "review_then_delete": True,
                            }
                        ],
                    )
                is_error = not scenario_record_passes(scenario, record)
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
                "passed": scenario_record_passes(
                    scenario_by_id[record.scenario_id], record
                ),
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
    detector_backend: str = "efficientdet",
    yolo_model_path: str | Path | None = None,
) -> Path:
    """Exercise the configured away rule against one continuous live session."""

    import cv2

    if suite not in {"full", "brief"}:
        raise ValueError("away event suite must be 'full' or 'brief'")
    if detector_backend not in {"efficientdet", "yolo"}:
        raise ValueError("detector_backend must be 'efficientdet' or 'yolo'")
    yolo_path = Path(yolo_model_path) if yolo_model_path is not None else None
    if detector_backend == "yolo" and (yolo_path is None or not yolo_path.is_file()):
        raise ValueError("an existing yolo_model_path is required when detector_backend='yolo'")
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
        object_backend = (
            YoloOnnxBackend(yolo_path, score_threshold=0.25, single_class_name="cell_phone")
            if detector_backend == "yolo"
            else None
        )
        probe = MediapipeVisionProbe(
            models, config.inference, profile, object_backend=object_backend
        )
        calibration_face_scale = profile.attention.context_statistics().get(
            "face_scale", (None, None)
        )[0]
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        alignment = guide_alignment(camera, probe, profile, window)
        session_face_scale = alignment.session_face_scale

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
            _await_ready(
                camera,
                probe,
                window,
                scenario,
                1,
                1,
                calibration_face_scale,
                session_face_scale,
            )
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
        report["object_detector"] = {
            "backend": detector_backend,
            "model_path": str(yolo_path.resolve()) if yolo_path else None,
            "sha256": sha256_file(yolo_path) if yolo_path else None,
        }
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
    detector_backend: str = "efficientdet",
    yolo_model_path: str | Path | None = None,
    phone_score_threshold: float = 0.25,
    phone_retain_seconds: float = 2.0,
    phone_center_distance: float = 0.18,
) -> Path:
    """Exercise the configured phone rule in one continuous live session."""

    import cv2

    if suite not in {"full", "positive"}:
        raise ValueError("phone event suite must be 'full' or 'positive'")
    if detector_backend not in {"efficientdet", "yolo"}:
        raise ValueError("detector_backend must be 'efficientdet' or 'yolo'")
    yolo_path = Path(yolo_model_path) if yolo_model_path is not None else None
    if detector_backend == "yolo" and (yolo_path is None or not yolo_path.is_file()):
        raise ValueError("an existing yolo_model_path is required when detector_backend='yolo'")
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
        object_backend = (
            YoloOnnxBackend(
                yolo_path,
                score_threshold=phone_score_threshold,
                single_class_name="cell_phone",
            )
            if detector_backend == "yolo"
            else None
        )
        probe = MediapipeVisionProbe(
            models,
            config.inference,
            profile,
            phone_score_threshold=phone_score_threshold,
            phone_retain_seconds=phone_retain_seconds,
            phone_center_distance=phone_center_distance,
            object_backend=object_backend,
        )
        calibration_face_scale = profile.attention.context_statistics().get(
            "face_scale", (None, None)
        )[0]
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
            _await_ready(
                camera,
                probe,
                window,
                scenario,
                1,
                1,
                calibration_face_scale,
                None,
            )
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
        report["object_detector"] = {
            "backend": detector_backend,
            "model_path": str(yolo_path.resolve()) if yolo_path else None,
            "sha256": sha256_file(yolo_path) if yolo_path else None,
        }
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
    calibration_face_scale: float | None,
    session_face_scale: float | None,
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
            state_matches = current_state in scenario_accepted_states(scenario)
            distance_matches, distance_status = scenario_distance_status(
                scenario, probe, calibration_face_scale, session_face_scale
            )
            model_matches = state_matches
            capture_matches, capture_status = scenario_capture_status(
                scenario,
                probe,
                distance_matches=distance_matches,
                distance_status=distance_status,
            )
            # Phone actions are externally visible to the tester, so the large
            # badge may show whether the detector still matches the requested
            # phone/non-phone state. Other protocols keep the historical
            # quality-only badge to avoid turning model output into gaze or
            # posture ground truth.
            show_target_badge = scenario.signal_name == "phone_held"
            confirmation_matches = capture_matches and (
                model_matches or not show_target_badge
            )
            if confirmation_matches:
                if target_since is None:
                    target_since = snapshot.captured_monotonic
            else:
                target_since = None
            ready_seconds = (
                max(0.0, snapshot.captured_monotonic - target_since)
                if target_since is not None
                else 0.0
            )
            target_ready = scenario_target_ready(
                capture_matches=capture_matches,
                model_matches=model_matches,
                stable_seconds=ready_seconds,
                require_model_match=show_target_badge,
            )
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
                    scenario_acceptance_summary(scenario),
                    attention_direction_status(raw_signal)
                    if scenario.scenario_id in _DIRECTIONAL_ATTENTION_SCENARIOS
                    else "",
                    attention_action_status(scenario, raw_signal),
                    distance_status,
                    f"MODEL TARGET: {'MATCH' if model_matches else 'NOT MATCH'} (diagnostic only)",
                    capture_status,
                    f"CAPTURE QUALITY: {'READY' if capture_matches else 'NOT READY'}",
                    (
                        "TARGET READY - confirm real action with SPACE/ENTER"
                        if target_ready and show_target_badge
                        else (
                            "TARGET NOT READY - hold the real action; SPACE/ENTER records this model error"
                            if show_target_badge
                            else (
                                "CAPTURE READY - confirm real action with SPACE/ENTER"
                                if target_ready
                                else "CAPTURE NOT READY - adjust image/face/eyes/distance"
                            )
                        )
                    ),
                    "SPACE/ENTER=confirm   ESC=question/cancel",
                ),
            )
            _draw_target_badge(canvas, target_ready, target_mode=show_target_badge)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(1)
        if key & 0xFF == 27:
            raise ValidationCancelled("validation cancelled by user")
        if key & 0xFF in (10, 13, 32):
            _run_prepare_countdown(
                camera,
                probe,
                window,
                scenario,
                TRIAL_PREPARE_SECONDS,
                calibration_face_scale,
                session_face_scale,
            )
            return


def _run_prepare_countdown(
    camera: CameraWorker,
    probe: MediapipeVisionProbe,
    window: str,
    scenario: Scenario,
    duration_seconds: float,
    calibration_face_scale: float | None,
    session_face_scale: float | None,
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
            distance_matches, distance_status = scenario_distance_status(
                scenario, probe, calibration_face_scale, session_face_scale
            )
            _, capture_status = scenario_capture_status(
                scenario,
                probe,
                distance_matches=distance_matches,
                distance_status=distance_status,
            )
            canvas = _scenario_canvas(snapshot.frame, scenario)
            remaining = max(0.0, duration_seconds - (time.monotonic() - started))
            _put_lines(
                canvas,
                (
                    scenario.instruction,
                    f"PREPARE NOW - recording starts in {remaining:0.1f}s",
                    "HOLD YOUR CONFIRMED REAL ACTION",
                    distance_status,
                    capture_status,
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
    *,
    trial_video_path: Path | None = None,
    nominal_fps: float = 15.0,
) -> tuple[TrialRecord, Any, list[dict[str, Any]], bool]:
    import cv2

    started = time.monotonic()
    last_sequence = 0
    states: list[SignalValue] = []
    state_times: list[float] = []
    candidate_started: float | None = None
    last_resource_sample = 0.0
    error_frame = None
    derived_records: list[dict[str, Any]] = []
    previous_runs = (-1, -1, -1)
    interrupted = False
    video_writer = None
    video_timestamps: list[float] = []
    while time.monotonic() - started < scenario.duration_seconds:
        snapshot = camera.buffer.latest(after_sequence=last_sequence)
        if snapshot is None:
            if cv2.waitKey(1) & 0xFF == 27:
                interrupted = True
                break
            continue
        last_sequence = snapshot.sequence
        if trial_video_path is not None:
            if video_writer is None:
                trial_video_path.parent.mkdir(parents=True, exist_ok=True)
                frame_height, frame_width = snapshot.frame.shape[:2]
                video_writer = cv2.VideoWriter(
                    str(trial_video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    nominal_fps,
                    (frame_width, frame_height),
                )
                if not video_writer.isOpened():
                    raise RuntimeError(
                        f"trial video could not be opened: {trial_video_path}"
                    )
            video_writer.write(snapshot.frame)
            video_timestamps.append(snapshot.captured_monotonic - started)
        probe.process_due(snapshot.frame, snapshot.captured_monotonic)
        raw_signal = probe.observation(snapshot.captured_monotonic).signal(
            scenario.signal_name
        )
        signal = scenario_validation_signal(scenario, raw_signal)
        states.append(signal)
        state_times.append(snapshot.captured_monotonic)
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
                    "validation_signal_policy": "production_identity_v1",
                    "direction_matches": attention_direction_matches(scenario, raw_signal)
                    if scenario.signal_name == 'attention_away' else None,
                }
            )
            derived_records.append(derived)
            previous_runs = runs
        if signal.state is SignalState.VIOLATION and candidate_started is None:
            candidate_started = snapshot.captured_monotonic
        error_frame = snapshot.frame.copy()
        canvas = _scenario_canvas(snapshot.frame, scenario)
        remaining = max(0.0, scenario.duration_seconds - (time.monotonic() - started))
        _put_lines(
            canvas,
            (
                scenario.instruction,
                f"RECORDING SCORED INTERVAL: {remaining:0.1f}s",
                "HOLD YOUR CONFIRMED REAL ACTION - model result hidden",
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
    if video_writer is not None:
        video_writer.release()
        timestamp_path = trial_video_path.with_suffix(".mp4.timestamps.json")
        timestamp_path.write_text(
            json.dumps(
                {
                    "video_path": str(trial_video_path.resolve()),
                    "nominal_fps": nominal_fps,
                    "frame_elapsed_seconds": [round(value, 6) for value in video_timestamps],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
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
    for state in (SignalState.COMPLIANT, SignalState.VIOLATION, SignalState.UNKNOWN):
        metrics[f"longest_{state.value}_run_seconds"] = _longest_state_run_seconds(
            states, state_times, state
        )
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


def _longest_state_run_seconds(
    states: Sequence[SignalValue],
    captured_times: Sequence[float],
    target: SignalState,
    *,
    maximum_gap_seconds: float = 0.5,
) -> float:
    if len(states) != len(captured_times):
        raise ValueError("state and timestamp counts must match")
    longest = 0.0
    started: float | None = None
    previous: float | None = None
    for signal, captured in zip(states, captured_times, strict=True):
        if signal.state is target:
            if (
                started is None
                or previous is None
                or captured - previous > maximum_gap_seconds
            ):
                started = captured
            longest = max(longest, captured - started)
        else:
            started = None
        previous = captured
    return longest


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

    font_scale = 0.46
    line_height = 22
    rendered_lines = [
        rendered
        for line in lines
        for rendered in _wrap_display_line(frame, line, font_scale)
        if line
    ]
    height = 18 + len(rendered_lines) * line_height
    cv2.rectangle(frame, (0, 0), (frame.shape[1], height), (20, 20, 20), -1)
    for index, line in enumerate(rendered_lines):
        cv2.putText(
            frame,
            line,
            (14, 26 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _wrap_display_line(frame: Any, line: str, font_scale: float) -> list[str]:
    """Wrap overlay text inside the camera frame without horizontal clipping."""

    import cv2

    max_width = max(40, frame.shape[1] - 28)
    rendered: list[str] = []
    current = ""
    for word in line.split():
        candidate = f"{current} {word}".strip()
        width = cv2.getTextSize(
            candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )[0][0]
        if width <= max_width:
            current = candidate
            continue
        if current:
            rendered.append(current)
            current = word
        else:
            current = ""
            for char in word:
                candidate = current + char
                width = cv2.getTextSize(
                    candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )[0][0]
                if width <= max_width:
                    current = candidate
                else:
                    rendered.append(current)
                    current = char
    if current:
        rendered.append(current)
    return rendered or [line]


def _draw_target_badge(
    frame: Any, ready: bool, *, target_mode: bool = False
) -> None:
    """Draw an obvious readiness mark without depending on font glyph support."""

    import cv2

    height, width = frame.shape[:2]
    color = (55, 200, 55) if ready else (60, 170, 230)
    if target_mode:
        label = "TARGET READY" if ready else "TARGET NOT READY"
    else:
        label = "CAPTURE READY" if ready else "CAPTURE NOT READY"
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
    repository = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repository.as_posix()}",
                "-C",
                str(repository),
                "rev-parse",
                "HEAD",
            ],
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
