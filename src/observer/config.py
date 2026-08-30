from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

from observer.rules import RuleConfig, default_rule_configs


@dataclass(frozen=True, slots=True)
class CameraConfig:
    device_index: int = 0
    width: int = 640
    height: int = 480
    capture_fps: float = 15.0
    backend: str = "dshow"
    read_failure_backoff_ms: int = 200

    def validate(self) -> None:
        if self.device_index < 0:
            raise ValueError("camera.device_index must be non-negative")
        if not 160 <= self.width <= 3840 or not 120 <= self.height <= 2160:
            raise ValueError("camera dimensions are outside the supported range")
        if not 1.0 <= self.capture_fps <= 60.0:
            raise ValueError("camera.capture_fps must be between 1 and 60")
        if self.backend not in {"auto", "msmf", "dshow"}:
            raise ValueError("camera.backend must be auto, msmf, or dshow")
        if not 10 <= self.read_failure_backoff_ms <= 5000:
            raise ValueError("camera.read_failure_backoff_ms must be between 10 and 5000")


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    face_fps: float = 8.0
    pose_fps: float = 4.0
    object_fps: float = 1.5
    object_burst_fps: float = 5.0
    busy_face_fps: float = 5.0
    busy_pose_fps: float = 2.0
    busy_object_fps: float = 0.75

    def validate(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if not 0.1 <= value <= 30.0:
                raise ValueError(f"inference.{name} must be between 0.1 and 30")
        if self.object_burst_fps < self.object_fps:
            raise ValueError("inference.object_burst_fps must not be lower than object_fps")


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    cpu_average_percent: float = 15.0
    cpu_p95_percent: float = 25.0
    memory_steady_mb: float = 500.0
    memory_growth_mb_per_hour: float = 50.0
    max_inference_workers: int = 3

    def validate(self) -> None:
        if not 1.0 <= self.cpu_average_percent <= 100.0:
            raise ValueError("resources.cpu_average_percent must be between 1 and 100")
        if not self.cpu_average_percent <= self.cpu_p95_percent <= 100.0:
            raise ValueError("resources.cpu_p95_percent must be between average and 100")
        if self.memory_steady_mb <= 0 or self.memory_growth_mb_per_hour < 0:
            raise ValueError("resource memory budgets must be non-negative")
        if not 1 <= self.max_inference_workers <= 4:
            raise ValueError("resources.max_inference_workers must be between 1 and 4")


@dataclass(frozen=True, slots=True)
class AppConfig:
    camera: CameraConfig
    inference: InferenceConfig
    resources: ResourceBudget
    rules: tuple[RuleConfig, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AppConfig":
        allowed = {"camera", "inference", "resources", "rules"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")
        raw_rules = value.get("rules")
        if raw_rules is None:
            rules = default_rule_configs()
        elif isinstance(raw_rules, list):
            rules = tuple(RuleConfig(**dict(rule)) for rule in raw_rules)
        else:
            raise ValueError("rules must be a JSON array")
        config = cls(
            camera=CameraConfig(**dict(value.get("camera", {}))),
            inference=InferenceConfig(**dict(value.get("inference", {}))),
            resources=ResourceBudget(**dict(value.get("resources", {}))),
            rules=rules,
        )
        config.validate()
        return config

    def validate(self) -> None:
        self.camera.validate()
        self.inference.validate()
        self.resources.validate()
        if self.inference.face_fps > self.camera.capture_fps:
            raise ValueError("inference.face_fps cannot exceed camera.capture_fps")
        if self.inference.object_burst_fps > self.camera.capture_fps:
            raise ValueError("inference.object_burst_fps cannot exceed camera.capture_fps")
        rule_ids = [rule.rule_id for rule in self.rules]
        if not rule_ids:
            raise ValueError("at least one rule must be configured")
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("configured rule ids must be unique")


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        config_text = resources.files("observer").joinpath("default_config.json").read_text(
            encoding="utf-8"
        )
    else:
        config_text = Path(path).read_text(encoding="utf-8")
    raw = json.loads(config_text)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a JSON object")
    return AppConfig.from_mapping(raw)
