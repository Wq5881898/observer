from __future__ import annotations

import math
import os
import statistics
import time
from dataclasses import dataclass
from typing import Iterable

from observer.config import ResourceBudget


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    captured_monotonic: float
    process_cpu_percent: float
    process_memory_mb: float
    system_cpu_percent: float
    system_available_memory_mb: float


@dataclass(frozen=True, slots=True)
class ResourceSummary:
    sample_count: int
    process_cpu_average_percent: float
    process_cpu_p95_percent: float
    process_memory_peak_mb: float
    process_memory_growth_mb_per_hour: float | None
    within_budget: bool
    violations: tuple[str, ...]


class ResourceSampler:
    """Collect process metrics using Windows Task Manager's 0-100% CPU convention."""

    def __init__(self) -> None:
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError("psutil is required for resource sampling") from exc
        self._psutil = psutil
        self._process = psutil.Process(os.getpid())
        self._logical_cpus = max(psutil.cpu_count(logical=True) or 1, 1)
        self._process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)

    def sample(self) -> ResourceSnapshot:
        raw_process_cpu = self._process.cpu_percent(interval=None)
        process_cpu = raw_process_cpu / self._logical_cpus
        memory = self._process.memory_info().rss / (1024 * 1024)
        virtual_memory = self._psutil.virtual_memory()
        return ResourceSnapshot(
            captured_monotonic=time.monotonic(),
            process_cpu_percent=process_cpu,
            process_memory_mb=memory,
            system_cpu_percent=self._psutil.cpu_percent(interval=None),
            system_available_memory_mb=virtual_memory.available / (1024 * 1024),
        )


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[rank]


def summarize_resources(
    samples: Iterable[ResourceSnapshot], budget: ResourceBudget
) -> ResourceSummary:
    values = list(samples)
    if not values:
        return ResourceSummary(0, 0.0, 0.0, 0.0, 0.0, False, ("no resource samples",))

    cpu_values = [value.process_cpu_percent for value in values]
    memory_values = [value.process_memory_mb for value in values]
    elapsed_hours = (values[-1].captured_monotonic - values[0].captured_monotonic) / 3600.0
    growth = None
    if elapsed_hours >= 0.25:
        growth = max(0.0, memory_values[-1] - memory_values[0]) / elapsed_hours

    cpu_average = statistics.fmean(cpu_values)
    cpu_p95 = percentile(cpu_values, 0.95)
    memory_peak = max(memory_values)
    violations: list[str] = []
    if cpu_average > budget.cpu_average_percent:
        violations.append("average CPU budget exceeded")
    if cpu_p95 > budget.cpu_p95_percent:
        violations.append("p95 CPU budget exceeded")
    if memory_peak > budget.memory_steady_mb:
        violations.append("steady memory budget exceeded")
    if growth is not None and growth > budget.memory_growth_mb_per_hour:
        violations.append("memory growth budget exceeded")

    return ResourceSummary(
        sample_count=len(values),
        process_cpu_average_percent=cpu_average,
        process_cpu_p95_percent=cpu_p95,
        process_memory_peak_mb=memory_peak,
        process_memory_growth_mb_per_hour=growth,
        within_budget=not violations,
        violations=tuple(violations),
    )
