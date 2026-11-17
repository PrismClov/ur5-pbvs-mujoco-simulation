"""Fixed-rate scheduling helpers for the vendor Dobot SDK."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Callable


@dataclass(frozen=True)
class LoopStats:
    requested_hz: float
    ticks: int
    duration_s: float
    actual_hz: float
    deadlines_missed: int
    max_lateness_s: float
    step_duration_p50_s: float
    step_duration_p95_s: float


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def run_control_loop(
    control_step: Callable[[], None],
    stop_event: threading.Event,
    control_hz: float = 50.0,
) -> LoopStats:
    """Run ``control_step`` at a fixed rate without catch-up bursts.

    If a step overruns one or more periods, missed slots are discarded. The
    loop never sends queued commands in a burst trying to recover time.
    """
    if control_hz <= 0.0 or not math.isfinite(control_hz):
        raise ValueError("control_hz must be positive")
    period = 1.0 / control_hz
    deadline = time.monotonic()
    started = deadline
    ticks = 0
    missed = 0
    max_lateness = 0.0
    step_durations: list[float] = []
    while not stop_event.is_set():
        remaining = deadline - time.monotonic()
        if remaining > 0.0 and stop_event.wait(remaining):
            break
        step_started = time.monotonic()
        control_step()
        step_finished = time.monotonic()
        step_durations.append(step_finished - step_started)
        ticks += 1
        deadline += period
        now = time.monotonic()
        if deadline <= now:
            lateness = now - deadline
            max_lateness = max(max_lateness, lateness)
            missed += int(lateness / period) + 1
            deadline += (int(lateness / period) + 1) * period
    duration = time.monotonic() - started
    return LoopStats(
        requested_hz=control_hz,
        ticks=ticks,
        duration_s=duration,
        actual_hz=ticks / duration if duration > 0.0 else 0.0,
        deadlines_missed=missed,
        max_lateness_s=max_lateness,
        step_duration_p50_s=_percentile(step_durations, 0.5),
        step_duration_p95_s=_percentile(step_durations, 0.95),
    )
