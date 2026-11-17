#!/usr/bin/env python3
"""Measure direct Dobot feedback packet rate without enabling the robot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from .dobot_api import DobotApiFeedBack


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="192.168.5.2")
    parser.add_argument("--port", type=int, default=30004)
    parser.add_argument("--read-hz", type=float, default=60.0)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.read_hz <= 0.0 or args.duration <= 0.0:
        raise ValueError("read-hz and duration must be positive")

    feedback = DobotApiFeedBack(args.ip, args.port)
    periods = []
    received = 0
    invalid = 0
    started = time.monotonic()
    previous = None
    deadline = started
    try:
        while time.monotonic() - started < args.duration:
            remaining = deadline - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            packet = feedback.feedBackData()
            now = time.monotonic()
            if packet is not None:
                received += 1
                if previous is not None:
                    periods.append(now - previous)
                previous = now
            else:
                invalid += 1
            deadline += 1.0 / args.read_hz
            now = time.monotonic()
            if deadline <= now:
                missed = int((now - deadline) * args.read_hz) + 1
                deadline += missed / args.read_hz
    finally:
        feedback.close()
    duration = time.monotonic() - started
    result = {
        "requested_read_hz": args.read_hz,
        "received_packets": received,
        "invalid_reads": invalid,
        "duration_s": duration,
        "actual_feedback_hz": received / duration,
        "period_p50_ms": (
            percentile(periods, 0.5) * 1000.0 if periods else None),
        "period_p95_ms": (
            percentile(periods, 0.95) * 1000.0 if periods else None),
        "period_max_ms": max(periods) * 1000.0 if periods else None,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
