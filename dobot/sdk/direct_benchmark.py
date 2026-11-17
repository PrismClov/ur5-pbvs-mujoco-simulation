#!/usr/bin/env python3
"""Benchmark direct vendor-SDK feedback and ServoJ scheduling without motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import threading
import time

from .interface import DobotInterface
from .realtime import run_control_loop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--ip", default="192.168.5.2")
    parser.add_argument("--dashboard-port", type=int, default=29999)
    parser.add_argument("--feedback-port", type=int, default=30004)
    parser.add_argument("--feedback-fps", type=float, default=60.0)
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--speed", type=int, default=20)
    parser.add_argument("--servo-t", type=float, default=0.02)
    parser.add_argument("--aheadtime", type=float, default=50.0)
    parser.add_argument("--gain", type=float, default=500.0)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.execute:
        print("Dry-run only. Add --execute; the benchmark holds the current joints.")
        return 0
    if args.duration <= 0.0:
        raise ValueError("duration must be positive")

    robot = DobotInterface(
        dashboardPort=args.dashboard_port,
        feedPortFour=args.feedback_port,
        ip=args.ip,
        speed=args.speed,
        t=args.servo_t,
        aheadtime=args.aheadtime,
        gain=args.gain,
        update_fps=args.feedback_fps,
        joint_command_max_velocity=2.0,
        joint_command_max_dt=0.05,
        joint_command_smoothing_factor=0.0,
    )
    stop_event = threading.Event()
    try:
        deadline = time.monotonic() + 3.0
        while robot.get_joint_position() is None:
            if time.monotonic() >= deadline:
                raise RuntimeError("no Dobot feedback before benchmark")
            time.sleep(0.01)

        feedback_start = robot.feedback_count
        feedback_time_start = robot.feedback_last_monotonic
        send_durations = []
        acknowledgements = []

        def hold_current_joints() -> None:
            current = robot.get_joint_position()
            if current is None or len(current) != 6:
                raise RuntimeError("joint feedback unavailable")
            robot.set_joints(current)
            if robot.servo_send_last_duration_s is not None:
                send_durations.append(robot.servo_send_last_duration_s)
            acknowledgements.append(str(robot.servo_send_last_result or ""))

        timer = threading.Timer(args.duration, stop_event.set)
        timer.start()
        stats = run_control_loop(
            hold_current_joints,
            stop_event,
            control_hz=args.control_hz,
        )
        timer.cancel()
        feedback_end = robot.feedback_count
        feedback_time_end = robot.feedback_last_monotonic
        feedback_duration = (
            feedback_time_end - feedback_time_start
            if feedback_time_start is not None and feedback_time_end is not None
            else 0.0
        )
        ack_ids = []
        ack_errors = []
        ack_order_violations = 0
        for acknowledgement in acknowledgements:
            numbers = [int(value) for value in re.findall(r"-?\d+", acknowledgement)]
            if len(numbers) >= 2:
                ack_errors.append(numbers[0])
                ack_ids.append(numbers[1])
        for previous, current in zip(ack_ids, ack_ids[1:]):
            if current <= previous:
                ack_order_violations += 1
        result = {
            "control": stats.__dict__,
            "servo_send_count": robot.servo_send_count,
            "servo_send_duration_p50_s": (
                sorted(send_durations)[len(send_durations) // 2]
                if send_durations else None),
            "servo_send_duration_max_s": (
                max(send_durations) if send_durations else None),
            "servo_acknowledgements": sum(bool(x) for x in acknowledgements),
            "servo_empty_acknowledgements": sum(
                not bool(x) for x in acknowledgements),
            "servo_ack_errors": sorted(set(ack_errors)),
            "servo_ack_command_ids_min": min(ack_ids) if ack_ids else None,
            "servo_ack_command_ids_max": max(ack_ids) if ack_ids else None,
            "servo_ack_order_violations": ack_order_violations,
            "feedback_packets": feedback_end - feedback_start,
            "feedback_duration_s": feedback_duration,
            "feedback_hz": (
                (feedback_end - feedback_start) / feedback_duration
                if feedback_duration > 0.0 else None),
        }
        rendered = json.dumps(result, indent=2)
        print(rendered)
        if args.output is not None:
            args.output.write_text(rendered + "\n")
        return 0
    finally:
        try:
            robot.dashboard.Stop()
        except Exception:
            pass
        robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
