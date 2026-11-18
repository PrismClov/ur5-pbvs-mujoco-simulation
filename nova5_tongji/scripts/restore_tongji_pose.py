#!/usr/bin/env python3
"""Restore the Nova5 to a captured joint baseline with bounded ServoJ motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dobot.sdk.interface import DobotInterface
from dobot.sdk.realtime import run_control_loop


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT
            / "nova5_tongji/field_runtime/config/tongji_direct_tracking.json"
        ),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="JSON from the pose capture containing joint_position_rad_median.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional restore report JSON.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--max-joint-speed", type=float, default=0.2)
    parser.add_argument("--max-joint-acceleration", type=float, default=0.6)
    parser.add_argument("--tolerance-rad", type=float, default=0.00873)
    parser.add_argument("--settle-s", type=float, default=0.5)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    return parser


def servo_ack_ok(ack):
    if not ack:
        return True
    try:
        return int(str(ack).strip().split(",", 1)[0]) == 0
    except ValueError:
        return False


def main():
    args = build_parser().parse_args()
    target = np.asarray(
        json.loads(args.baseline.read_text())["joint_position_rad_median"],
        dtype=float,
    )
    if target.shape != (6,) or not np.isfinite(target).all():
        raise ValueError("baseline must contain six finite joint angles")
    if args.control_hz <= 0.0:
        raise ValueError("control-hz must be positive")
    if (
        args.max_joint_speed <= 0.0
        or args.max_joint_acceleration <= 0.0
        or args.tolerance_rad <= 0.0
    ):
        raise ValueError("speed, acceleration and tolerance must be positive")

    if not args.execute:
        print(json.dumps({
            "mode": "dry-run",
            "baseline": str(args.baseline),
            "target_joint_position_rad": target.tolist(),
            "target_joint_position_deg": np.rad2deg(target).tolist(),
        }, indent=2))
        return 0

    settings = json.loads(args.config.read_text())
    robot = settings["robot"]
    transport = DobotInterface(
        dashboardPort=robot["dashboard_port"],
        feedPortFour=robot["feedback_port"],
        ip=robot["ip"],
        speed=robot["speed_factor"],
        t=robot["servo_t_s"],
        aheadtime=robot["aheadtime"],
        gain=robot["gain"],
        update_fps=robot["feedback_fps"],
        joint_command_max_velocity=args.max_joint_speed,
        joint_command_max_dt=0.05,
        joint_command_smoothing_factor=0.0,
    )
    stop_event = threading.Event()
    state = {
        "started": time.monotonic(),
        "settled_since": None,
        "max_feedback_error_rad": 0.0,
        "samples": 0,
        "last_feedback_error_rad": None,
        "last_command_error_rad": None,
        "trajectory_duration_s": None,
    }

    def restore_step():
        current = transport.get_joint_position()
        if current is None:
            raise RuntimeError("joint feedback unavailable during restore")
        current = np.asarray(current, dtype=float)
        elapsed = time.monotonic() - state["started"]
        duration = state["trajectory_duration_s"]
        progress = (
            1.0 if duration is None or duration <= 0.0
            else float(np.clip(elapsed / duration, 0.0, 1.0))
        )
        blend = progress * progress * progress * (
            10.0 + progress * (-15.0 + 6.0 * progress)
        )
        command_target = before + (target - before) * blend
        feedback_error = float(np.max(np.abs(current - target)))
        command = np.asarray(
            transport.set_joints(command_target), dtype=float
        )
        if not servo_ack_ok(transport.servo_send_last_result):
            raise RuntimeError(
                f"ServoJ restore command rejected: "
                f"{transport.servo_send_last_result}"
            )
        state["max_feedback_error_rad"] = max(
            state["max_feedback_error_rad"], feedback_error
        )
        state["last_feedback_error_rad"] = feedback_error
        state["last_command_error_rad"] = float(np.max(np.abs(command - target)))
        state["samples"] += 1
        now = time.monotonic()
        if feedback_error <= args.tolerance_rad:
            if state["settled_since"] is None:
                state["settled_since"] = now
            elif now - state["settled_since"] >= args.settle_s:
                stop_event.set()
        else:
            state["settled_since"] = None
        if now - state["started"] >= args.timeout_s:
            raise TimeoutError("pose restore timed out")

    before = None
    error = None
    stats = None
    try:
        deadline = time.monotonic() + 3.0
        while transport.get_joint_position() is None:
            if time.monotonic() >= deadline:
                raise RuntimeError("no joint feedback before pose restore")
            time.sleep(0.02)
        before = np.asarray(transport.get_joint_position(), dtype=float)
        max_delta = float(np.max(np.abs(target - before)))
        duration_for_speed = 1.875 * max_delta / args.max_joint_speed
        duration_for_acceleration = np.sqrt(
            5.7735026919 * max_delta / args.max_joint_acceleration
        )
        state["trajectory_duration_s"] = max(
            0.5, duration_for_speed, duration_for_acceleration
        )
        stats = run_control_loop(
            restore_step,
            stop_event,
            control_hz=args.control_hz,
            pace_from_start=True,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        stop_event.set()
    finally:
        after = transport.get_joint_position()
        try:
            transport.dashboard.Stop()
        except Exception:
            pass
        transport.disconnect()

    result = {
        "error": error,
        "baseline": str(args.baseline.resolve()),
        "target_joint_position_rad": target.tolist(),
        "before_joint_position_rad": (
            None if before is None else before.tolist()
        ),
        "after_joint_position_rad": (
            None if after is None else np.asarray(after).tolist()
        ),
        "final_error_rad": (
            None
            if after is None
            else float(np.max(np.abs(np.asarray(after) - target)))
        ),
        "tolerance_rad": args.tolerance_rad,
        "settle_s": args.settle_s,
        "max_joint_speed_rad_s": args.max_joint_speed,
        "max_joint_acceleration_rad_s2": args.max_joint_acceleration,
        "trajectory_duration_s": state["trajectory_duration_s"],
        "control_loop": None if stats is None else stats.__dict__,
        "restore_samples": state["samples"],
        "max_feedback_error_rad": state["max_feedback_error_rad"],
        "last_command_error_rad": state["last_command_error_rad"],
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return 0 if error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
