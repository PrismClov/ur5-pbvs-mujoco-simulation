#!/usr/bin/env python3
"""Fixed-rate ArUco PBVS using the vendor Dobot Python SDK directly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import threading
import time

import numpy as np

from nova5_visual_servo.core import Config, Controller, TargetTracker, Fault
from nova5_visual_servo.robot import URDFRobot

from .sdk.interface import DobotInterface
from .sdk.realtime import run_control_loop


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--ip", default="192.168.5.2")
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--feedback-fps", type=float, default=60.0)
    parser.add_argument("--speed", type=int, default=20)
    parser.add_argument("--servo-t", type=float, default=0.02)
    parser.add_argument("--aheadtime", type=float, default=50.0)
    parser.add_argument("--gain", type=float, default=500.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "dobot/direct_tracking_site.json",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=REPO_ROOT / "nova5_visual_servo/models/nova5_robot.urdf",
    )
    parser.add_argument("--pose-topic", default="/tag_tracker/pose_camera")
    parser.add_argument("--output", type=Path)
    return parser


def pose_from_message(message):
    from scipy.spatial.transform import Rotation
    p, q = message.pose.position, message.pose.orientation
    T = np.eye(4)
    T[:3, 3] = [p.x, p.y, p.z]
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    return T


def main() -> int:
    args = build_parser().parse_args()
    if not args.execute:
        print("Dry-run only. Add --execute to enable direct SDK motion.")
        return 0

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node

    settings = json.loads(args.config.read_text())
    T_EC = np.asarray(settings["T_EC"], dtype=float)
    T_QC_des = np.asarray(settings["T_QC_des"], dtype=float)
    limits = Config(**settings["control"])
    model = URDFRobot(args.urdf)
    tracker = TargetTracker(**settings["tracker"])

    robot = DobotInterface(
        ip=args.ip,
        speed=args.speed,
        t=args.servo_t,
        aheadtime=args.aheadtime,
        gain=args.gain,
        update_fps=args.feedback_fps,
        joint_command_max_velocity=0.5,
        joint_command_max_dt=0.05,
        joint_command_smoothing_factor=0.0,
    )
    rclpy.init()
    node = Node("dobot_direct_visual_servo")
    state_lock = threading.Lock()
    stop_event = threading.Event()
    state = {
        "sequence": 0,
        "controller": None,
        "previous_q": None,
        "previous_feedback_time": None,
        "qdot": np.zeros(6),
        "callback_error": None,
        "commands": 0,
        "last_error": None,
    }

    def target_callback(message: PoseStamped) -> None:
        if stop_event.is_set():
            return
        try:
            feedback = robot.get_joint_position()
            if feedback is None or len(feedback) != 6:
                return
            q = np.asarray(feedback, dtype=float)
            with state_lock:
                state["sequence"] += 1
                tracker.update(
                    state["sequence"],
                    time.monotonic(),
                    model.fk(q),
                    T_EC,
                    pose_from_message(message),
                )
        except Exception as exc:
            state["callback_error"] = str(exc)
            stop_event.set()

    node.create_subscription(PoseStamped, args.pose_topic, target_callback, 10)
    executor_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        name="ros-callback-thread",
        daemon=True,
    )
    executor_thread.start()

    log_rows = []

    def control_step() -> None:
        if state["callback_error"] is not None:
            raise RuntimeError(state["callback_error"])
        snapshot = robot.feedback_snapshot()
        q_list = snapshot["joint_position_rad"]
        feedback_time = snapshot["monotonic"]
        if q_list is None or feedback_time is None:
            return
        q = np.asarray(q_list, dtype=float)
        now = time.monotonic()
        with state_lock:
            previous_q = state["previous_q"]
            previous_t = state["previous_feedback_time"]
            if previous_q is not None and previous_t is not None:
                dt = feedback_time - previous_t
                if 0.002 <= dt <= 0.1:
                    raw_qdot = (q - previous_q) / dt
                    state["qdot"] = (
                        0.3 * raw_qdot + 0.7 * state["qdot"])
            state["previous_q"] = q.copy()
            state["previous_feedback_time"] = feedback_time
            if state["controller"] is None:
                if not tracker.ready:
                    return
                state["controller"] = Controller(
                    limits,
                    tracker,
                    T_QC_des,
                    T_EC,
                    model.fk,
                    model.jac,
                    lambda candidate: bool(
                        np.all(model.fk(candidate)[:3, 3] >=
                               np.asarray(settings["workspace_min"]))
                        and np.all(model.fk(candidate)[:3, 3] <=
                                   np.asarray(settings["workspace_max"]))),
                    q,
                    state["qdot"],
                    now,
                )
                return
            controller = state["controller"]
            output = controller.tick(
                now, q, state["qdot"], feedback_time)
        sent_q = robot.set_joints(output["q_plan"])
        with state_lock:
            state["commands"] += 1
            state["last_error"] = output["error"].tolist()
        row = {
            "time": now,
            "feedback_time": feedback_time,
            "q_actual": q.tolist(),
            "q_sent": sent_q,
            "commands": state["commands"],
            "feedback_count": snapshot["feedback_count"],
            **{
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in output.items()
            },
        }
        log_rows.append(row)

    def handle_signal(signum, frame):
        del signum, frame
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    timer = None
    if args.duration > 0.0:
        timer = threading.Timer(args.duration, stop_event.set)
        timer.start()
    stats = None
    error = None
    try:
        while not stop_event.is_set() and state["callback_error"] is None:
            if tracker.ready:
                break
            time.sleep(0.05)
        if not tracker.ready:
            raise RuntimeError(
                f"no usable target from {args.pose_topic}: {tracker.reason}")
        stats = run_control_loop(
            control_step, stop_event, control_hz=args.control_hz)
    except (Exception, Fault) as exc:
        error = f"{type(exc).__name__}: {exc}"
        stop_event.set()
    finally:
        if timer is not None:
            timer.cancel()
        try:
            robot.dashboard.Stop()
        except Exception:
            pass
        robot.disconnect()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        executor_thread.join(timeout=2.0)

    report = {
        "error": error,
        "tracker": {
            "ready": tracker.ready,
            "reason": tracker.reason,
            "rejections": tracker.rejections,
        },
        "control": stats.__dict__ if stats is not None else None,
        "commands": state["commands"],
        "last_error": state["last_error"],
        "samples": log_rows,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    return 0 if error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
