#!/usr/bin/env python3
"""Track a 100 mm marker with Nova5 using the upstream Tongji Aimer."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import signal
import threading
import time

import numpy as np

from nova5_visual_servo.core import Fault
from nova5_visual_servo.robot import URDFRobot

from .tongji_controller import (
    TongjiControlConfig,
    TongjiController,
    cartesian_step,
)
from .tracker import TongjiAimer, TongjiTargetTracker


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "nova5_tongji/field_runtime/config/tongji_direct_tracking.json"
DEFAULT_AIMER = ROOT / "nova5_tongji/field_runtime/config/tongji_marker_aimer.yaml"
DEFAULT_URDF = ROOT / "nova5_visual_servo/models/nova5_robot.urdf"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--aimer-config", type=Path, default=DEFAULT_AIMER)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--pose-topic", default="/tag_tracker/pose_camera")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--output-log", type=Path)
    parser.add_argument("--ip")
    parser.add_argument("--control-hz", type=float)
    return parser


def pose_from_message(message):
    from scipy.spatial.transform import Rotation

    p, q = message.pose.position, message.pose.orientation
    result = np.eye(4)
    result[:3, 3] = [p.x, p.y, p.z]
    result[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    return result


def dobot_pose_to_matrix(values):
    from scipy.spatial.transform import Rotation

    values = np.asarray(values, dtype=float)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("invalid Dobot TCP feedback")
    result = np.eye(4)
    result[:3, 3] = values[:3] / 1000.0
    result[:3, :3] = Rotation.from_euler(
        "xyz", values[3:], degrees=True
    ).as_matrix()
    return result


def matrix_to_dobot_pose(matrix):
    from scipy.spatial.transform import Rotation

    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("invalid Cartesian target")
    xyz = matrix[:3, 3] * 1000.0
    rpy = Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz", degrees=True)
    return np.r_[xyz, rpy]


def servo_ack_success(acknowledgement, allow_empty=False):
    if not acknowledgement:
        return allow_empty
    match = re.match(r"\s*(-?\d+)", str(acknowledgement or ""))
    return match is not None and int(match.group(1)) == 0


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def main():
    args = build_parser().parse_args()
    settings = json.loads(args.config.read_text())
    if not args.execute:
        print(json.dumps({
            "mode": "dry-run",
            "config": str(args.config),
            "aimer_config": str(args.aimer_config),
            "pose_topic": args.pose_topic,
            "controller": settings["controller"],
            "robot": settings["robot"],
        }, indent=2))
        return 0

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from dobot.sdk.interface import DobotInterface
    from dobot.sdk.realtime import run_control_loop

    robot_settings = settings["robot"]
    control_settings = TongjiControlConfig(**settings["controller"])
    control_hz = args.control_hz or robot_settings["control_hz"]
    control_period = 1.0 / control_hz
    model = URDFRobot(args.urdf)
    tracker = TongjiTargetTracker(**settings["tracker"])
    aimer = TongjiAimer(args.aimer_config)

    transport = DobotInterface(
        dashboardPort=robot_settings["dashboard_port"],
        feedPortFour=robot_settings["feedback_port"],
        ip=args.ip or robot_settings["ip"],
        speed=robot_settings["speed_factor"],
        t=robot_settings["servo_t_s"],
        aheadtime=robot_settings["aheadtime"],
        gain=robot_settings["gain"],
        update_fps=robot_settings["feedback_fps"],
        joint_command_max_velocity=control_settings.max_joint_velocity_rad_s,
        joint_command_max_dt=0.05,
        joint_command_smoothing_factor=0.0,
    )
    command_mode = settings.get("command_mode", "servo_j")
    if command_mode not in ("servo_j", "servo_p"):
        raise ValueError(f"unsupported command mode: {command_mode}")

    rclpy.init()
    node = Node("nova5_tongji_direct_tracking")
    stop_event = threading.Event()
    lock = threading.Lock()
    state = {
        "sequence": 0,
        "q_previous": None,
        "feedback_previous": None,
        "qdot": np.zeros(6),
        "controller": None,
        "commands": 0,
        "callback_error": None,
        "last_output": None,
        "servo_pose": None,
    }

    def target_callback(message):
        if stop_event.is_set():
            return
        try:
            snapshot = transport.feedback_snapshot()
            q = snapshot["joint_position_rad"]
            if q is None or len(q) != 6:
                return
            q = np.asarray(q, dtype=float)
            with lock:
                state["sequence"] += 1
                tracker.update(
                    state["sequence"],
                    time.monotonic(),
                    model.fk(q),
                    np.asarray(settings["T_EC"], dtype=float),
                    pose_from_message(message),
                )
        except Exception as exc:
            state["callback_error"] = f"{type(exc).__name__}: {exc}"
            stop_event.set()

    node.create_subscription(PoseStamped, args.pose_topic, target_callback, 1)
    ros_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        name="tongji-ros-callback",
        daemon=True,
    )
    ros_thread.start()

    log_handle = None
    if args.output_log is not None:
        args.output_log.parent.mkdir(parents=True, exist_ok=True)
        log_handle = args.output_log.open("w", encoding="utf-8")

    def workspace_ok(q):
        position = model.fk(q)[:3, 3]
        return bool(
            np.all(position >= np.asarray(settings["workspace_min"]))
            and np.all(position <= np.asarray(settings["workspace_max"]))
        )

    def control_step():
        if state["callback_error"] is not None:
            raise RuntimeError(state["callback_error"])
        snapshot = transport.feedback_snapshot()
        q_list = snapshot["joint_position_rad"]
        feedback_time = snapshot["monotonic"]
        if q_list is None or feedback_time is None:
            return
        q = np.asarray(q_list, dtype=float)
        tool_vector = snapshot["tool_vector_mm_deg"]
        if tool_vector is None or len(tool_vector) != 6:
            return
        now = time.monotonic()
        with lock:
            previous_q = state["q_previous"]
            previous_t = state["feedback_previous"]
            if previous_q is not None and previous_t is not None:
                feedback_dt = feedback_time - previous_t
                if 0.002 <= feedback_dt <= 0.1:
                    state["qdot"] = (
                        0.3 * (q - previous_q) / feedback_dt
                        + 0.7 * state["qdot"]
                    )
            state["q_previous"] = q.copy()
            state["feedback_previous"] = feedback_time
            if state["controller"] is None:
                if not tracker.ready:
                    return
                state["controller"] = TongjiController(
                    control_settings,
                    tracker,
                    aimer,
                    np.asarray(settings["T_EC"], dtype=float),
                    model,
                    workspace_ok,
                    q,
                    state["qdot"],
                    now,
                )
                return
            output = state["controller"].tick(
                now, q, state["qdot"], feedback_time
            )

        if command_mode == "servo_p":
            current_pose = dobot_pose_to_matrix(tool_vector)
            raw_twist = np.asarray(
                output["cartesian_twist"], dtype=float
            )
            with lock:
                if state["servo_pose"] is None:
                    state["servo_pose"] = current_pose.copy()
                target_pose = cartesian_step(
                    state["servo_pose"],
                    raw_twist,
                    control_period,
                )
                state["servo_pose"] = target_pose.copy()
            command_value = matrix_to_dobot_pose(target_pose)
            send_started = time.monotonic()
            acknowledgement = transport.dashboard.ServoP(
                *command_value.tolist(),
                t=robot_settings["servo_t_s"],
                aheadtime=robot_settings["aheadtime"],
                gain=robot_settings["gain"],
            )
            send_duration = time.monotonic() - send_started
        else:
            target_pose = None
            command_value = np.asarray(output["q_plan"], dtype=float)
            send_started = time.monotonic()
            acknowledgement = transport.dashboard.ServoJ(
                *np.rad2deg(command_value).tolist(),
                t=robot_settings["servo_t_s"],
                aheadtime=robot_settings["aheadtime"],
                gain=robot_settings["gain"],
            )
            send_duration = time.monotonic() - send_started
        with lock:
            state["commands"] += 1
            state["last_output"] = output
            row = {
                "time": now,
                "feedback_time": feedback_time,
                "q_actual": q.tolist(),
                "tool_vector_mm_deg": np.asarray(tool_vector).tolist(),
                "command_mode": command_mode,
                "command_value": json_value(command_value),
                "servo_ack": str(acknowledgement or ""),
                "send_duration_s": send_duration,
                "cartesian_target": (
                    None if target_pose is None else target_pose.tolist()
                ),
                "command": state["commands"],
                "controller": {
                    key: json_value(value) for key, value in output.items()
                },
            }
        if log_handle is not None:
            log_handle.write(json.dumps(row, allow_nan=False) + "\n")
            log_handle.flush()
        if command_mode == "servo_p" and not servo_ack_success(acknowledgement):
            raise Fault(
                f"ServoP rejected at command {state['commands']}: "
                f"{acknowledgement}"
            )
        if (
            command_mode == "servo_j"
            and not servo_ack_success(acknowledgement, allow_empty=True)
        ):
            raise Fault(
                f"ServoJ rejected at command {state['commands']}: "
                f"{acknowledgement}"
            )

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
    accepted_count = 0
    try:
        deadline = time.monotonic() + 10.0
        while not stop_event.is_set() and not tracker.ready:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"no usable target on {args.pose_topic}: {tracker.reason}"
                )
            time.sleep(0.05)
        if not tracker.ready:
            raise RuntimeError(f"no usable target: {tracker.reason}")
        stats = run_control_loop(
            control_step,
            stop_event,
            control_hz=control_hz,
        )
    except (Exception, Fault) as exc:
        error = f"{type(exc).__name__}: {exc}"
        stop_event.set()
    finally:
        accepted_count = tracker.accepted_count
        if timer is not None:
            timer.cancel()
        try:
            transport.dashboard.Stop()
        except Exception:
            pass
        transport.disconnect()
        tracker.close()
        aimer.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        ros_thread.join(timeout=2.0)
        if log_handle is not None:
            log_handle.close()

    report = {
        "error": error,
        "controller": "sp_vision_25_auto_aim_aimer",
        "tracker": {
            "ready": tracker.ready,
            "reason": tracker.reason,
            "rejections": tracker.rejections,
            "accepted_count": accepted_count,
        },
        "control_loop": None if stats is None else asdict(stats),
        "command_mode": command_mode,
        "commands": state["commands"],
        "last_controller_output": (
            None if state["last_output"] is None
            else {key: json_value(value)
                  for key, value in state["last_output"].items()}
        ),
    }
    print(json.dumps(report, indent=2))
    return 0 if error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
