"""Bounded joint-space return used after a tracking target is lost."""

from __future__ import annotations

import time

import numpy as np


TARGET_LOSS_REASONS = (
    "stale vision",
    "no usable target",
    "target estimate is not usable",
    "prediction horizon exceeded",
    "target prediction horizon invalid",
    "accepted observation gap exceeded",
)

RETURN_FAULT_REASONS = TARGET_LOSS_REASONS + (
    "jacobian near singularity",
    "joint limit margin",
    "joint target is non-finite",
    "invalid nova5 jacobian",
    "tcp position outside configured workspace",
    "estimated target speed outside device configuration",
)


def is_target_loss_reason(message):
    """Return whether a controller failure means the visual target is unavailable."""
    text = str(message or "").lower()
    return any(reason in text for reason in TARGET_LOSS_REASONS)


def should_return_to_initial(message):
    """Return after target loss or a recoverable kinematics safety stop."""
    text = str(message or "").lower()
    return any(reason in text for reason in RETURN_FAULT_REASONS)


def servo_ack_ok(acknowledgement):
    if not acknowledgement:
        return True
    try:
        return int(str(acknowledgement).strip().split(",", 1)[0]) == 0
    except ValueError:
        return False


def smoothstep(progress):
    progress = float(np.clip(progress, 0.0, 1.0))
    return progress * progress * progress * (
        10.0 + progress * (-15.0 + 6.0 * progress)
    )


def return_duration(max_delta_rad, max_speed_rad_s, max_acceleration_rad_s2):
    """Minimum quintic duration for the requested speed and acceleration."""
    max_delta_rad = float(max_delta_rad)
    if max_delta_rad <= 0.0:
        return 0.5
    duration_for_speed = 1.875 * max_delta_rad / max_speed_rad_s
    duration_for_acceleration = np.sqrt(
        5.7735026919 * max_delta_rad / max_acceleration_rad_s2
    )
    return max(
        0.5,
        float(duration_for_speed),
        float(duration_for_acceleration),
    )


def return_to_joint_pose(
    transport,
    target_joint_position,
    *,
    control_hz=50.0,
    max_speed_rad_s=0.2,
    max_acceleration_rad_s2=0.6,
    tolerance_rad=0.00873,
    settle_s=0.5,
    timeout_s=20.0,
):
    """Move to a target joint pose with zero endpoint velocity and acceleration."""
    target = np.asarray(target_joint_position, dtype=float)
    if target.shape != (6,) or not np.isfinite(target).all():
        raise ValueError("return target must contain six finite joint angles")
    if control_hz <= 0.0:
        raise ValueError("control_hz must be positive")
    if (
        max_speed_rad_s <= 0.0
        or max_acceleration_rad_s2 <= 0.0
        or tolerance_rad <= 0.0
        or settle_s < 0.0
        or timeout_s <= 0.0
    ):
        raise ValueError("invalid return trajectory limits")

    current = transport.get_joint_position()
    if current is None:
        raise RuntimeError("joint feedback unavailable before return")
    before = np.asarray(current, dtype=float)
    duration = return_duration(
        float(np.max(np.abs(target - before))),
        max_speed_rad_s,
        max_acceleration_rad_s2,
    )
    period = 1.0 / control_hz
    started = time.monotonic()
    deadline = started + min(
        timeout_s,
        duration + settle_s + 5.0,
    )
    deadline = min(timeout_s + started, max(deadline, started + duration + settle_s))
    next_tick = started
    settled_since = None
    samples = 0
    max_feedback_error = 0.0
    last_command_error = None

    while time.monotonic() < deadline:
        remaining = next_tick - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)
        current = transport.get_joint_position()
        if current is None:
            raise RuntimeError("joint feedback unavailable during return")
        current = np.asarray(current, dtype=float)
        elapsed = time.monotonic() - started
        command_target = before + (target - before) * smoothstep(
            elapsed / duration
        )
        command = np.asarray(
            transport.send_joint_target(command_target), dtype=float
        )
        if not servo_ack_ok(transport.servo_send_last_result):
            raise RuntimeError(
                f"ServoJ return command rejected: "
                f"{transport.servo_send_last_result}"
            )
        feedback_error = float(np.max(np.abs(current - target)))
        max_feedback_error = max(max_feedback_error, feedback_error)
        last_command_error = float(np.max(np.abs(command - target)))
        samples += 1

        now = time.monotonic()
        if elapsed >= duration and feedback_error <= tolerance_rad:
            if settled_since is None:
                settled_since = now
            elif now - settled_since >= settle_s:
                return {
                    "success": True,
                    "started_joint_position_rad": before.tolist(),
                    "target_joint_position_rad": target.tolist(),
                    "final_joint_position_rad": current.tolist(),
                    "duration_s": elapsed,
                    "trajectory_duration_s": duration,
                    "samples": samples,
                    "max_feedback_error_rad": max_feedback_error,
                    "last_command_error_rad": last_command_error,
                    "final_feedback_error_rad": feedback_error,
                }
        else:
            settled_since = None
        next_tick += period
        now = time.monotonic()
        if next_tick <= now:
            next_tick += (
                int((now - next_tick) / period) + 1
            ) * period

    raise TimeoutError(
        f"target pose return exceeded {deadline - started:.3f} s"
    )
