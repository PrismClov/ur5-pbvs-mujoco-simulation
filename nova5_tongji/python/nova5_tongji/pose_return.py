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
    "target loss deceleration timeout",
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


def hermite_return_position(before, target, initial_velocity, duration, elapsed):
    """Quintic bridge with matched initial velocity and zero terminal motion."""
    progress = float(np.clip(elapsed / duration, 0.0, 1.0))
    p2 = progress * progress
    p3 = p2 * progress
    p4 = p3 * progress
    p5 = p4 * progress
    h00 = 1.0 - 10.0 * p3 + 15.0 * p4 - 6.0 * p5
    h10 = progress - 6.0 * p3 + 8.0 * p4 - 3.0 * p5
    h01 = 10.0 * p3 - 15.0 * p4 + 6.0 * p5
    return (
        before * h00
        + initial_velocity * duration * h10
        + target * h01
    )


def hermite_return_limits(
    before, target, initial_velocity, duration
):
    """Return peak absolute joint velocity and acceleration of the bridge."""
    progress = np.linspace(0.0, 1.0, 201)
    p2 = progress * progress
    p3 = p2 * progress
    p4 = p3 * progress
    h00 = 1.0 - 10.0 * p3 + 15.0 * p4 - 6.0 * progress**5
    h10 = progress - 6.0 * p3 + 8.0 * p4 - 3.0 * progress**5
    h01 = 10.0 * p3 - 15.0 * p4 + 6.0 * progress**5
    dh00 = -30.0 * p2 + 60.0 * p3 - 30.0 * p4
    dh10 = 1.0 - 18.0 * p2 + 32.0 * p3 - 15.0 * p4
    dh01 = 30.0 * p2 - 60.0 * p3 + 30.0 * p4
    ddh00 = -60.0 * progress + 180.0 * p2 - 120.0 * p3
    ddh10 = -36.0 * progress + 96.0 * p2 - 60.0 * p3
    ddh01 = 60.0 * progress - 180.0 * p2 + 120.0 * p3
    velocity = (
        before[None, :] * dh00[:, None]
        + initial_velocity[None, :] * duration * dh10[:, None]
        + target[None, :] * dh01[:, None]
    ) / duration
    acceleration = (
        before[None, :] * ddh00[:, None]
        + initial_velocity[None, :] * duration * ddh10[:, None]
        + target[None, :] * ddh01[:, None]
    ) / duration**2
    return (
        float(np.max(np.abs(velocity))),
        float(np.max(np.abs(acceleration))),
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
    initial_velocity_rad_s=None,
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
    initial_velocity = np.zeros(6)
    if initial_velocity_rad_s is not None:
        initial_velocity = np.asarray(initial_velocity_rad_s, dtype=float)
        if initial_velocity.shape != (6,) or not np.isfinite(initial_velocity).all():
            raise ValueError("initial velocity must contain six finite values")
    duration = return_duration(
        float(np.max(np.abs(target - before))),
        max_speed_rad_s,
        max_acceleration_rad_s2,
    )
    speed_budget = max(
        max_speed_rad_s,
        float(np.max(np.abs(initial_velocity))),
    )
    for _ in range(12):
        peak_speed, peak_acceleration = hermite_return_limits(
            before, target, initial_velocity, duration
        )
        if (
            peak_speed <= speed_budget + 1e-9
            and peak_acceleration <= max_acceleration_rad_s2 + 1e-9
        ):
            break
        duration *= 1.25
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
        command_target = hermite_return_position(
            before, target, initial_velocity, duration, elapsed
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
                    "initial_velocity_speed_budget_rad_s": speed_budget,
                    "initial_joint_velocity_rad_s": (
                        initial_velocity.tolist()
                    ),
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
