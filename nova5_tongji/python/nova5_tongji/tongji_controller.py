"""Nova5 six-axis adapter driven by the upstream Tongji Aimer.

The upstream Aimer produces yaw/pitch for a pointing axis. This controller
turns that command into a desired camera pose, keeps the requested standoff,
and maps the pose error through the official Nova5 URDF Jacobian.
"""

from dataclasses import dataclass

import numpy as np

from nova5_visual_servo.core import Fault, exp_so3, log_so3, pose, vector


@dataclass(frozen=True)
class TongjiControlConfig:
    standoff_m: float = 0.40
    kp_linear: float = 10.0
    kp_angular: float = 10.0
    damping: float = 0.04
    max_cart_linear_m_s: float = 0.20
    max_cart_angular_rad_s: float = 1.5
    max_joint_velocity_rad_s: float = 0.8
    max_joint_acceleration_rad_s2: float = 5.0
    max_joint_step_rad: float = 0.035
    max_follow_error_rad: float = 0.15
    max_visual_age_s: float = 0.15
    max_feedback_age_s: float = 0.06
    max_prediction_s: float = 0.20
    execution_lead_s: float = 0.03
    bullet_speed_m_s: float = 1000.0
    align_marker_plane: bool = False
    align_marker_roll: bool = False
    upright_marker: bool = False
    normal_smoothing_alpha: float = 0.25
    min_jacobian_singular_value: float = 0.03
    joint_limit_margin_rad: float = 0.08
    twist_smoothing_alpha: float = 0.20
    min_joint_command_speed_rad_s: float = 0.01
    reversal_deadband_rad_s: float = 0.04
    reversal_confirm_cycles: int = 2
    hold_translation_error_m: float = 0.002
    resume_translation_error_m: float = 0.0035
    hold_rotation_error_rad: float = 0.005
    resume_rotation_error_rad: float = 0.010
    command_feedback_gain: float = 0.10
    linear_feedforward_gain: float = 1.0
    angular_feedforward_gain: float = 1.0
    angular_prediction_deadband_rad_s: float = 0.15
    orientation_smoothing_alpha: float = 0.12
    command_ramp_s: float = 0.0
    hold_entry_velocity_rad_s: float = 0.02
    motion_boost_linear_speed_m_s: float = 0.025
    motion_boost_angular_speed_rad_s: float = 0.12
    motion_boost_ramp_s: float = 0.35
    moving_max_cart_linear_m_s: float = 0.30
    moving_max_cart_angular_rad_s: float = 1.50
    moving_max_joint_velocity_rad_s: float = 1.10
    moving_max_joint_acceleration_rad_s2: float = 6.0
    moving_max_joint_step_rad: float = 0.040
    lock_static_target: bool = False
    lock_marker_orientation: bool = False
    static_target_linear_speed_m_s: float = 0.05
    static_target_angular_speed_rad_s: float = 0.30
    static_target_hold_s: float = 0.30
    dt_min_s: float = 0.010
    dt_max_s: float = 0.040

    def __post_init__(self):
        positive = (
            self.standoff_m,
            self.kp_linear,
            self.kp_angular,
            self.damping,
            self.max_cart_linear_m_s,
            self.max_cart_angular_rad_s,
            self.max_joint_velocity_rad_s,
            self.max_joint_acceleration_rad_s2,
            self.max_joint_step_rad,
            self.max_follow_error_rad,
            self.max_visual_age_s,
            self.max_feedback_age_s,
            self.max_prediction_s,
            self.bullet_speed_m_s,
            self.min_jacobian_singular_value,
        )
        if not np.isfinite(positive).all() or min(positive) <= 0.0:
            raise ValueError("all controller limits must be finite and positive")
        if self.execution_lead_s < 0.0 or self.dt_min_s >= self.dt_max_s:
            raise ValueError("invalid execution lead or control period bounds")
        if not 0.0 < self.twist_smoothing_alpha <= 1.0:
            raise ValueError("twist smoothing alpha must be in (0, 1]")
        if not 0.0 < self.normal_smoothing_alpha <= 1.0:
            raise ValueError("normal smoothing alpha must be in (0, 1]")
        if not 0.0 <= self.min_joint_command_speed_rad_s < self.max_joint_velocity_rad_s:
            raise ValueError("invalid joint command speed deadband")
        if not self.min_joint_command_speed_rad_s <= self.reversal_deadband_rad_s:
            raise ValueError("reversal deadband must not be below the static deadband")
        if self.reversal_confirm_cycles < 1:
            raise ValueError("reversal confirmation cycles must be positive")
        if (
            self.hold_translation_error_m <= 0.0
            or self.resume_translation_error_m <= self.hold_translation_error_m
            or self.hold_rotation_error_rad <= 0.0
            or self.resume_rotation_error_rad <= self.hold_rotation_error_rad
        ):
            raise ValueError("invalid target hold hysteresis")
        if not 0.0 < self.command_feedback_gain <= 1.0:
            raise ValueError("command feedback gain must be in (0, 1]")
        if not 0.0 <= self.linear_feedforward_gain <= 1.0:
            raise ValueError("linear feedforward gain must be in [0, 1]")
        if not 0.0 <= self.angular_feedforward_gain <= 1.0:
            raise ValueError("angular feedforward gain must be in [0, 1]")
        if self.angular_prediction_deadband_rad_s < 0.0:
            raise ValueError("angular prediction deadband must be nonnegative")
        if not 0.0 < self.orientation_smoothing_alpha <= 1.0:
            raise ValueError("orientation smoothing alpha must be in (0, 1]")
        if self.command_ramp_s < 0.0:
            raise ValueError("command ramp duration must be nonnegative")
        if not 0.0 < self.hold_entry_velocity_rad_s <= self.max_joint_velocity_rad_s:
            raise ValueError("invalid hold entry velocity")
        moving_positive = (
            self.motion_boost_linear_speed_m_s,
            self.motion_boost_angular_speed_rad_s,
            self.motion_boost_ramp_s,
            self.moving_max_cart_linear_m_s,
            self.moving_max_cart_angular_rad_s,
            self.moving_max_joint_velocity_rad_s,
            self.moving_max_joint_acceleration_rad_s2,
            self.moving_max_joint_step_rad,
        )
        if not np.isfinite(moving_positive).all() or min(moving_positive) <= 0.0:
            raise ValueError("moving-target limits must be finite and positive")
        if (
            self.moving_max_cart_linear_m_s < self.max_cart_linear_m_s
            or self.moving_max_cart_angular_rad_s < self.max_cart_angular_rad_s
            or self.moving_max_joint_velocity_rad_s < self.max_joint_velocity_rad_s
            or self.moving_max_joint_acceleration_rad_s2
            < self.max_joint_acceleration_rad_s2
            or self.moving_max_joint_step_rad < self.max_joint_step_rad
        ):
            raise ValueError("moving-target limits must not be below static limits")
        if (
            self.static_target_linear_speed_m_s <= 0.0
            or self.static_target_angular_speed_rad_s <= 0.0
            or self.static_target_hold_s <= 0.0
        ):
            raise ValueError("invalid static target lock configuration")
        if self.joint_limit_margin_rad < 0.0:
            raise ValueError("joint limit margin must be nonnegative")


def aim_direction(aim):
    """Convert upstream yaw/down-positive-pitch into a base-frame unit vector."""
    yaw = float(aim["yaw"])
    elevation = -float(aim["pitch"])
    cosine = np.cos(elevation)
    direction = np.array(
        [cosine * np.cos(yaw), cosine * np.sin(yaw), np.sin(elevation)],
        dtype=float,
    )
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm < 1e-12:
        raise Fault("Tongji Aimer returned an invalid direction")
    return direction / norm


def desired_camera_pose(
    target_position,
    direction,
    current_rotation,
    standoff_m,
    preferred_x=None,
):
    """Build a level camera pose whose +Z optical axis points at the target."""
    target_position = vector(target_position, 3)
    direction = vector(direction, 3)
    direction /= np.linalg.norm(direction)
    current_rotation = pose(np.block(
        [[current_rotation, np.zeros((3, 1))], [np.zeros((1, 3)), 1.0]]
    ))[:3, :3]

    reference_x = (
        current_rotation[:, 0]
        if preferred_x is None
        else vector(preferred_x, 3)
    )
    x_axis = reference_x - (reference_x @ direction) * direction
    if np.linalg.norm(x_axis) < 1e-6:
        up = np.array([0.0, 0.0, 1.0])
        x_axis = np.cross(up, direction)
        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.array([1.0, 0.0, 0.0])
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(direction, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    result = np.eye(4)
    result[:3, 0] = x_axis
    result[:3, 1] = y_axis
    result[:3, 2] = direction
    result[:3, 3] = target_position - direction * standoff_m
    return pose(result)


def cartesian_step(current_pose, twist, dt):
    """Apply a bounded Cartesian velocity for one control period."""
    current = pose(current_pose)
    twist = vector(twist, 6)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive")
    result = np.eye(4)
    result[:3, 3] = current[:3, 3] + twist[:3] * dt
    result[:3, :3] = exp_so3(twist[3:] * dt) @ current[:3, :3]
    return pose(result)


def smooth_rotation(current, target, alpha):
    """Low-pass a rotation while preserving the shortest rotation branch."""
    current = pose(np.block(
        [[current, np.zeros((3, 1))], [np.zeros((1, 3)), 1.0]]
    ))[:3, :3]
    target = pose(np.block(
        [[target, np.zeros((3, 1))], [np.zeros((1, 3)), 1.0]]
    ))[:3, :3]
    step = log_so3(target @ current.T)
    return exp_so3(float(alpha) * step) @ current


def smooth_ramp_scale(now, start_time, duration):
    """Return a zero-slope command ramp from 0 to 1."""
    if duration <= 0.0:
        return 1.0
    progress = float(np.clip((now - start_time) / duration, 0.0, 1.0))
    return progress * progress * (3.0 - 2.0 * progress)


def blend_limit(static_limit, moving_limit, scale):
    """Blend between stationary and moving-target controller limits."""
    scale = float(np.clip(scale, 0.0, 1.0))
    return float(static_limit + scale * (moving_limit - static_limit))


class TongjiController:
    """Single-owner controller; callers remain responsible for the hardware send."""

    def __init__(
        self,
        config,
        tracker,
        aimer,
        T_EC,
        model,
        workspace_ok,
        q_initial,
        qd_initial,
        start_time,
        joint_lower=None,
        joint_upper=None,
    ):
        self.c = config
        self.tracker = tracker
        self.aimer = aimer
        self.T_EC = pose(T_EC)
        self.model = model
        self.workspace_ok = workspace_ok
        self.q_plan = vector(q_initial, 6).copy()
        self.v = vector(qd_initial, 6).copy()
        self.smoothed_twist = None
        self._goal_sign = np.zeros(6, dtype=int)
        self._pending_sign = np.zeros(6, dtype=int)
        self._pending_count = np.zeros(6, dtype=int)
        self.motion_hold = False
        self.q_hold_target = None
        self.filtered_marker_normal = None
        self.static_target_pose = None
        self.static_target_since = None
        self.locked_marker_rotation = None
        self.filtered_marker_rotation = None
        self.command_start_time = float(start_time)
        initial_speed = float(np.linalg.norm(
            np.asarray(getattr(tracker, "v", np.zeros(3)), dtype=float)
        ))
        initial_angular_speed = float(np.linalg.norm(
            np.asarray(getattr(tracker, "w", np.zeros(3)), dtype=float)
        ))
        self.motion_boost_scale = float(
            initial_speed >= self.c.motion_boost_linear_speed_m_s
            or initial_angular_speed >= self.c.motion_boost_angular_speed_rad_s
        )
        self.last_time = float(start_time)
        self.joint_lower = np.asarray(
            model.lower if joint_lower is None else joint_lower, dtype=float
        )
        self.joint_upper = np.asarray(
            model.upper if joint_upper is None else joint_upper, dtype=float
        )
        if (
            self.joint_lower.shape != (6,)
            or self.joint_upper.shape != (6,)
            or np.any(self.joint_lower >= self.joint_upper)
            or np.any(abs(self.v) > self.c.max_joint_velocity_rad_s)
        ):
            raise ValueError("invalid initial joint state")
        self._check_jacobian_condition(self.q_plan)
        self._check_joint_and_workspace(self.q_plan)
        self.faulted = False

    def tick(self, now, q_actual, qd_actual, feedback_sample_time):
        if self.faulted:
            raise Fault("Tongji controller fault latched")
        try:
            return self._tick(now, q_actual, qd_actual, feedback_sample_time)
        except Exception:
            self.faulted = True
            raise

    def _tick(self, now, q_actual, qd_actual, feedback_sample_time):
        c = self.c
        q = vector(q_actual, 6)
        qd = vector(qd_actual, 6)
        self._check_jacobian_condition(q)
        dt = float(now) - self.last_time
        if not c.dt_min_s <= dt <= c.dt_max_s:
            raise Fault(f"control period {dt*1000:.3f} ms outside bounds; no catch-up")
        if not 0.0 <= now - feedback_sample_time <= c.max_feedback_age_s:
            raise Fault("stale joint feedback")
        receive_time = getattr(self.tracker, "received_time", self.tracker.t)
        if self.tracker.t is None or not 0.0 <= now - receive_time <= c.max_visual_age_s:
            raise Fault("stale vision")
        if np.any(abs(self.q_plan - q) > c.max_follow_error_rad):
            raise Fault("command-feedback divergence")

        execution_time = now + c.execution_lead_s
        if execution_time - self.tracker.t > c.max_prediction_s:
            raise Fault("Tongji prediction horizon exceeded")
        estimate = self.tracker.predict_state(execution_time, c.max_prediction_s)
        if not estimate["usable"]:
            raise Fault("Tongji target estimate is not usable")
        estimate_pose = pose(
            estimate.get("pose", estimate.get("T"))
        )
        if self.filtered_marker_rotation is None:
            self.filtered_marker_rotation = estimate_pose[:3, :3].copy()
        else:
            self.filtered_marker_rotation = smooth_rotation(
                self.filtered_marker_rotation,
                estimate_pose[:3, :3],
                c.orientation_smoothing_alpha,
            )
        estimate_pose = estimate_pose.copy()
        estimate_pose[:3, :3] = self.filtered_marker_rotation
        if c.lock_marker_orientation:
            if self.locked_marker_rotation is None:
                self.locked_marker_rotation = estimate_pose[:3, :3].copy()
            else:
                estimate_pose = estimate_pose.copy()
                estimate_pose[:3, :3] = self.locked_marker_rotation
        estimate_position = np.asarray(
            estimate.get("position", estimate_pose[:3, 3]),
            dtype=float,
        )
        estimate_velocity = np.asarray(
            estimate.get("velocity", np.zeros(3)), dtype=float
        )
        estimate_angular_velocity = np.asarray(
            estimate.get("angular_velocity", np.zeros(3)), dtype=float
        )
        if c.lock_marker_orientation:
            estimate_angular_velocity = np.zeros(3)
        moving_target = (
            np.linalg.norm(estimate_velocity)
            >= c.motion_boost_linear_speed_m_s
            or np.linalg.norm(estimate_angular_velocity)
            >= c.motion_boost_angular_speed_rad_s
        )
        boost_target = 1.0 if moving_target else 0.0
        boost_step = dt / c.motion_boost_ramp_s
        self.motion_boost_scale += float(np.clip(
            boost_target - self.motion_boost_scale,
            -boost_step,
            boost_step,
        ))
        self.motion_boost_scale = float(np.clip(
            self.motion_boost_scale, 0.0, 1.0
        ))
        max_cart_linear = blend_limit(
            c.max_cart_linear_m_s,
            c.moving_max_cart_linear_m_s,
            self.motion_boost_scale,
        )
        max_cart_angular = blend_limit(
            c.max_cart_angular_rad_s,
            c.moving_max_cart_angular_rad_s,
            self.motion_boost_scale,
        )
        max_joint_velocity = blend_limit(
            c.max_joint_velocity_rad_s,
            c.moving_max_joint_velocity_rad_s,
            self.motion_boost_scale,
        )
        max_joint_acceleration = blend_limit(
            c.max_joint_acceleration_rad_s2,
            c.moving_max_joint_acceleration_rad_s2,
            self.motion_boost_scale,
        )
        max_joint_step = blend_limit(
            c.max_joint_step_rad,
            c.moving_max_joint_step_rad,
            self.motion_boost_scale,
        )
        if c.lock_static_target:
            static_now = (
                np.linalg.norm(estimate_velocity)
                < c.static_target_linear_speed_m_s
                and np.linalg.norm(estimate_angular_velocity)
                < c.static_target_angular_speed_rad_s
            )
            if static_now:
                if self.static_target_since is None:
                    self.static_target_since = now
                    self.static_target_pose = estimate_pose.copy()
                if now - self.static_target_since >= c.static_target_hold_s:
                    estimate_pose = self.static_target_pose.copy()
                    estimate_position = estimate_pose[:3, 3].copy()
                    estimate_velocity = np.zeros(3)
                    estimate_angular_velocity = np.zeros(3)
            else:
                self.static_target_since = None
                self.static_target_pose = None
        control_angular_velocity = estimate_angular_velocity.copy()
        angular_prediction_suppressed = (
            not c.lock_marker_orientation
            and np.linalg.norm(estimate_angular_velocity)
            < c.angular_prediction_deadband_rad_s
        )
        if angular_prediction_suppressed:
            current_estimate = self.tracker.predict_state(
                self.tracker.t, c.max_prediction_s
            )
            current_pose = pose(
                current_estimate.get(
                    "pose", current_estimate.get("T")
                )
            )
            estimate_pose = estimate_pose.copy()
            estimate_pose[:3, :3] = current_pose[:3, :3]
            control_angular_velocity[:] = 0.0

        robot_horizon = execution_time - feedback_sample_time
        if not 0.0 <= robot_horizon <= c.max_prediction_s:
            raise Fault("robot prediction horizon exceeded")
        q_predicted = q + qd * robot_horizon
        T_BE_predicted = pose(self.model.fk(q_predicted))
        T_BC_predicted = pose(T_BE_predicted @ self.T_EC)

        aim = None
        if self.aimer is None:
            if not c.align_marker_plane:
                raise Fault("marker-plane tracking requires a plane normal")
            offset = estimate_position - T_BC_predicted[:3, 3]
            distance = np.linalg.norm(offset)
            if not np.isfinite(distance) or distance < 1e-6:
                raise Fault("invalid camera-to-marker distance")
            center_direction = offset / distance
        else:
            aim = self.aimer.aim(
                self.tracker,
                execution_time,
                T_BC_predicted[:3, 3],
                bullet_speed=c.bullet_speed_m_s,
            )
            if aim is None:
                raise Fault("Tongji Aimer rejected the target")
            center_direction = aim_direction(aim)
        direction = center_direction
        preferred_x = None
        normal_alignment_error = 0.0
        marker_normal = None
        if c.align_marker_plane:
            marker_rotation = estimate_pose[:3, :3]
            raw_normal = marker_rotation[:, 2].copy()
            to_camera = T_BC_predicted[:3, 3] - estimate_position
            if raw_normal @ to_camera < 0.0:
                raw_normal *= -1.0
            if self.filtered_marker_normal is None:
                self.filtered_marker_normal = raw_normal
            else:
                if self.filtered_marker_normal @ raw_normal < 0.0:
                    raw_normal *= -1.0
                normal_alpha = c.normal_smoothing_alpha
                marker_normal = (
                    normal_alpha * raw_normal
                    + (1.0 - normal_alpha) * self.filtered_marker_normal
                )
                marker_normal /= np.linalg.norm(marker_normal)
                self.filtered_marker_normal = marker_normal.copy()
            marker_normal = self.filtered_marker_normal.copy()
            normal_direction = -marker_normal
            direction = normal_direction / np.linalg.norm(normal_direction)
            if c.upright_marker:
                # Decoded ArUco corner 0 -> 1 points along marker +X.
                preferred_x = marker_rotation[:, 0]
            elif c.align_marker_roll:
                marker_x = marker_rotation[:, 0]
                current_x = T_BC_predicted[:3, 0] - (
                    T_BC_predicted[:3, 0] @ direction
                ) * direction
                if np.linalg.norm(current_x) < 1e-6:
                    preferred_x = marker_x
                else:
                    # A square target has a 180-degree roll ambiguity. Pick
                    # the in-plane marker axis closest to the current roll.
                    preferred_x = (
                        marker_x
                        if marker_x @ current_x >= 0.0
                        else -marker_x
                    )
            normal_alignment_error = float(np.arccos(np.clip(
                center_direction @ direction, -1.0, 1.0
            )))
        T_BC_desired = desired_camera_pose(
            estimate_position,
            direction,
            T_BC_predicted[:3, :3],
            c.standoff_m,
            preferred_x,
        )
        T_BE_desired = pose(T_BC_desired @ np.linalg.inv(self.T_EC))

        translation_error = T_BE_desired[:3, 3] - T_BE_predicted[:3, 3]
        rotation_error = log_so3(
            T_BE_desired[:3, :3] @ T_BE_predicted[:3, :3].T
        )
        error = np.r_[translation_error, rotation_error]
        inside_hold = (
            np.linalg.norm(translation_error) <= c.hold_translation_error_m
            and np.linalg.norm(rotation_error) <= c.hold_rotation_error_rad
        )
        outside_resume = (
            np.linalg.norm(translation_error) >= c.resume_translation_error_m
            or np.linalg.norm(rotation_error) >= c.resume_rotation_error_rad
        )
        if self.motion_hold:
            if outside_resume:
                self.motion_hold = False
                self.q_hold_target = None
        stopping_for_hold = inside_hold and not self.motion_hold
        twist = np.zeros(6)
        if not self.motion_hold and not stopping_for_hold:
            twist[3:] = c.kp_angular * rotation_error
            if c.upright_marker:
                twist[3:] += (
                    c.angular_feedforward_gain * control_angular_velocity
                )
            angular_speed = np.linalg.norm(twist[3:])
            if angular_speed > max_cart_angular:
                twist[3:] *= max_cart_angular / angular_speed
            if c.upright_marker:
                # Control translation at the optical center, then express the
                # requested velocity at the flange origin for its geometric Jacobian.
                optical_axis = T_BC_predicted[:3, 2]
                camera_to_center = estimate_position - T_BC_predicted[:3, 3]
                reference = c.standoff_m * optical_axis
                camera_error = camera_to_center - reference
                lever = T_BC_predicted[:3, 3] - T_BE_predicted[:3, 3]
                twist[:3] = (c.kp_linear * camera_error
                             + c.linear_feedforward_gain * estimate_velocity
                             - np.cross(twist[3:], reference + lever))
            else:
                twist[:3] = (c.kp_linear * translation_error
                             + c.linear_feedforward_gain * estimate_velocity)
            linear_speed = np.linalg.norm(twist[:3])
            if linear_speed > max_cart_linear:
                twist[:3] *= max_cart_linear / linear_speed
        if self.smoothed_twist is None:
            self.smoothed_twist = twist.copy()
        else:
            alpha = c.twist_smoothing_alpha
            self.smoothed_twist = (
                alpha * twist + (1.0 - alpha) * self.smoothed_twist
            )
        twist = self.smoothed_twist
        twist = twist * smooth_ramp_scale(
            now, self.command_start_time, c.command_ramp_s
        )
        if stopping_for_hold:
            twist = np.zeros(6)
            self.smoothed_twist = np.zeros(6)

        jacobian = np.asarray(self.model.jac(q_predicted), dtype=float)
        if jacobian.shape != (6, 6) or not np.isfinite(jacobian).all():
            raise Fault("invalid Nova5 Jacobian")
        goal = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + c.damping**2 * np.eye(6),
            twist,
        )
        goal = np.clip(
            goal, -max_joint_velocity, max_joint_velocity
        )
        goal = self._shape_joint_goal(goal)
        if stopping_for_hold:
            goal = np.zeros(6)
        if self.motion_hold:
            velocity_next = np.zeros(6)
            q_next = self.q_hold_target.copy()
        else:
            velocity_next = self.v + np.clip(
                goal - self.v,
                -max_joint_acceleration * dt,
                max_joint_acceleration * dt,
            )
            # Continue a smooth command trajectory, with only a small
            # correction from encoder feedback. The previous implementation
            # rebuilt every target from raw feedback and injected sensor noise
            # directly into ServoJ.
            q_next = (
                self.q_plan
                + 0.5 * (self.v + velocity_next) * dt
                + c.command_feedback_gain * (q - self.q_plan)
            )
            q_next = self.q_plan + np.clip(
                q_next - self.q_plan,
                -max_joint_step,
                max_joint_step,
            )
            if (
                stopping_for_hold
                and np.linalg.norm(velocity_next)
                <= c.hold_entry_velocity_rad_s
            ):
                velocity_next = np.zeros(6)
                self.motion_hold = True
                self.q_hold_target = q_next.copy()
        for candidate in (q_predicted, q_next):
            self._check_joint_and_workspace(candidate)
        self._check_jacobian_condition(q_next)

        self.q_plan = q_next
        self.v = velocity_next
        self.last_time = float(now)
        return {
            "q_plan": q_next.copy(),
            "v_next": velocity_next.copy(),
            "qdot_goal": goal.copy(),
            "error": error,
            "translation_error_m": float(np.linalg.norm(translation_error)),
            "rotation_error_rad": float(np.linalg.norm(rotation_error)),
            "desired_camera": T_BC_desired,
            "desired_end": T_BE_desired,
            "aim": aim,
            "aim_direction": direction,
            "center_aim_direction": center_direction,
            "marker_normal": marker_normal,
            "normal_alignment_error_rad": normal_alignment_error,
            "motion_hold": self.motion_hold,
            "static_target_locked": bool(
                self.static_target_pose is not None
                and self.static_target_since is not None
                and now - self.static_target_since >= c.static_target_hold_s
            ),
            "marker_orientation_locked": bool(
                self.locked_marker_rotation is not None
            ),
            "cartesian_twist": twist,
            "target_position": estimate_position.copy(),
            "target_velocity": estimate_velocity.copy(),
            "target_angular_velocity": estimate_angular_velocity.copy(),
            "control_angular_velocity": control_angular_velocity.copy(),
            "angular_prediction_suppressed": angular_prediction_suppressed,
            "motion_boost_scale": self.motion_boost_scale,
            "effective_limits": {
                "max_cart_linear_m_s": max_cart_linear,
                "max_cart_angular_rad_s": max_cart_angular,
                "max_joint_velocity_rad_s": max_joint_velocity,
                "max_joint_acceleration_rad_s2": max_joint_acceleration,
                "max_joint_step_rad": max_joint_step,
            },
            "execution_time": execution_time,
            "visual_age": now - self.tracker.t,
            "visual_receive_age": now - receive_time,
            "feedback_age": now - feedback_sample_time,
            "velocity_limited": bool(
                np.any(abs(goal - jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + c.damping**2 * np.eye(6),
                    twist,
                )) > 1e-12)
            ),
        }

    def _check_jacobian_condition(self, q):
        singular_values = np.linalg.svd(
            np.asarray(self.model.jac(q), dtype=float),
            compute_uv=False,
        )
        if (
            not np.isfinite(singular_values).all()
            or singular_values[-1] < self.c.min_jacobian_singular_value
        ):
            raise Fault(
                "Nova5 Jacobian near singularity: "
                f"min_sv={singular_values[-1]:.5f}"
            )

    def _check_joint_and_workspace(self, q):
        q = np.asarray(q, dtype=float)
        if not np.isfinite(q).all():
            raise Fault("joint target is non-finite")
        lower = self.joint_lower + self.c.joint_limit_margin_rad
        upper = self.joint_upper - self.c.joint_limit_margin_rad
        violated = np.flatnonzero((q < lower) | (q > upper))
        if len(violated):
            details = []
            for index in violated:
                name = self.model.names[index]
                details.append(
                    f"{name}={np.rad2deg(q[index]):.2f} deg "
                    f"allowed[{np.rad2deg(lower[index]):.2f},"
                    f"{np.rad2deg(upper[index]):.2f}]"
                )
            raise Fault("joint limit margin: " + "; ".join(details))
        if not self.workspace_ok(q):
            raise Fault("TCP position outside configured workspace")

    def _shape_joint_goal(self, goal):
        c = self.c
        goal = np.asarray(goal, dtype=float).copy()
        for index, value in enumerate(goal):
            magnitude = abs(value)
            if magnitude < c.min_joint_command_speed_rad_s:
                goal[index] = 0.0
                continue
            sign = 1 if value > 0.0 else -1
            previous = int(self._goal_sign[index])
            if previous == 0 or sign == previous:
                self._goal_sign[index] = sign
                self._pending_sign[index] = 0
                self._pending_count[index] = 0
                continue
            if magnitude >= c.reversal_deadband_rad_s:
                self._goal_sign[index] = sign
                self._pending_sign[index] = 0
                self._pending_count[index] = 0
                continue
            if self._pending_sign[index] != sign:
                self._pending_sign[index] = sign
                self._pending_count[index] = 1
            else:
                self._pending_count[index] += 1
            if self._pending_count[index] < c.reversal_confirm_cycles:
                goal[index] = 0.0
            else:
                self._goal_sign[index] = sign
                self._pending_sign[index] = 0
                self._pending_count[index] = 0
        return goal
