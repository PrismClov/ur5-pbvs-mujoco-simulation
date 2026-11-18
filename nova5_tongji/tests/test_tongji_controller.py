import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from nova5_tongji.tongji_controller import (
    TongjiControlConfig,
    TongjiController,
    aim_direction,
    blend_limit,
    cartesian_step,
    smooth_ramp_scale,
    smooth_rotation,
)
from nova5_tongji.tongji_direct_tracking import (
    MarkerPoseTracker,
    open_unique_log,
    servo_ack_success,
)
from nova5_visual_servo.core import Fault, exp_so3, log_so3
from nova5_tongji.tracker import TongjiAimer, TongjiTargetTracker
from nova5_visual_servo.target_pose_ekf import TargetTracker
from nova5_visual_servo.robot import URDFRobot


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


class TongjiControllerTests(unittest.TestCase):
    def setUp(self):
        self.model = URDFRobot(REPO / "nova5_visual_servo/models/nova5_robot.urdf")
        self.q = np.array([0.2, -0.4, 0.4, 0.1, 1.2, 0.2])
        self.T_EC = np.array([
            [-0.0052082273, 0.9991615126, -0.0406096803, -0.0480247319],
            [-0.9995613667, -0.0040177570, 0.0293416415, 0.0371031510],
            [0.0291538790, 0.0407446855, 0.9987441724, 0.0275515220],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.aimer = TongjiAimer(
            ROOT / "field_runtime/config/tongji_marker_aimer.yaml"
        )
        self.addCleanup(self.aimer.close)

    def tracker_for_marker(self, marker):
        tracker = TongjiTargetTracker()
        self.addCleanup(tracker.close)
        for index in range(15):
            self.assertTrue(
                tracker.update(
                    index,
                    index * 0.02,
                    self.model.fk(self.q),
                    self.T_EC,
                    marker,
                )
            )
        return tracker

    def controller(self, tracker):
        return TongjiController(
            TongjiControlConfig(),
            tracker,
            self.aimer,
            self.T_EC,
            self.model,
            lambda q: True,
            self.q,
            np.zeros(6),
            tracker.t,
        )

    def test_aligned_camera_is_a_fixed_point(self):
        T_BC = self.model.fk(self.q) @ self.T_EC
        marker = np.eye(4)
        marker[:3, 3] = T_BC[:3, 3] + T_BC[:3, 2] * 0.40
        tracker = self.tracker_for_marker(np.linalg.inv(T_BC) @ marker)
        controller = self.controller(tracker)

        output = controller.tick(
            tracker.t + 0.02, self.q, np.zeros(6), tracker.t + 0.02
        )

        np.testing.assert_allclose(output["q_plan"], self.q, atol=2e-5)
        self.assertLess(output["translation_error_m"], 2e-6)
        self.assertLess(output["rotation_error_rad"], 2e-5)

    def test_lateral_marker_offset_commands_toward_target(self):
        T_BC = self.model.fk(self.q) @ self.T_EC
        local_marker = np.eye(4)
        local_marker[:3, 3] = [0.10, 0.0, 0.40]
        tracker = self.tracker_for_marker(local_marker)
        controller = self.controller(tracker)
        output = controller.tick(
            tracker.t + 0.02, self.q, np.zeros(6), tracker.t + 0.02
        )

        marker_world = T_BC @ local_marker
        desired_forward = output["desired_camera"][:3, 2]
        truth_forward = marker_world[:3, 3] - output["desired_camera"][:3, 3]
        truth_forward /= np.linalg.norm(truth_forward)
        self.assertGreater(np.linalg.norm(output["q_plan"] - self.q), 1e-3)
        np.testing.assert_allclose(desired_forward, truth_forward, atol=2e-6)

    def test_aim_direction_convention(self):
        direction = aim_direction({"yaw": math.pi / 2, "pitch": -math.pi / 4})
        expected = np.array([0.0, math.sqrt(0.5), math.sqrt(0.5)])
        np.testing.assert_allclose(direction, expected, atol=1e-12)

    def test_cartesian_step_is_bounded_by_twist_and_period(self):
        current = np.eye(4)
        target = cartesian_step(
            current,
            np.array([0.2, 0.0, 0.0, 0.0, 0.0, 1.5]),
            0.02,
        )
        np.testing.assert_allclose(target[:3, 3], [0.004, 0.0, 0.0])
        self.assertAlmostEqual(math.acos(np.clip(target[0, 0], -1.0, 1.0)), 0.03)

    def test_rotation_low_pass_limits_a_single_frame_jump(self):
        current = np.eye(3)
        target = exp_so3(np.array([0.0, np.deg2rad(10.0), 0.0]))
        filtered = smooth_rotation(current, target, 0.12)
        angle = np.linalg.norm(log_so3(filtered @ current.T))

        self.assertAlmostEqual(np.rad2deg(angle), 1.2, places=6)
        np.testing.assert_allclose(
            filtered @ filtered.T, np.eye(3), atol=1e-12
        )

    def test_command_ramp_has_zero_slope_at_start_and_end(self):
        self.assertEqual(smooth_ramp_scale(10.0, 10.0, 0.6), 0.0)
        self.assertAlmostEqual(
            smooth_ramp_scale(10.3, 10.0, 0.6), 0.5, places=12
        )
        self.assertEqual(smooth_ramp_scale(10.6, 10.0, 0.6), 1.0)

    def test_moving_target_limits_blend_between_static_and_fast_modes(self):
        self.assertEqual(blend_limit(0.6, 1.1, 0.0), 0.6)
        self.assertAlmostEqual(blend_limit(0.6, 1.1, 0.5), 0.85)
        self.assertEqual(blend_limit(0.6, 1.1, 1.0), 1.1)

    def test_small_joint_command_speed_is_suppressed(self):
        T_BC = self.model.fk(self.q) @ self.T_EC
        marker = np.eye(4)
        marker[:3, 3] = T_BC[:3, 3] + T_BC[:3, 2] * 0.4001
        tracker = self.tracker_for_marker(np.linalg.inv(T_BC) @ marker)
        output = TongjiController(
            TongjiControlConfig(
                min_joint_command_speed_rad_s=0.05,
                reversal_deadband_rad_s=0.05,
                max_joint_velocity_rad_s=0.8,
            ),
            tracker,
            self.aimer,
            self.T_EC,
            self.model,
            lambda q: True,
            self.q,
            np.zeros(6),
            tracker.t,
        ).tick(tracker.t + 0.02, self.q, np.zeros(6), tracker.t + 0.02)

        self.assertTrue(np.all(np.abs(output["qdot_goal"]) < 0.05))

    def test_small_reversal_requires_confirmation_but_large_reversal_is_fast(self):
        controller = object.__new__(TongjiController)
        controller.c = TongjiControlConfig(
            min_joint_command_speed_rad_s=0.01,
            reversal_deadband_rad_s=0.04,
            reversal_confirm_cycles=2,
        )
        controller._goal_sign = np.zeros(6, dtype=int)
        controller._pending_sign = np.zeros(6, dtype=int)
        controller._pending_count = np.zeros(6, dtype=int)

        first = np.zeros(6)
        first[0] = 0.03
        self.assertEqual(controller._shape_joint_goal(first)[0], 0.03)
        self.assertEqual(controller._shape_joint_goal(-first)[0], 0.0)
        self.assertEqual(controller._shape_joint_goal(first)[0], 0.03)
        self.assertEqual(controller._shape_joint_goal(-first)[0], 0.0)
        self.assertEqual(controller._shape_joint_goal(-first)[0], -0.03)

        large = np.zeros(6)
        large[1] = 0.2
        self.assertEqual(controller._shape_joint_goal(large)[1], 0.2)
        large[1] = -0.2
        self.assertEqual(controller._shape_joint_goal(large)[1], -0.2)

    def test_target_hold_hysteresis_stops_submillimeter_searching(self):
        T_BC = self.model.fk(self.q) @ self.T_EC
        marker = np.eye(4)
        marker[:3, 3] = T_BC[:3, 3] + T_BC[:3, 2] * 0.4001
        tracker = self.tracker_for_marker(np.linalg.inv(T_BC) @ marker)
        controller = self.controller(tracker)
        output = controller.tick(
            tracker.t + 0.02, self.q, np.zeros(6), tracker.t + 0.02
        )

        self.assertTrue(output["motion_hold"])
        self.assertEqual(output["motion_boost_scale"], 0.0)
        np.testing.assert_array_equal(output["cartesian_twist"], np.zeros(6))
        fixed_target = output["q_plan"].copy()
        noisy_q = self.q.copy()
        noisy_q[0] += 0.0001
        second = controller.tick(
            tracker.t + 0.04, noisy_q, np.zeros(6), tracker.t + 0.04
        )
        np.testing.assert_array_equal(second["q_plan"], fixed_target)

    def test_servop_acknowledgement_must_report_success(self):
        self.assertTrue(servo_ack_success("0,{42},ServoP(...);"))
        self.assertTrue(servo_ack_success("", allow_empty=True))
        self.assertFalse(servo_ack_success("-2,{},ServoP(...);"))
        self.assertFalse(servo_ack_success("-3,{},ServoP(...);"))
        self.assertFalse(servo_ack_success(""))
        self.assertFalse(servo_ack_success(None))

    def test_observation_log_name_is_unique(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        requested = directory / "observations.jsonl"
        first_path, first = open_unique_log(requested)
        first.close()
        second_path, second = open_unique_log(requested)
        second.close()
        self.assertEqual(first_path, requested)
        self.assertEqual(second_path, directory / "observations.1.jsonl")

    def test_12d_tracker_can_use_four_frame_warmup(self):
        tracker = MarkerPoseTracker(
            max_gap=0.25,
            max_speed=1.0,
            max_omega=2.0,
            warmup_updates=4,
        )
        for index in range(4):
            self.assertTrue(
                tracker.update(
                    index,
                    index * 0.03,
                    np.eye(4),
                    np.eye(4),
                    np.eye(4),
                )
            )
        self.assertTrue(tracker.ready)

    def test_rejects_near_singular_jacobian(self):
        T_BC = self.model.fk(self.q) @ self.T_EC
        marker = np.eye(4)
        marker[:3, 3] = T_BC[:3, 3] + T_BC[:3, 2] * 0.40
        tracker = self.tracker_for_marker(np.linalg.inv(T_BC) @ marker)
        singular_q = self.q.copy()
        singular_q[4] = 0.001
        with self.assertRaisesRegex(Fault, "near singularity"):
            TongjiController(
                TongjiControlConfig(),
                tracker,
                self.aimer,
                self.T_EC,
                self.model,
                lambda q: True,
                singular_q,
                np.zeros(6),
                tracker.t,
            )

    def test_rejects_motion_inside_joint_limit_margin(self):
        T_BC = self.model.fk(self.q) @ self.T_EC
        marker = np.eye(4)
        marker[:3, 3] = T_BC[:3, 3] + T_BC[:3, 2] * 0.40
        tracker = self.tracker_for_marker(np.linalg.inv(T_BC) @ marker)
        near_limit = self.q.copy()
        near_limit[5] = self.model.upper[5] - 0.04
        with self.assertRaisesRegex(Fault, "joint limit margin"):
            TongjiController(
                TongjiControlConfig(joint_limit_margin_rad=0.08),
                tracker,
                self.aimer,
                self.T_EC,
                self.model,
                lambda q: True,
                near_limit,
                np.zeros(6),
                tracker.t,
            )

    def test_marker_plane_alignment_uses_full_pnp_orientation(self):
        T_BC = self.model.fk(self.q) @ self.T_EC
        marker = np.eye(4)
        marker[:3, 3] = [0.0, 0.0, 0.40]
        marker[:3, :3] = np.diag([1.0, -1.0, -1.0])
        tracker = self.tracker_for_marker(marker)
        controller = TongjiController(
            TongjiControlConfig(align_marker_plane=True, align_marker_roll=True),
            tracker,
            self.aimer,
            self.T_EC,
            self.model,
            lambda q: True,
            self.q,
            np.zeros(6),
            tracker.t,
        )

        output = controller.tick(
            tracker.t + 0.02, self.q, np.zeros(6), tracker.t + 0.02
        )

        np.testing.assert_allclose(
            output["desired_camera"][:3, 2],
            T_BC[:3, 2],
            atol=2e-5,
        )
        self.assertLess(output["normal_alignment_error_rad"], 2e-5)

    def test_plane_alignment_chooses_nearest_square_marker_roll(self):
        T_BC = self.model.fk(self.q) @ self.T_EC
        marker = np.eye(4)
        marker[:3, 3] = [0.0, 0.0, 0.40]
        marker[:3, :3] = np.diag([-1.0, 1.0, -1.0])
        tracker = self.tracker_for_marker(marker)
        output = TongjiController(
            TongjiControlConfig(align_marker_plane=True, align_marker_roll=True),
            tracker,
            self.aimer,
            self.T_EC,
            self.model,
            lambda q: True,
            self.q,
            np.zeros(6),
            tracker.t,
        ).tick(tracker.t + 0.02, self.q, np.zeros(6), tracker.t + 0.02)

        current_x = T_BC[:3, 0]
        desired_x = output["desired_camera"][:3, 0]
        self.assertGreaterEqual(current_x @ desired_x, 0.0)

    def test_12d_full_pose_esekf_tracks_upright_marker_fixed_point(self):
        T_BC = self.model.fk(self.q) @ self.T_EC
        marker_in_camera = np.eye(4)
        marker_in_camera[:3, 3] = [0.0, 0.0, 0.40]
        # ArUco +X is image-right, +Y is image-up, and +Z faces the camera.
        marker_in_camera[:3, :3] = np.diag([1.0, -1.0, -1.0])
        tracker = TargetTracker(
            max_gap=0.25,
            max_speed=1.0,
            max_omega=2.0,
        )
        start = 100.0
        for index in range(12):
            self.assertTrue(
                tracker.update(
                    index,
                    start + index * 0.03,
                    self.model.fk(self.q),
                    self.T_EC,
                    marker_in_camera,
                )
            )

        controller = TongjiController(
            TongjiControlConfig(
                align_marker_plane=True,
                upright_marker=True,
            ),
            tracker,
            None,
            self.T_EC,
            self.model,
            lambda q: True,
            self.q,
            np.zeros(6),
            tracker.t,
        )
        output = controller.tick(
            tracker.t + 0.02, self.q, np.zeros(6), tracker.t + 0.02
        )

        np.testing.assert_allclose(
            output["desired_camera"][:3, 2],
            T_BC[:3, 2],
            atol=2e-6,
        )
        np.testing.assert_allclose(
            output["desired_camera"][:3, 0],
            T_BC[:3, 0],
            atol=2e-6,
        )
        self.assertLess(output["translation_error_m"], 2e-6)
        self.assertLess(output["rotation_error_rad"], 2e-5)

    def test_visual_freshness_uses_receive_time_not_exposure_age(self):
        T_BC = self.model.fk(self.q) @ self.T_EC
        marker = np.eye(4)
        marker[:3, 3] = [0.0, 0.0, 0.40]
        marker[:3, :3] = np.diag([1.0, -1.0, -1.0])
        tracker = TargetTracker(
            max_gap=0.25,
            max_speed=1.0,
            max_omega=2.0,
            linear_accel_psd_m2_s3=0.005,
            angular_accel_psd_rad2_s3=0.02,
        )
        start = 400.0
        for index in range(12):
            self.assertTrue(
                tracker.update(
                    index,
                    start + index * 0.03,
                    self.model.fk(self.q),
                    self.T_EC,
                    marker,
                )
            )
        tracker.received_time = tracker.t + 0.10
        controller = TongjiController(
            TongjiControlConfig(
                align_marker_plane=True,
                upright_marker=True,
                max_visual_age_s=0.15,
                max_prediction_s=0.30,
            ),
            tracker,
            None,
            self.T_EC,
            self.model,
            lambda q: True,
            self.q,
            np.zeros(6),
            tracker.t + 0.20,
        )
        output = controller.tick(
            tracker.t + 0.22,
            self.q,
            np.zeros(6),
            tracker.t + 0.22,
        )
        self.assertGreater(output["visual_age"], 0.20)
        self.assertLess(output["visual_receive_age"], 0.13)

    def test_upright_roll_starts_with_large_translation_error(self):
        T_BC = self.model.fk(self.q) @ self.T_EC
        marker_in_camera = np.eye(4)
        marker_in_camera[:3, 3] = [0.10, 0.0, 0.40]
        marker_in_camera[:3, :3] = (
            np.diag([1.0, -1.0, -1.0])
            @ exp_so3(np.array([0.0, 0.0, np.deg2rad(45.0)]))
        )
        tracker = TargetTracker(
            max_gap=0.25, max_speed=1.0, max_omega=2.0,
            innovation_gate=1e9,
        )
        start = 200.0
        for index in range(12):
            self.assertTrue(
                tracker.update(
                    index,
                    start + index * 0.03,
                    self.model.fk(self.q),
                    self.T_EC,
                    marker_in_camera,
                )
            )
        controller = TongjiController(
            TongjiControlConfig(
                align_marker_plane=True,
                upright_marker=True,
            ),
            tracker,
            None,
            self.T_EC,
            self.model,
            lambda q: True,
            self.q,
            np.zeros(6),
            tracker.t,
        )
        output = controller.tick(
            tracker.t + 0.02, self.q, np.zeros(6), tracker.t + 0.02
        )

        self.assertGreater(output["translation_error_m"], 0.05)
        self.assertGreater(np.linalg.norm(output["cartesian_twist"][3:]), 0.1)

    def test_static_target_lock_engages_after_quiet_window(self):
        T_BC = self.model.fk(self.q) @ self.T_EC
        marker = np.eye(4)
        marker[:3, 3] = [0.0, 0.0, 0.40]
        marker[:3, :3] = np.diag([1.0, -1.0, -1.0])
        tracker = TargetTracker(max_gap=0.25, max_speed=1.0, max_omega=2.0)
        start = 300.0
        for index in range(12):
            self.assertTrue(
                tracker.update(
                    index,
                    start + index * 0.03,
                    self.model.fk(self.q),
                    self.T_EC,
                    marker,
                )
            )
        controller = TongjiController(
            TongjiControlConfig(
                align_marker_plane=True,
                upright_marker=True,
                lock_static_target=True,
                static_target_hold_s=0.10,
                max_visual_age_s=0.25,
            ),
            tracker,
            None,
            self.T_EC,
            self.model,
            lambda q: True,
            self.q,
            np.zeros(6),
            tracker.t,
        )
        q = self.q.copy()
        output = None
        for step in range(6):
            now = tracker.t + 0.02 * (step + 1)
            output = controller.tick(now, q, np.zeros(6), now)
            q = output["q_plan"].copy()

        self.assertTrue(output["static_target_locked"])

    def test_conveyor_orientation_lock_keeps_plane_fixed(self):
        T_BC = self.model.fk(self.q) @ self.T_EC
        first_marker = np.eye(4)
        first_marker[:3, 3] = [0.0, 0.0, 0.40]
        first_marker[:3, :3] = np.diag([1.0, -1.0, -1.0])
        tracker = TargetTracker(
            max_gap=0.25, max_speed=1.0, max_omega=2.0,
            innovation_gate=1e9,
        )
        start = 500.0
        for index in range(12):
            self.assertTrue(
                tracker.update(
                    index,
                    start + index * 0.03,
                    self.model.fk(self.q),
                    self.T_EC,
                    first_marker,
                )
            )
        controller = TongjiController(
            TongjiControlConfig(
                align_marker_plane=True,
                upright_marker=True,
                lock_marker_orientation=True,
            ),
            tracker,
            None,
            self.T_EC,
            self.model,
            lambda q: True,
            self.q,
            np.zeros(6),
            tracker.t,
        )
        first = controller.tick(
            tracker.t + 0.02, self.q, np.zeros(6), tracker.t + 0.02
        )
        locked_normal = first["desired_camera"][:3, 2].copy()

        rotated_marker = np.eye(4)
        rotated_marker[:3, 3] = [0.002, 0.0, 0.40]
        rotated_marker[:3, :3] = (
            np.diag([1.0, -1.0, -1.0])
            @ exp_so3(np.array([0.0, np.deg2rad(2.0), 0.0]))
        )
        self.assertTrue(
            tracker.update(
                12, tracker.t + 0.03, self.model.fk(self.q),
                self.T_EC, rotated_marker,
            )
        )
        second = controller.tick(
            tracker.t + 0.02,
            first["q_plan"],
            np.zeros(6),
            tracker.t + 0.02,
        )

        self.assertTrue(second["marker_orientation_locked"])
        np.testing.assert_allclose(
            second["desired_camera"][:3, 2], locked_normal, atol=1e-10
        )

    def test_slow_translation_and_rotation_are_both_tracked(self):
        T_BC = self.model.fk(self.q) @ self.T_EC
        tracker = TargetTracker(
            max_gap=0.25,
            max_speed=1.0,
            max_omega=2.0,
            linear_accel_psd_m2_s3=0.005,
            angular_accel_psd_rad2_s3=0.02,
        )
        start = 600.0
        marker = None
        for index in range(30):
            marker = np.eye(4)
            marker[:3, 3] = [index * 0.002, 0.0, 0.40]
            marker[:3, :3] = (
                np.diag([1.0, -1.0, -1.0])
                @ exp_so3(np.array([0.0, 0.004 * index, 0.0]))
            )
            self.assertTrue(
                tracker.update(
                    index,
                    start + index * 0.03,
                    self.model.fk(self.q),
                    self.T_EC,
                    marker,
                )
            )
        controller = TongjiController(
            TongjiControlConfig(
                align_marker_plane=True,
                upright_marker=True,
            ),
            tracker,
            None,
            self.T_EC,
            self.model,
            lambda q: True,
            self.q,
            np.zeros(6),
            tracker.t,
        )
        output = controller.tick(
            tracker.t + 0.02, self.q, np.zeros(6), tracker.t + 0.02
        )

        marker_normal_world = (T_BC @ marker)[:3, 2]
        expected_direction = -marker_normal_world
        expected_direction /= np.linalg.norm(expected_direction)
        self.assertFalse(output["marker_orientation_locked"])
        self.assertEqual(output["motion_boost_scale"], 1.0)
        self.assertEqual(
            output["effective_limits"]["max_joint_velocity_rad_s"], 1.1
        )
        self.assertGreater(
            np.linalg.norm(output["target_angular_velocity"]), 0.05
        )
        self.assertGreater(
            expected_direction @ output["desired_camera"][:3, 2], 0.99
        )

    def test_small_angular_velocity_is_not_extrapolated(self):
        T_BC = self.model.fk(self.q) @ self.T_EC
        tracker = TargetTracker(max_gap=0.25, max_speed=1.0, max_omega=2.0)
        start = 700.0
        marker = np.eye(4)
        marker[:3, 3] = [0.0, 0.0, 0.40]
        marker[:3, :3] = np.diag([1.0, -1.0, -1.0])
        for index in range(20):
            self.assertTrue(
                tracker.update(
                    index,
                    start + index * 0.03,
                    self.model.fk(self.q),
                    self.T_EC,
                    marker,
                )
            )
        tracker.w = np.array([0.05, 0.0, 0.0])
        controller = TongjiController(
            TongjiControlConfig(
                align_marker_plane=True,
                upright_marker=True,
                angular_prediction_deadband_rad_s=0.15,
            ),
            tracker,
            None,
            self.T_EC,
            self.model,
            lambda q: True,
            self.q,
            np.zeros(6),
            tracker.t,
        )
        output = controller.tick(
            tracker.t + 0.02, self.q, np.zeros(6), tracker.t + 0.02
        )

        self.assertTrue(output["angular_prediction_suppressed"])
        np.testing.assert_array_equal(
            output["control_angular_velocity"], np.zeros(3)
        )


if __name__ == "__main__":
    unittest.main()
