import math
from pathlib import Path
import unittest

import numpy as np

from nova5_tongji.tongji_controller import (
    TongjiControlConfig,
    TongjiController,
    aim_direction,
    cartesian_step,
)
from nova5_tongji.tongji_direct_tracking import servo_ack_success
from nova5_visual_servo.core import Fault
from nova5_tongji.tracker import TongjiAimer, TongjiTargetTracker
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


if __name__ == "__main__":
    unittest.main()
