import unittest

import numpy as np

from nova5_tongji.pose_return import (
    is_target_loss_reason,
    return_duration,
    return_to_joint_pose,
    should_return_to_initial,
)


class PoseReturnTests(unittest.TestCase):
    def test_target_loss_reasons_cover_vision_and_estimator_faults(self):
        self.assertTrue(is_target_loss_reason("Fault: stale vision"))
        self.assertTrue(
            is_target_loss_reason(
                "RuntimeError: no usable target on /tag_tracker/pose_camera"
            )
        )
        self.assertTrue(
            is_target_loss_reason(
                "Fault: Tongji target estimate is not usable"
            )
        )
        self.assertFalse(
            is_target_loss_reason("Fault: ServoJ rejected at command 10")
        )
        self.assertTrue(
            should_return_to_initial(
                "Fault: Nova5 Jacobian near singularity: min_sv=0.02955"
            )
        )
        self.assertTrue(should_return_to_initial("Fault: joint limit margin"))
        self.assertTrue(
            should_return_to_initial(
                "Fault: estimated target speed outside device configuration"
            )
        )
        self.assertTrue(
            should_return_to_initial(
                "Fault: target loss deceleration timeout"
            )
        )
        self.assertFalse(
            should_return_to_initial("Fault: ServoJ rejected at command 10")
        )
        self.assertFalse(
            should_return_to_initial("Fault: stale joint feedback")
        )

    def test_return_duration_accounts_for_speed_and_acceleration(self):
        duration = return_duration(0.5, 0.2, 0.6)
        self.assertGreaterEqual(duration, 0.5)
        self.assertGreaterEqual(duration, 1.875 * 0.5 / 0.2)
        self.assertGreaterEqual(duration, np.sqrt(5.7735026919 * 0.5 / 0.6))

    def test_return_uses_smooth_joint_trajectory(self):
        class FakeTransport:
            def __init__(self):
                self.position = np.zeros(6)
                self.commands = []
                self.servo_send_last_result = "0,{1},ServoJ(...);"

            def get_joint_position(self):
                return self.position.tolist()

            def send_joint_target(self, command):
                command = np.asarray(command, dtype=float)
                self.commands.append(command.copy())
                self.position = command.copy()
                return command.tolist()

        transport = FakeTransport()
        target = np.full(6, 0.05)
        result = return_to_joint_pose(
            transport,
            target,
            control_hz=100.0,
            max_speed_rad_s=1.0,
            max_acceleration_rad_s2=10.0,
            tolerance_rad=0.001,
            settle_s=0.01,
            timeout_s=2.0,
        )

        commands = np.asarray(transport.commands)
        self.assertTrue(result["success"])
        self.assertLess(result["final_feedback_error_rad"], 0.001)
        np.testing.assert_allclose(commands[0], np.zeros(6), atol=1e-12)
        np.testing.assert_allclose(commands[-1], target, atol=1e-12)
        steps = np.linalg.norm(np.diff(commands, axis=0), axis=1)
        self.assertLess(steps[0], steps.max())
        self.assertLess(steps[-1], steps.max())

    def test_return_matches_nonzero_initial_joint_velocity(self):
        class FakeTransport:
            def __init__(self):
                self.position = np.zeros(6)
                self.commands = []
                self.servo_send_last_result = "0,{1},ServoJ(...);"

            def get_joint_position(self):
                return self.position.tolist()

            def send_joint_target(self, command):
                command = np.asarray(command, dtype=float)
                self.commands.append(command.copy())
                self.position = command.copy()
                return command.tolist()

        transport = FakeTransport()
        result = return_to_joint_pose(
            transport,
            np.full(6, -0.08),
            control_hz=100.0,
            max_speed_rad_s=0.6,
            max_acceleration_rad_s2=2.0,
            tolerance_rad=0.001,
            settle_s=0.01,
            timeout_s=4.0,
            initial_velocity_rad_s=np.full(6, 0.2),
        )

        commands = np.asarray(transport.commands)
        self.assertTrue(result["success"])
        self.assertGreater(commands[1, 0], commands[0, 0])
        np.testing.assert_allclose(commands[-1], np.full(6, -0.08), atol=1e-12)

    def test_fast_target_loss_decelerates_before_return(self):
        class FakeTransport:
            def __init__(self):
                self.position = np.zeros(6)
                self.commands = []
                self.servo_send_last_result = "0,{1},ServoJ(...);"

            def get_joint_position(self):
                return self.position.tolist()

            def send_joint_target(self, command):
                command = np.asarray(command, dtype=float)
                self.commands.append(command.copy())
                self.position = command.copy()
                return command.tolist()

        transport = FakeTransport()
        result = return_to_joint_pose(
            transport,
            np.zeros(6),
            control_hz=100.0,
            max_speed_rad_s=0.2,
            max_acceleration_rad_s2=2.0,
            tolerance_rad=0.001,
            settle_s=0.01,
            timeout_s=5.0,
            initial_velocity_rad_s=np.full(6, 1.0),
        )

        commands = np.asarray(transport.commands)
        velocities = np.diff(commands, axis=0) * 100.0
        self.assertTrue(result["success"])
        self.assertGreater(commands[1, 0], 0.0)
        self.assertLessEqual(np.max(np.abs(velocities)), 1.1)
        self.assertAlmostEqual(np.max(np.abs(velocities[-1])), 0.0, places=2)


if __name__ == "__main__":
    unittest.main()
