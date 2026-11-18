import threading
import unittest

import numpy as np

from dobot.sdk.interface import DobotInterface, FeedbackItem


class DobotFeedbackUnitsTests(unittest.TestCase):
    def test_feedback_snapshot_exposes_degrees_and_radians_separately(self):
        robot = object.__new__(DobotInterface)
        robot.value_lock = threading.Lock()
        robot.feedData = FeedbackItem()
        robot.feedData.controllerTimestamp = 123456
        robot.feedData.controllerRuntime = 789
        robot.feedData.QTarget = [1.0] * 6
        robot.feedData.QDTarget = [2.0] * 6
        robot.feedData.QDDTarget = [3.0] * 6
        robot.feedData.QActual = [0.0, 90.0, -45.0, 0.0, 0.0, 180.0]
        robot.feedData.QDActual = [4.0] * 6
        robot.feedData.ToolVectorActual = [0.0] * 6
        robot.feedData.ToolVectorTarget = [1.0] * 6
        robot.feedData.TCPSpeedActual = [5.0] * 6
        robot.feedData.TCPSpeedTarget = [6.0] * 6
        robot.feedback_last_monotonic = 12.0
        robot.feedback_count = 3
        robot.feedback_last_duration_s = 0.001

        snapshot = robot.feedback_snapshot()

        self.assertEqual(snapshot["joint_position_deg"][1], 90.0)
        self.assertEqual(snapshot["controller_timestamp"], 123456)
        self.assertEqual(snapshot["controller_runtime"], 789)
        self.assertEqual(snapshot["joint_target_deg"], [1.0] * 6)
        self.assertEqual(snapshot["joint_velocity_target_deg_s"], [2.0] * 6)
        self.assertEqual(
            snapshot["joint_acceleration_target_deg_s2"], [3.0] * 6)
        self.assertEqual(snapshot["joint_velocity_actual_deg_s"], [4.0] * 6)
        self.assertEqual(snapshot["tool_vector_target_mm_deg"], [1.0] * 6)
        self.assertEqual(snapshot["tcp_speed_actual"], [5.0] * 6)
        self.assertEqual(snapshot["tcp_speed_target"], [6.0] * 6)
        np.testing.assert_allclose(
            snapshot["joint_position_rad"],
            np.deg2rad([0.0, 90.0, -45.0, 0.0, 0.0, 180.0]),
        )
        self.assertIsNot(
            snapshot["joint_position_deg"], robot.feedData.QActual
        )


if __name__ == "__main__":
    unittest.main()
