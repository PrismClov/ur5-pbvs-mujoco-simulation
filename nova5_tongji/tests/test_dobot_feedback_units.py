import threading
import unittest

import numpy as np

from dobot.sdk.interface import DobotInterface, FeedbackItem


class DobotFeedbackUnitsTests(unittest.TestCase):
    def test_feedback_snapshot_exposes_degrees_and_radians_separately(self):
        robot = object.__new__(DobotInterface)
        robot.value_lock = threading.Lock()
        robot.feedData = FeedbackItem()
        robot.feedData.QActual = [0.0, 90.0, -45.0, 0.0, 0.0, 180.0]
        robot.feedData.ToolVectorActual = [0.0] * 6
        robot.feedback_last_monotonic = 12.0
        robot.feedback_count = 3
        robot.feedback_last_duration_s = 0.001

        snapshot = robot.feedback_snapshot()

        self.assertEqual(snapshot["joint_position_deg"][1], 90.0)
        np.testing.assert_allclose(
            snapshot["joint_position_rad"],
            np.deg2rad([0.0, 90.0, -45.0, 0.0, 0.0, 180.0]),
        )
        self.assertIsNot(
            snapshot["joint_position_deg"], robot.feedData.QActual
        )


if __name__ == "__main__":
    unittest.main()
