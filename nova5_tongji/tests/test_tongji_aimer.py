import math
from pathlib import Path
import unittest

import numpy as np

from nova5_tongji.tracker import TongjiAimer, TongjiTargetTracker


ROOT = Path(__file__).resolve().parents[1]


class TongjiAimerTests(unittest.TestCase):
    def test_original_aimer_yaw_pitch_for_marker_offset(self):
        tracker = TongjiTargetTracker()
        aimer = TongjiAimer(ROOT / "field_runtime/config/tongji_marker_aimer.yaml")
        self.addCleanup(aimer.close)
        self.addCleanup(tracker.close)
        marker = np.eye(4)
        marker[:3, 3] = [0.2, 0.3, 0.4]
        for index in range(15):
            self.assertTrue(
                tracker.update(index, index * 0.02, np.eye(4), np.eye(4), marker)
            )

        result = aimer.aim(tracker, tracker.t, [0.0, 0.0, 0.0])

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["yaw"], math.atan2(0.3, 0.2), places=6)
        elevation = math.atan2(0.4, math.hypot(0.2, 0.3))
        self.assertAlmostEqual(result["pitch"], -elevation, places=5)
        np.testing.assert_allclose(result["target_xyz"], marker[:3, 3])
        np.testing.assert_allclose(result["upstream_armor_xyz"], marker[:3, 3])

    def test_aimer_uses_current_camera_origin(self):
        tracker = TongjiTargetTracker()
        aimer = TongjiAimer(ROOT / "field_runtime/config/tongji_marker_aimer.yaml")
        self.addCleanup(aimer.close)
        self.addCleanup(tracker.close)
        marker = np.eye(4)
        marker[:3, 3] = [0.4, 0.0, 0.4]
        for index in range(15):
            self.assertTrue(
                tracker.update(index, index * 0.02, np.eye(4), np.eye(4), marker)
            )

        result = aimer.aim(tracker, tracker.t, [0.2, 0.0, 0.0])

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["yaw"], 0.0, places=6)
        self.assertAlmostEqual(
            result["pitch"], -math.atan2(0.4, 0.2), places=5
        )
        np.testing.assert_allclose(result["target_xyz"], [0.2, 0.0, 0.4])


if __name__ == "__main__":
    unittest.main()
