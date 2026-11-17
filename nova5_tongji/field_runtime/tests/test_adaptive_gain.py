import importlib.util
from pathlib import Path
import sys
import unittest


PBVS = (
    Path(__file__).resolve().parents[1] / "mujoco" / "pbvs"
)
sys.path.insert(0, str(PBVS))
SPEC = importlib.util.spec_from_file_location(
    "nova5_low_speed_executor",
    PBVS / "nova5_low_speed_executor.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def parameters(**overrides):
    values = {
        "enabled": True,
        "feature_rms_px": 30.0,
        "target_speed_px_s": 10.0,
        "target_speed_available": True,
        "base_gain": 2.0,
        "boosted_gain": 6.0,
        "base_max_linear_speed": 0.05,
        "boosted_max_linear_speed": 0.08,
        "large_error_start_px": 12.0,
        "large_error_full_px": 30.0,
        "slow_target_full_px_s": 20.0,
        "slow_target_zero_px_s": 60.0,
    }
    values.update(overrides)
    return MODULE.adaptive_gain_parameters(**values)


class AdaptiveGainTests(unittest.TestCase):
    def test_large_slow_error_uses_full_boost(self):
        result = parameters()
        self.assertTrue(result["active"])
        self.assertAlmostEqual(result["blend"], 1.0)
        self.assertAlmostEqual(result["gain"], 6.0)
        self.assertAlmostEqual(result["max_linear_speed"], 0.08)

    def test_small_or_fast_target_keeps_base_gain(self):
        for overrides in (
            {"feature_rms_px": 5.0},
            {"target_speed_px_s": 80.0},
            {"target_speed_available": False},
            {"enabled": False},
        ):
            result = parameters(**overrides)
            self.assertFalse(result["active"])
            self.assertAlmostEqual(result["gain"], 2.0)
            self.assertAlmostEqual(result["max_linear_speed"], 0.05)

    def test_transition_is_continuous(self):
        middle = parameters(feature_rms_px=21.0, target_speed_px_s=40.0)
        self.assertGreater(middle["blend"], 0.0)
        self.assertLess(middle["blend"], 1.0)
        self.assertGreater(middle["gain"], 2.0)
        self.assertLess(middle["gain"], 6.0)

    def test_planned_joint_speed_is_scaled_before_servoj(self):
        current = [0.0] * 6
        target = [0.0, 0.0, 30.0, 0.0, 0.0, 0.0]
        joints, requested, limited = MODULE.limit_planned_joint_step(
            current, target, 0.1, 100.0)
        self.assertTrue(limited)
        self.assertAlmostEqual(requested, 300.0)
        self.assertAlmostEqual(joints[2], 10.0)

    def test_planned_joint_speed_keeps_safe_target(self):
        current = [0.0] * 6
        target = [0.0, 0.0, 5.0, 0.0, 0.0, 0.0]
        joints, requested, limited = MODULE.limit_planned_joint_step(
            current, target, 0.1, 100.0)
        self.assertFalse(limited)
        self.assertAlmostEqual(requested, 50.0)
        self.assertEqual(joints, target)


if __name__ == "__main__":
    unittest.main()
