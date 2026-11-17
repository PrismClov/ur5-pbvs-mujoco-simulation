import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_recenter_gradient.py"
)
SPEC = importlib.util.spec_from_file_location("run_recenter_gradient", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RecenterGradientTests(unittest.TestCase):
    def test_marker_center_and_principal_error(self):
        detection = {
            "valid": True,
            "corners": [
                [218.0, 100.0],
                [222.0, 100.0],
                [222.0, 120.0],
                [218.0, 120.0],
            ],
        }
        self.assertEqual(
            MODULE.marker_center(detection), (220.0, 110.0))
        self.assertEqual(
            MODULE.center_error(detection, (210.0, 110.0)), 10.0)

    def test_latency_uses_first_detection_in_tolerance_band(self):
        samples = [
            {"elapsed_s": 0.00, "error_px": 20.0},
            {"elapsed_s": 0.07, "error_px": 12.0},
            {"elapsed_s": 0.13, "error_px": 4.5},
            {"elapsed_s": 0.20, "error_px": 3.0},
            {"elapsed_s": 0.27, "error_px": 2.5},
            {"elapsed_s": 0.34, "error_px": 2.0},
            {"elapsed_s": 0.41, "error_px": 1.8},
        ]
        result = MODULE.latency_metrics(samples, 5.0)
        self.assertAlmostEqual(result["recovery_entry_s"], 0.13)
        self.assertTrue(result["meets_200ms"])
        self.assertAlmostEqual(result["error_at_100ms_px"], 8.25)
        self.assertAlmostEqual(result["error_at_200ms_px"], 3.0)
        self.assertEqual(result["recovery_hold_200ms_s"], 0.13)

    def test_hold_requirement_rejects_late_reentry(self):
        samples = [
            {"elapsed_s": 0.00, "error_px": 20.0},
            {"elapsed_s": 0.10, "error_px": 15.0},
            {"elapsed_s": 0.20, "error_px": 10.0},
            {"elapsed_s": 0.30, "error_px": 4.0},
            {"elapsed_s": 0.40, "error_px": 3.0},
            {"elapsed_s": 0.50, "error_px": 2.0},
        ]
        result = MODULE.latency_metrics(samples, 5.0)
        self.assertAlmostEqual(result["recovery_entry_s"], 0.30)
        self.assertFalse(result["meets_200ms"])
        self.assertAlmostEqual(result["recovery_hold_200ms_s"], 0.30)


if __name__ == "__main__":
    unittest.main()
