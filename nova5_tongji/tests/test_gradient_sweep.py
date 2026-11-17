import importlib.util
from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_gradient_sweep.py"
)
SPEC = importlib.util.spec_from_file_location("run_gradient_sweep", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GradientSweepTests(unittest.TestCase):
    def test_gradient_ranges_are_inclusive(self):
        self.assertEqual(MODULE.parse_levels("1:2:0.5"), [1.0, 1.5, 2.0])
        self.assertEqual(MODULE.parse_levels("0.5:1:0.5"), [0.5, 1.0])

    def test_comma_levels_preserve_order(self):
        self.assertEqual(MODULE.parse_levels("4,1,2"), [4.0, 1.0, 2.0])

    def test_rejects_invalid_or_nonpositive_levels(self):
        with self.assertRaises(Exception):
            MODULE.parse_levels("4:1:1")
        with self.assertRaises(Exception):
            MODULE.parse_levels("0")

    def test_summary_uses_unique_command_events(self):
        samples = [
            {
                "executor_state": "EXECUTING",
                "commands_sent_this_start": 1,
                "last_command": {
                    "stamp_sec": 10.0,
                    "feature_rms_px": 20.0,
                    "adaptive_gain_active": True,
                    "effective_gain": 4.0,
                    "effective_max_linear_speed": 0.06,
                    "target_center_velocity_px_s": 10.0,
                },
                "servop_timing_summary": {},
            },
            {
                "executor_state": "EXECUTING",
                "commands_sent_this_start": 1,
                "last_command": {
                    "stamp_sec": 10.0,
                    "feature_rms_px": 20.0,
                    "adaptive_gain_active": True,
                    "effective_gain": 4.0,
                    "effective_max_linear_speed": 0.06,
                    "target_center_velocity_px_s": 10.0,
                },
                "servop_timing_summary": {},
            },
            {
                "executor_state": "EXECUTING",
                "commands_sent_this_start": 3,
                "last_command": {
                    "stamp_sec": 11.0,
                    "feature_rms_px": 4.0,
                    "adaptive_gain_active": False,
                    "effective_gain": 2.0,
                    "effective_max_linear_speed": 0.05,
                    "target_center_velocity_px_s": 30.0,
                },
                "max_abs_joint_speed_rad_s": 0.3,
                "cumulative_translation_mm": 12.0,
                "cumulative_rotation_deg": 0.4,
                "servop_timing_summary": {
                    "service_latency": {
                        "mean_ms": 39.0,
                        "p95_ms": 42.0,
                        "max_ms": 45.0,
                    },
                    "send_period": {
                        "mean_ms": 80.0,
                        "p95_ms": 84.0,
                        "max_ms": 90.0,
                    },
                },
            },
        ]
        summary = MODULE.summarize_status_samples(
            samples, elapsed_s=4.0, tolerance_px=5.0)
        self.assertEqual(summary["command_events"], 2)
        self.assertEqual(summary["initial_error_px"], 20.0)
        self.assertEqual(summary["final_error_px"], 4.0)
        self.assertEqual(summary["error_reduction_percent"], 80.0)
        self.assertTrue(summary["within_tolerance"])
        self.assertEqual(summary["error_source"], "commands")
        self.assertEqual(summary["command_rate_hz"], 0.75)
        self.assertEqual(summary["max_joint_speed_rad_s"], 0.3)
        self.assertEqual(summary["adaptive_events"], 1)
        self.assertEqual(summary["adaptive_active_fraction"], 0.5)
        self.assertEqual(summary["effective_gain_max"], 4.0)
        self.assertEqual(summary["effective_max_linear_speed_max"], 0.06)

    def test_summary_falls_back_to_diagnostics_in_deadzone(self):
        summary = MODULE.summarize_status_samples(
            [{
                "executor_state": "EXECUTING",
                "last_control_diagnostics": {
                    "stamp_sec": 12.0,
                    "feature_rms_px": 1.7,
                },
            }],
            elapsed_s=1.0,
            tolerance_px=5.0,
        )
        self.assertEqual(summary["error_source"], "diagnostics")
        self.assertEqual(summary["final_error_px"], 1.7)
        self.assertTrue(summary["within_tolerance"])

    def test_summary_reports_fault_reason(self):
        summary = MODULE.summarize_status_samples(
            [{"executor_state": "FAULT", "reason": "service_timeout"}],
            elapsed_s=1.0,
            tolerance_px=5.0,
        )
        self.assertEqual(summary["fault_reasons"], ["service_timeout"])
        self.assertFalse(summary["within_tolerance"])


if __name__ == "__main__":
    unittest.main()
