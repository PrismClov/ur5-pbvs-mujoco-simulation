#!/usr/bin/env python3
"""Run bounded Nova5/ArUco control-parameter sweeps.

This script orchestrates one short tracking trial per parameter combination.
It expects the Dobot driver, D435i camera and real_tag_tracker.py to already
be running. An existing nova5_low_speed_executor.py must be stopped first.

Without --execute the script only prints the trial plan. With --execute it
starts each executor itself, arms and starts tracking through the existing
services, records executor_status samples, stops the trial, and writes JSONL,
JSON and CSV results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Iterable


def parse_levels(text: str) -> list[float]:
    """Parse ``1,2,4`` or inclusive ``1:4:1`` into positive values."""
    pieces = [part.strip() for part in text.split(",") if part.strip()]
    values: list[float] = []
    for piece in pieces:
        if ":" not in piece:
            values.append(float(piece))
            continue
        fields = piece.split(":")
        if len(fields) != 3:
            raise argparse.ArgumentTypeError(
                f"gradient range must be start:stop:step, got {piece!r}")
        start, stop, step = (float(field) for field in fields)
        if step <= 0.0 or stop < start:
            raise argparse.ArgumentTypeError(
                f"invalid gradient range {piece!r}")
        count = int(math.floor((stop - start) / step + 1e-9))
        values.extend(start + index * step for index in range(count + 1))
    if not values:
        raise argparse.ArgumentTypeError("parameter levels cannot be empty")
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("parameter levels must be positive")
    return [float(value) for value in values]


def percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(value))
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize_status_samples(
    samples: list[dict[str, Any]],
    *,
    elapsed_s: float,
    tolerance_px: float,
) -> dict[str, Any]:
    commands: dict[float, dict[str, Any]] = {}
    diagnostics: dict[float, dict[str, Any]] = {}
    for sample in samples:
        command = sample.get("last_command")
        if isinstance(command, dict) and command:
            try:
                stamp = float(command.get("stamp_sec"))
            except (TypeError, ValueError):
                pass
            else:
                commands[stamp] = command
        diagnostic = sample.get("last_control_diagnostics")
        if isinstance(diagnostic, dict) and diagnostic:
            try:
                stamp = float(diagnostic.get("stamp_sec"))
            except (TypeError, ValueError):
                pass
            else:
                diagnostics[stamp] = diagnostic

    command_rows = [commands[stamp] for stamp in sorted(commands)]
    diagnostic_rows = [diagnostics[stamp] for stamp in sorted(diagnostics)]
    errors = []
    for command in command_rows:
        try:
            value = float(command.get("feature_rms_px"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            errors.append(value)
    error_source = "commands"
    if not errors:
        error_source = "diagnostics"
        for diagnostic in diagnostic_rows:
            try:
                value = float(diagnostic.get("feature_rms_px"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                errors.append(value)

    initial_error = errors[0] if errors else None
    final_error = errors[-1] if errors else None
    reduction = None
    if initial_error not in (None, 0.0) and final_error is not None:
        reduction = 100.0 * (initial_error - final_error) / initial_error

    status_nodes = [
        sample for sample in samples
        if sample.get("executor_state") != "FAULT"
    ]
    final_status = status_nodes[-1] if status_nodes else (
        samples[-1] if samples else {})
    faults = sorted({
        str(sample.get("reason"))
        for sample in samples
        if sample.get("executor_state") == "FAULT"
    })
    state_counts: dict[str, int] = {}
    for sample in samples:
        state = str(sample.get("executor_state") or "UNKNOWN")
        state_counts[state] = state_counts.get(state, 0) + 1

    timing = final_status.get("servop_timing_summary") or {}
    latency = timing.get("service_latency") or {}
    send_period = timing.get("send_period") or {}
    command_count = int(final_status.get("commands_sent_this_start") or 0)
    adaptive_rows = command_rows if command_rows else diagnostic_rows
    effective_gains = []
    effective_speed_caps = []
    target_speeds = []
    adaptive_events = 0
    for row in adaptive_rows:
        if bool(row.get("adaptive_gain_active")):
            adaptive_events += 1
        for key, values in (
            ("effective_gain", effective_gains),
            ("effective_max_linear_speed", effective_speed_caps),
            ("target_center_velocity_px_s", target_speeds),
        ):
            try:
                value = float(row.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
    return {
        "samples": len(samples),
        "command_events": len(command_rows),
        "diagnostic_events": len(diagnostic_rows),
        "commands_sent": command_count,
        "command_rate_hz": (
            command_count / elapsed_s if elapsed_s > 0.0 else None),
        "initial_error_px": initial_error,
        "final_error_px": final_error,
        "error_reduction_percent": reduction,
        "error_mean_px": (
            statistics.fmean(errors) if errors else None),
        "error_p95_px": percentile(errors, 0.95),
        "error_max_px": max(errors) if errors else None,
        "error_source": error_source if errors else None,
        "adaptive_events": adaptive_events,
        "adaptive_active_fraction": (
            adaptive_events / len(adaptive_rows) if adaptive_rows else None),
        "effective_gain_mean": (
            statistics.fmean(effective_gains)
            if effective_gains else None),
        "effective_gain_max": (
            max(effective_gains) if effective_gains else None),
        "effective_max_linear_speed_max": (
            max(effective_speed_caps)
            if effective_speed_caps else None),
        "target_speed_mean_px_s": (
            statistics.fmean(target_speeds)
            if target_speeds else None),
        "within_tolerance": (
            final_error is not None and final_error <= tolerance_px),
        "service_latency_mean_ms": latency.get("mean_ms"),
        "service_latency_p95_ms": latency.get("p95_ms"),
        "service_latency_max_ms": latency.get("max_ms"),
        "send_period_mean_ms": send_period.get("mean_ms"),
        "send_period_p95_ms": send_period.get("p95_ms"),
        "send_period_max_ms": send_period.get("max_ms"),
        "max_joint_speed_rad_s": max(
            [
                float(sample.get("max_abs_joint_speed_rad_s") or 0.0)
                for sample in samples
            ] or [0.0]),
        "cumulative_translation_mm": final_status.get(
            "cumulative_translation_mm"),
        "cumulative_rotation_deg": final_status.get(
            "cumulative_rotation_deg"),
        "final_state": final_status.get("executor_state"),
        "final_reason": final_status.get("reason"),
        "fault_reasons": faults,
        "state_counts": state_counts,
    }


def trial_command(args: argparse.Namespace, trial: dict[str, float],
                  output_log: Path) -> list[str]:
    return [
        "ros2", "run", args.executor_package, args.executor_executable,
        "--config", str(args.config),
        "--desired-file", str(args.desired_file),
        "--min-markers", "0",
        "--min-charuco-corners", "0",
        "--min-features", "4",
        "--min-feature-fraction", "1.0",
        "--min-velocity-features", "4",
        "--execute",
        "--output-log", str(output_log),
        "--gain", f"{trial['gain']:.6f}",
        "--large-error-gain",
        f"{trial['large_error_gain']:.6f}",
        "--damping", f"{args.damping:.6f}",
        "--angular-gain", f"{trial['angular_gain']:.6f}",
        "--max-linear-speed", f"{args.max_linear_speed:.6f}",
        "--large-error-max-linear-speed",
        f"{args.large_error_max_linear_speed:.6f}",
        "--large-error-start-px", f"{args.large_error_start_px:.6f}",
        "--large-error-full-px", f"{args.large_error_full_px:.6f}",
        "--slow-target-full-px-s", f"{args.slow_target_full_px_s:.6f}",
        "--slow-target-zero-px-s", f"{args.slow_target_zero_px_s:.6f}",
        "--max-angular-speed-deg", f"{args.max_angular_speed_deg:.6f}",
        "--max-step-rotation-deg", f"{args.max_step_rotation_deg:.6f}",
        "--command-duration", f"{args.command_duration:.6f}",
        "--command-settle-margin", f"{args.command_settle_margin:.6f}",
        "--servo-aheadtime", f"{args.servo_aheadtime:.6f}",
        "--servo-gain", f"{args.servo_gain:.6f}",
        "--speed-factor", str(int(trial["speed_factor"])),
        "--max-cumulative-translation-mm",
        f"{args.max_cumulative_translation_mm:.6f}",
        "--max-cumulative-rotation-deg",
        f"{args.max_cumulative_rotation_deg:.6f}",
        "--max-runtime-sec", f"{args.duration + 30.0:.6f}",
        "--adaptive-gain" if args.adaptive_gain else "--no-adaptive-gain",
        "--command-mode", args.command_mode,
        (
            "--use-tool-pose-directly"
            if args.use_tool_pose_directly
            else "--no-use-tool-pose-directly"
        ),
    ]


class RosHarness:
    def __init__(self) -> None:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
        from std_srvs.srv import SetBool

        self.rclpy = rclpy
        self.SetBool = SetBool
        rclpy.init()
        self.node = Node("nova5_gradient_sweep")
        self.samples: list[dict[str, Any]] = []
        self.latest: dict[str, Any] | None = None
        self.node.create_subscription(
            String, "/nova5_ibvs/executor_status", self._status, 10)
        self.arm = self.node.create_client(
            SetBool, "/nova5_ibvs/arm_execution")
        self.start = self.node.create_client(
            SetBool, "/nova5_ibvs/start_execution")

    def _status(self, message: Any) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        self.latest = payload
        self.samples.append(payload)

    def spin_for(self, duration_s: float) -> None:
        deadline = time.monotonic() + max(0.0, duration_s)
        while time.monotonic() < deadline and self.rclpy.ok():
            self.rclpy.spin_once(
                self.node,
                timeout_sec=min(0.05, max(0.0, deadline - time.monotonic())),
            )

    def wait_for(
        self,
        predicate: Callable[[dict[str, Any] | None], bool],
        timeout_s: float,
        label: str,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and self.rclpy.ok():
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            if predicate(self.latest):
                assert self.latest is not None
                return self.latest
        raise TimeoutError(f"timed out waiting for {label}")

    def wait_services(self, timeout_s: float = 10.0) -> None:
        for name, client in (("arm", self.arm), ("start", self.start)):
            if not client.wait_for_service(timeout_sec=timeout_s):
                raise RuntimeError(f"{name} service did not become available")

    def service_present(self, timeout_s: float = 0.5) -> bool:
        return self.arm.wait_for_service(timeout_sec=timeout_s)

    def call_bool(self, client: Any, value: bool) -> Any:
        request = self.SetBool.Request()
        request.data = value
        future = client.call_async(request)
        deadline = time.monotonic() + 5.0
        while not future.done() and time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
        if not future.done():
            raise TimeoutError("service call timed out")
        return future.result()

    def stop(self) -> None:
        if self.start.service_is_ready():
            try:
                self.call_bool(self.start, False)
            except Exception:
                pass
        self.node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()


def terminate_process(process: subprocess.Popen[str], log_path: Path) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=8.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=3.0)
    if log_path.exists():
        lines = log_path.read_text(errors="replace").splitlines()
        if lines:
            print("  executor tail:", " | ".join(lines[-3:]))


def prompt_reset(enabled: bool, index: int, total: int) -> None:
    if not enabled:
        return
    if not sys.stdin.isatty():
        raise RuntimeError(
            "--prompt-between-runs requires an interactive terminal")
    print(
        f"\nTrial {index}/{total}: move the marker to the same starting "
        "offset, verify the workspace is clear, then press Enter.")
    input()


def write_outputs(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    if not rows:
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    temporary = output_dir / "summary.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            rendered = {}
            for key in columns:
                value = row.get(key)
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)
                rendered[key] = value
            writer.writerow(rendered)
    temporary.replace(output_dir / "summary.csv")


def run_sweep(args: argparse.Namespace) -> int:
    if not args.execute:
        print("Dry run only. Add --execute to start bounded motion trials.")
        print_trials(args)
        return 0

    harness = RosHarness()
    current_process: subprocess.Popen[str] | None = None
    current_log: Path | None = None
    rows: list[dict[str, Any]] = []
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        if harness.service_present(0.5):
            raise RuntimeError(
                "an executor is already running; stop it before sweeping")
        print_trials(args)
        trials = list(iter_trials(args))
        for index, trial in enumerate(trials, 1):
            prompt_reset(args.prompt_between_runs, index, len(trials))
            trial_dir = output_dir / f"trial_{index:03d}"
            trial_dir.mkdir()
            output_log = trial_dir / "tracking.jsonl"
            current_log = trial_dir / "executor.stdout.log"
            command = trial_command(args, trial, output_log)
            (trial_dir / "command.json").write_text(
                json.dumps(
                    {"parameters": trial, "argv": command},
                    indent=2,
                    ensure_ascii=False,
                ) + "\n")
            print(
                f"\nStarting trial {index}/{len(trials)}: "
                f"gain={trial['gain']:.3f}, "
                f"large_error_gain={trial['large_error_gain']:.3f}, "
                f"angular_gain={trial['angular_gain']:.3f}, "
                f"speed_factor={trial['speed_factor']:.0f}%")
            stream = current_log.open("w")
            current_process = subprocess.Popen(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            stream.close()
            harness.samples.clear()
            harness.latest = None
            harness.wait_services()
            armed = harness.call_bool(harness.arm, True)
            if not armed.success:
                raise RuntimeError(f"arm rejected: {armed.message}")
            harness.wait_for(
                lambda sample: bool(
                    sample and sample.get("executor_state") == "READY"),
                timeout_s=8.0,
                label="executor READY",
            )
            started = harness.call_bool(harness.start, True)
            if not started.success:
                raise RuntimeError(f"start rejected: {started.message}")
            harness.samples.clear()
            started_at = time.monotonic()
            harness.spin_for(args.duration)
            elapsed = time.monotonic() - started_at
            stopped = harness.call_bool(harness.start, False)
            if not stopped.success:
                print(f"  warning: stop response: {stopped.message}")

            summary = summarize_status_samples(
                harness.samples,
                elapsed_s=elapsed,
                tolerance_px=args.tolerance_px,
            )
            row = {
                "trial": index,
                "gain": trial["gain"],
                "large_error_gain": trial["large_error_gain"],
                "angular_gain": trial["angular_gain"],
                "speed_factor": trial["speed_factor"],
                "damping": args.damping,
                "adaptive_gain": args.adaptive_gain,
                "duration_s": elapsed,
                **summary,
                "output_dir": str(trial_dir),
            }
            rows.append(row)
            (trial_dir / "summary.json").write_text(
                json.dumps(row, indent=2, ensure_ascii=False) + "\n")
            write_outputs(output_dir, rows)
            print(
                "  result: commands={commands_sent}, "
                "error mean={error_mean_px}, p95={error_p95_px}, "
                "final={final_error_px}, faults={fault_reasons}".format(
                    **row))
            terminate_process(current_process, current_log)
            current_process = None
            current_log = None
            if index < len(trials):
                harness.spin_for(args.cooldown)
                deadline = time.monotonic() + 8.0
                while (
                    harness.service_present(0.2)
                    and time.monotonic() < deadline
                ):
                    harness.spin_for(0.1)
        print(f"\nSweep complete. Results: {output_dir / 'summary.csv'}")
        return 0
    finally:
        try:
            if harness.start.service_is_ready():
                harness.call_bool(harness.start, False)
        except Exception:
            pass
        if current_process is not None and current_log is not None:
            terminate_process(current_process, current_log)
        harness.stop()


def iter_trials(args: argparse.Namespace) -> Iterable[dict[str, float]]:
    for gain in args.gains:
        for large_error_gain in args.large_error_gains:
            for angular_gain in args.angular_gains:
                for speed_factor in args.speed_factors:
                    yield {
                        "gain": float(gain),
                        "large_error_gain": float(large_error_gain),
                        "angular_gain": float(angular_gain),
                        "speed_factor": float(speed_factor),
                    }


def print_trials(args: argparse.Namespace) -> None:
    trials = list(iter_trials(args))
    print(
        f"Created {len(trials)} trial(s), "
        f"{args.duration:.1f} s each, tolerance {args.tolerance_px:.1f} px:")
    for index, trial in enumerate(trials, 1):
        print(
            f"  {index:03d}: gain={trial['gain']:g}, "
            f"large_error_gain={trial['large_error_gain']:g}, "
            f"angular_gain={trial['angular_gain']:g}, "
            f"speed_factor={trial['speed_factor']:g}%")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--gains", type=parse_levels, default=parse_levels("1:4:1"))
    parser.add_argument(
        "--large-error-gains", type=parse_levels,
        default=parse_levels("6"))
    parser.add_argument(
        "--angular-gains", type=parse_levels,
        default=parse_levels("0.5:1:0.5"))
    parser.add_argument(
        "--speed-factors", type=parse_levels, default=parse_levels("30"))
    parser.add_argument("--damping", type=float, default=0.4)
    parser.add_argument(
        "--adaptive-gain",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Increase gain for large image error while target image "
            "velocity is low."
        ),
    )
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--cooldown", type=float, default=1.5)
    parser.add_argument("--tolerance-px", type=float, default=5.0)
    parser.add_argument(
        "--config", type=Path,
        default=Path("/home/clover/ur_ws/ibvs_rm65/config/"
                     "nova5_fast_execution.yaml"))
    parser.add_argument(
        "--desired-file", type=Path,
        default=Path("/home/clover/ur_ws/ibvs_rm65/runtime/"
                     "nova5_aruco_center_040m.json"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--executor-package", default="ibvs_ur5")
    parser.add_argument(
        "--executor-executable",
        default="nova5_low_speed_executor.py")
    parser.add_argument("--max-linear-speed", type=float, default=0.05)
    parser.add_argument(
        "--large-error-max-linear-speed", type=float, default=0.08)
    parser.add_argument("--large-error-start-px", type=float, default=12.0)
    parser.add_argument("--large-error-full-px", type=float, default=30.0)
    parser.add_argument("--slow-target-full-px-s", type=float, default=20.0)
    parser.add_argument("--slow-target-zero-px-s", type=float, default=60.0)
    parser.add_argument("--max-angular-speed-deg", type=float, default=5.0)
    parser.add_argument("--max-step-rotation-deg", type=float, default=0.2)
    parser.add_argument("--command-duration", type=float, default=0.03)
    parser.add_argument("--command-settle-margin", type=float, default=0.02)
    parser.add_argument(
        "--command-mode",
        choices=("servo_j", "servo_p", "servo_p_stream", "mov_l"),
        default="servo_j",
    )
    parser.add_argument(
        "--use-tool-pose-directly",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--servo-aheadtime", type=float, default=50.0)
    parser.add_argument("--servo-gain", type=float, default=500.0)
    parser.add_argument(
        "--max-cumulative-translation-mm", type=float, default=50.0)
    parser.add_argument(
        "--max-cumulative-rotation-deg", type=float, default=5.0)
    parser.add_argument(
        "--prompt-between-runs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pause before every trial so the marker starts at the same pose.")
    args = parser.parse_args()
    if any(value > 100.0 for value in args.speed_factors):
        parser.error("speed factors must be in [1, 100]")
    if any(value < max(args.gains) for value in args.large_error_gains):
        parser.error("large-error gains must be at least the base gain")
    if args.large_error_max_linear_speed < args.max_linear_speed:
        parser.error(
            "large-error linear-speed cap must be at least the base cap")
    if not 0.0 <= args.large_error_start_px < args.large_error_full_px:
        parser.error(
            "large-error thresholds must satisfy 0 <= start < full")
    if not (
        0.0 <= args.slow_target_full_px_s <
        args.slow_target_zero_px_s
    ):
        parser.error(
            "slow-target thresholds must satisfy 0 <= full < zero")
    if args.duration <= 0.0 or args.cooldown < 0.0:
        parser.error("duration must be positive and cooldown non-negative")
    if args.tolerance_px <= 0.0:
        parser.error("tolerance must be positive")
    if args.output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = (
            Path("/home/clover/ur_ws/ibvs_rm65/runtime") /
            f"gradient_sweep_{stamp}")
    return args


def main() -> int:
    return run_sweep(build_parser())


if __name__ == "__main__":
    raise SystemExit(main())
