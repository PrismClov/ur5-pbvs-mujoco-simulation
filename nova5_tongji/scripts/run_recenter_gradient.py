#!/usr/bin/env python3
"""Create bounded image offsets with the arm, then measure recentering latency.

For every trial this script:
1. runs the existing executor against an offset desired-feature file,
2. stops once the observed marker center reaches that offset,
3. starts a second executor against the centered desired-feature file,
4. records the first time the target returns to the pixel tolerance band,
5. saves per-trial JSON/CSV and stops the robot.

The Dobot driver, D435i camera and real_tag_tracker.py must already run.
Nothing moves unless --execute is provided.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_gradient_sweep as gradient  # noqa: E402


def marker_center(detection: dict[str, Any]) -> tuple[float, float] | None:
    if not detection.get("valid"):
        return None
    corners = detection.get("corners") or []
    if len(corners) < 4:
        return None
    try:
        u = statistics.fmean(float(point[0]) for point in corners)
        v = statistics.fmean(float(point[1]) for point in corners)
    except (TypeError, ValueError, IndexError):
        return None
    if not math.isfinite(u) or not math.isfinite(v):
        return None
    return u, v


def center_error(detection: dict[str, Any],
                 principal: tuple[float, float]) -> float | None:
    center = marker_center(detection)
    if center is None:
        return None
    return math.hypot(center[0] - principal[0], center[1] - principal[1])


def offset_error(detection: dict[str, Any],
                 principal: tuple[float, float],
                 offset: tuple[float, float]) -> float | None:
    center = marker_center(detection)
    if center is None:
        return None
    return math.hypot(
        center[0] - principal[0] - offset[0],
        center[1] - principal[1] - offset[1],
    )


def create_offset_desired(
    source: Path,
    destination: Path,
    offset: tuple[float, float],
) -> None:
    data = json.loads(source.read_text())
    features = data.get("features")
    if not isinstance(features, dict) or len(features) < 4:
        raise ValueError("desired file has too few image features")
    shifted = {}
    for key, point in features.items():
        shifted[key] = [
            float(point[0]) + offset[0],
            float(point[1]) + offset[1],
        ]
    data["features"] = shifted
    data["source"] = (
        f"runtime offset ({offset[0]:+.3f}, {offset[1]:+.3f}) px "
        f"from {source.name}"
    )
    destination.write_text(json.dumps(data, indent=2) + "\n")


def attach_detection_subscription(harness: gradient.RosHarness) -> list[dict]:
    from std_msgs.msg import String

    detections: list[dict[str, Any]] = []

    def callback(message: Any) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        payload["_received_monotonic"] = time.monotonic()
        if payload.get("valid"):
            detections.append(payload)
            if len(detections) > 2000:
                del detections[:500]

    harness.node.create_subscription(
        String, "/tag_tracker/detections", callback, 10)
    return detections


def wait_offset_reached(
    harness: gradient.RosHarness,
    detections: list[dict[str, Any]],
    principal: tuple[float, float],
    target_offset: tuple[float, float],
    tolerance_px: float,
    stable_s: float,
    timeout_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    stable_since = None
    last_error = None
    while time.monotonic() < deadline:
        harness.spin_for(0.02)
        if not detections:
            continue
        last_error = offset_error(
            detections[-1], principal, target_offset)
        if last_error is None:
            stable_since = None
            continue
        if last_error <= tolerance_px:
            if stable_since is None:
                stable_since = time.monotonic()
            if time.monotonic() - stable_since >= stable_s:
                return {
                    "reached": True,
                    "offset_error_px": last_error,
                    "actual_error_from_center_px": center_error(
                        detections[-1], principal),
                }
        else:
            stable_since = None
    return {
        "reached": False,
        "offset_error_px": last_error,
        "actual_error_from_center_px": (
            None if not detections
            else center_error(detections[-1], principal)),
    }


def sample_after_time(
    detections: list[dict[str, Any]],
    started_monotonic: float,
    principal: tuple[float, float],
) -> list[dict[str, float]]:
    rows = []
    for detection in detections:
        received = float(detection.get("_received_monotonic", -1.0))
        if received < started_monotonic:
            continue
        error = center_error(detection, principal)
        if error is None:
            continue
        rows.append({
            "elapsed_s": received - started_monotonic,
            "error_px": error,
            "stamp_sec": float(detection.get("stamp_sec", 0.0)),
        })
    rows.sort(key=lambda row: row["elapsed_s"])
    return rows


def latency_metrics(
    samples: list[dict[str, float]],
    tolerance_px: float,
) -> dict[str, Any]:
    if not samples:
        return {
            "initial_error_px": None,
            "final_error_px": None,
            "recovery_entry_s": None,
            "recovery_hold_200ms_s": None,
            "error_at_100ms_px": None,
            "error_at_200ms_px": None,
            "error_at_300ms_px": None,
            "meets_200ms": False,
            "meets_200ms_hold": False,
        }
    entry = next(
        (row["elapsed_s"] for row in samples
         if row["error_px"] <= tolerance_px),
        None,
    )
    hold_entry = None
    for index, row in enumerate(samples):
        start = row["elapsed_s"]
        end = start + 0.2
        window = [
            item for item in samples[index:]
            if item["elapsed_s"] <= end
        ]
        following = next(
            (item for item in samples[index:] if item["elapsed_s"] > end),
            None,
        )
        if (
            window and
            (
                window[-1]["elapsed_s"] >= end - 1e-9 or
                following is not None
            ) and
            all(
                item["error_px"] <= tolerance_px
                for item in (
                    [*window, following]
                    if following is not None else window
                )
            )
        ):
            hold_entry = start
            break

    def interpolate(at_s: float) -> float | None:
        earlier = [row for row in samples if row["elapsed_s"] <= at_s]
        later = [row for row in samples if row["elapsed_s"] >= at_s]
        if not earlier or not later:
            return None
        left, right = earlier[-1], later[0]
        if left is right or right["elapsed_s"] == left["elapsed_s"]:
            return left["error_px"]
        ratio = (
            (at_s - left["elapsed_s"]) /
            (right["elapsed_s"] - left["elapsed_s"])
        )
        return left["error_px"] + ratio * (
            right["error_px"] - left["error_px"])

    return {
        "initial_error_px": samples[0]["error_px"],
        "final_error_px": samples[-1]["error_px"],
        "minimum_error_px": min(row["error_px"] for row in samples),
        "recovery_entry_s": entry,
        "recovery_hold_200ms_s": hold_entry,
        "error_at_100ms_px": interpolate(0.1),
        "error_at_200ms_px": interpolate(0.2),
        "error_at_300ms_px": interpolate(0.3),
        "meets_200ms": entry is not None and entry <= 0.2,
        "meets_200ms_hold": hold_entry is not None and hold_entry <= 0.2,
        "sample_count": len(samples),
    }


def trial_combinations(
    args: argparse.Namespace,
) -> list[dict[str, float]]:
    rows = []
    for offset_px in args.offsets_px:
        for gain in args.gains:
            for large_error_gain in args.large_error_gains:
                for max_linear_speed in args.max_linear_speeds:
                    for large_max_speed in args.large_error_max_linear_speeds:
                        if large_max_speed < max_linear_speed:
                            continue
                        for speed_factor in args.speed_factors:
                            rows.append({
                                "offset_px": float(offset_px),
                                "gain": float(gain),
                                "large_error_gain": float(large_error_gain),
                                "max_linear_speed": float(max_linear_speed),
                                "large_error_max_linear_speed": float(
                                    large_max_speed),
                                "speed_factor": float(speed_factor),
                                "angular_gain": float(args.angular_gain),
                            })
    return rows


def command_for(
    args: argparse.Namespace,
    trial: dict[str, float],
    desired_file: Path,
    output_log: Path,
    duration_s: float,
) -> list[str]:
    trial_args = copy.copy(args)
    trial_args.desired_file = desired_file
    trial_args.duration = duration_s
    trial_args.max_linear_speed = trial["max_linear_speed"]
    trial_args.large_error_max_linear_speed = (
        trial["large_error_max_linear_speed"])
    values = dict(trial)
    values["angular_gain"] = trial["angular_gain"]
    return gradient.trial_command(trial_args, values, output_log)


def start_executor(
    harness: gradient.RosHarness,
    command: list[str],
    log_path: Path,
) -> tuple[Any, float]:
    stream = log_path.open("w")
    process = subprocess.Popen(
        command,
        stdout=stream,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    stream.close()
    try:
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
    except Exception:
        gradient.terminate_process(process, log_path)
        raise
    return process, time.monotonic()


def stop_executor(
    harness: gradient.RosHarness,
    process: Any,
    log_path: Path,
) -> None:
    try:
        if harness.start.service_is_ready():
            response = harness.call_bool(harness.start, False)
            if not response.success:
                print(f"  warning: stop response: {response.message}")
    finally:
        # Let an already-issued ServoP response finish deserialization before
        # interrupting the Python process.
        harness.spin_for(0.15)
        gradient.terminate_process(process, log_path)


def run_center_cleanup(
    args: argparse.Namespace,
    harness: gradient.RosHarness,
    detections: list[dict[str, Any]],
    principal: tuple[float, float],
    output_dir: Path,
) -> bool:
    trial = {
        "gain": args.cleanup_gain,
        "large_error_gain": args.cleanup_gain,
        "angular_gain": 1.0,
        "speed_factor": min(args.speed_factors),
        "max_linear_speed": min(args.max_linear_speeds),
        "large_error_max_linear_speed": min(args.max_linear_speeds),
    }
    command = command_for(
        args, trial, args.desired_file,
        output_dir / "cleanup_tracking.jsonl",
        args.cleanup_seconds,
    )
    process = None
    log_path = output_dir / "cleanup_executor.log"
    try:
        detections.clear()
        process, started = start_executor(harness, command, log_path)
        deadline = time.monotonic() + args.cleanup_seconds
        while time.monotonic() < deadline:
            harness.spin_for(0.02)
            if (
                detections and
                (center_error(detections[-1], principal) or math.inf) <=
                args.tolerance_px
            ):
                break
        samples = sample_after_time(detections, started, principal)
        metrics = latency_metrics(samples, args.tolerance_px)
        return bool(
            metrics["final_error_px"] is not None and
            metrics["final_error_px"] <= args.tolerance_px)
    finally:
        if process is not None:
            stop_executor(harness, process, log_path)


def run(args: argparse.Namespace) -> int:
    center_data = json.loads(args.desired_file.read_text())
    principal = tuple(
        float(value) for value in center_data["principal_point_px"])
    if len(principal) != 2:
        raise ValueError("principal_point_px must contain two values")
    combos = trial_combinations(args)
    if not combos:
        raise ValueError("no valid trial combinations")
    print(
        f"{len(combos)} recenter trial(s), offsets "
        f"{', '.join(f'{value:g}' for value in args.offsets_px)} px")
    if not args.execute:
        for index, combo in enumerate(combos, 1):
            print(
                f"  {index:03d}: offset={combo['offset_px']:g}px, "
                f"gain={combo['gain']:g}, "
                f"large_gain={combo['large_error_gain']:g}, "
                f"vmax={combo['max_linear_speed']:g}, "
                f"large_vmax={combo['large_error_max_linear_speed']:g}, "
                f"speed={combo['speed_factor']:g}%, "
                f"mode={args.command_mode}")
        print("Dry run only. Add --execute to run the arm.")
        return 0
    if not sys.stdin.isatty() and args.prompt_once:
        raise RuntimeError("--prompt-once requires an interactive terminal")
    if args.prompt_once:
        print(
            "Confirm the robot workspace is clear and the emergency stop is "
            "available. Press Enter to begin.")
        input()

    harness = gradient.RosHarness()
    detections = attach_detection_subscription(harness)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    total_translation = 0.0
    try:
        if harness.service_present(0.5):
            raise RuntimeError(
                "an executor is already running; stop it before testing")
        for index, combo in enumerate(combos, 1):
            if total_translation >= args.total_translation_budget_mm:
                print("Stopping: total translation budget exhausted")
                break
            sign = -1.0 if index % 2 == 0 else 1.0
            if args.axis == "x":
                offset = (sign * combo["offset_px"], 0.0)
            elif args.axis == "y":
                offset = (0.0, sign * combo["offset_px"])
            else:
                component = combo["offset_px"] / math.sqrt(2.0)
                offset = (sign * component, component)
            trial_dir = output_dir / f"trial_{index:03d}"
            trial_dir.mkdir()
            offset_file = trial_dir / "offset_desired.json"
            create_offset_desired(args.desired_file, offset_file, offset)
            print(
                f"\nTrial {index}/{len(combos)}: offset="
                f"({offset[0]:+.1f}, {offset[1]:+.1f}) px, "
                f"gain={combo['gain']:g}, "
                f"large_gain={combo['large_error_gain']:g}, "
                f"vmax={combo['max_linear_speed']:g}, "
                f"large_vmax={combo['large_error_max_linear_speed']:g}, "
                f"speed={combo['speed_factor']:g}%, "
                f"mode={args.command_mode}")

            setup_combo = dict(combo)
            setup_combo["gain"] = args.setup_gain
            setup_combo["large_error_gain"] = args.setup_gain
            setup_combo["max_linear_speed"] = max(
                combo["max_linear_speed"], args.setup_max_linear_speed)
            setup_combo["large_error_max_linear_speed"] = max(
                combo["large_error_max_linear_speed"],
                args.setup_max_linear_speed)
            setup_command = command_for(
                args, setup_combo, offset_file,
                trial_dir / "offset_tracking.jsonl",
                args.offset_timeout)
            detections.clear()
            setup_process, _ = start_executor(
                harness, setup_command, trial_dir / "offset_executor.log")
            reached = wait_offset_reached(
                harness, detections, principal, offset,
                args.offset_tolerance_px,
                args.offset_stable_seconds,
                args.offset_timeout,
            )
            stop_executor(
                harness, setup_process,
                trial_dir / "offset_executor.log")
            harness.spin_for(args.stop_settle_seconds)
            pre_error = (
                center_error(detections[-1], principal)
                if detections else None)
            setup_summary = gradient.summarize_status_samples(
                harness.samples,
                elapsed_s=args.offset_timeout,
                tolerance_px=args.tolerance_px,
            )
            if not reached["reached"]:
                row = {
                    "trial": index,
                    **combo,
                    "offset_x_px": offset[0],
                    "offset_y_px": offset[1],
                    "offset_reached": False,
                    "command_mode": args.command_mode,
                    "pre_recovery_error_px": pre_error,
                    "meets_200ms": False,
                    "output_dir": str(trial_dir),
                }
                rows.append(row)
                gradient.write_outputs(output_dir, rows)
                if not run_center_cleanup(
                    args, harness, detections, principal, output_dir
                ):
                    raise RuntimeError(
                        "offset failed and automatic centering did not "
                        "restore the target")
                continue

            recovery_trial = dict(combo)
            recovery_command = command_for(
                args, recovery_trial, args.desired_file,
                trial_dir / "recovery_tracking.jsonl",
                args.recovery_seconds)
            detections.clear()
            harness.samples.clear()
            recovery_process, started = start_executor(
                harness, recovery_command,
                trial_dir / "recovery_executor.log")
            harness.spin_for(args.recovery_seconds)
            stop_executor(
                harness, recovery_process,
                trial_dir / "recovery_executor.log")

            samples = sample_after_time(detections, started, principal)
            latency = latency_metrics(samples, args.tolerance_px)
            status_summary = gradient.summarize_status_samples(
                harness.samples,
                elapsed_s=args.recovery_seconds,
                tolerance_px=args.tolerance_px,
            )
            row = {
                "trial": index,
                **combo,
                "offset_x_px": offset[0],
                "offset_y_px": offset[1],
                "offset_reached": True,
                "command_mode": args.command_mode,
                "offset_achievement_error_px": reached["offset_error_px"],
                "pre_recovery_error_px": pre_error,
                **latency,
                **status_summary,
                "output_dir": str(trial_dir),
            }
            rows.append(row)
            gradient.write_outputs(output_dir, rows)
            (trial_dir / "recovery_samples.json").write_text(
                json.dumps(samples, indent=2) + "\n")
            total_translation += float(
                status_summary.get("cumulative_translation_mm") or 0.0)
            total_translation += float(
                setup_summary.get("cumulative_translation_mm") or 0.0)
            print(
                f"  entry={latency['recovery_entry_s']} s, "
                f"at100ms={latency['error_at_100ms_px']}, "
                f"at200ms={latency['error_at_200ms_px']}, "
                f"final={latency['final_error_px']}, "
                f"meets_200ms={latency['meets_200ms']}, "
                f"faults={status_summary['fault_reasons']}")
            if (
                latency["final_error_px"] is None or
                latency["final_error_px"] > args.tolerance_px
            ):
                if not run_center_cleanup(
                    args, harness, detections, principal, output_dir
                ):
                    raise RuntimeError(
                        "recovery did not reach tolerance and automatic "
                        "centering failed")
        print(f"\nResults: {output_dir / 'summary.csv'}")
        return 0
    finally:
        try:
            if harness.start.service_is_ready():
                harness.call_bool(harness.start, False)
        except Exception:
            pass
        harness.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--offsets-px", type=gradient.parse_levels,
        default=gradient.parse_levels("10:40:10"))
    parser.add_argument(
        "--gains", type=gradient.parse_levels,
        default=gradient.parse_levels("2.4"))
    parser.add_argument(
        "--large-error-gains", type=gradient.parse_levels,
        default=gradient.parse_levels("6"))
    parser.add_argument(
        "--max-linear-speeds", type=gradient.parse_levels,
        default=gradient.parse_levels("0.08"))
    parser.add_argument(
        "--large-error-max-linear-speeds", type=gradient.parse_levels,
        default=gradient.parse_levels("0.12"))
    parser.add_argument(
        "--speed-factors", type=gradient.parse_levels,
        default=gradient.parse_levels("50:100:50"))
    parser.add_argument("--angular-gain", type=float, default=1.0)
    parser.add_argument("--damping", type=float, default=0.4)
    parser.add_argument(
        "--axis", choices=("x", "y", "diagonal"), default="x")
    parser.add_argument("--tolerance-px", type=float, default=5.0)
    parser.add_argument("--offset-tolerance-px", type=float, default=4.0)
    parser.add_argument("--offset-stable-seconds", type=float, default=0.15)
    parser.add_argument("--offset-timeout", type=float, default=6.0)
    parser.add_argument("--recovery-seconds", type=float, default=1.5)
    parser.add_argument("--stop-settle-seconds", type=float, default=0.15)
    parser.add_argument("--setup-gain", type=float, default=4.0)
    parser.add_argument("--setup-max-linear-speed", type=float, default=0.12)
    parser.add_argument("--cleanup-gain", type=float, default=2.4)
    parser.add_argument("--cleanup-seconds", type=float, default=4.0)
    parser.add_argument(
        "--total-translation-budget-mm", type=float, default=600.0)
    parser.add_argument(
        "--max-cumulative-translation-mm", type=float, default=120.0)
    parser.add_argument(
        "--max-cumulative-rotation-deg", type=float, default=5.0)
    parser.add_argument("--max-angular-speed-deg", type=float, default=5.0)
    parser.add_argument("--max-step-rotation-deg", type=float, default=0.2)
    parser.add_argument("--command-duration", type=float, default=0.02)
    parser.add_argument("--command-settle-margin", type=float, default=0.005)
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
        "--large-error-start-px", type=float, default=12.0)
    parser.add_argument(
        "--large-error-full-px", type=float, default=30.0)
    parser.add_argument(
        "--slow-target-full-px-s", type=float, default=20.0)
    parser.add_argument(
        "--slow-target-zero-px-s", type=float, default=60.0)
    parser.add_argument("--adaptive-gain", action="store_true", default=True)
    parser.add_argument(
        "--config", type=Path,
        default=Path("/home/clover/ur_ws/ibvs_rm65/config/"
                     "nova5_fast_execution.yaml"))
    parser.add_argument(
        "--desired-file", type=Path,
        default=Path("/home/clover/ur_ws/ibvs_rm65/runtime/"
                     "nova5_aruco_center_040m.json"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--executor-package", default="ibvs_ur5")
    parser.add_argument(
        "--executor-executable",
        default="nova5_low_speed_executor.py")
    parser.add_argument(
        "--prompt-once",
        action=argparse.BooleanOptionalAction,
        default=True)
    args = parser.parse_args()
    if args.output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = (
            Path("/home/clover/ur_ws/ibvs_rm65/runtime") /
            f"recenter_gradient_{stamp}")
    if args.recovery_seconds < 0.4:
        parser.error("recovery duration must be at least 0.4 s")
    if args.total_translation_budget_mm <= 0.0:
        parser.error("total translation budget must be positive")
    if any(value < max(args.gains) for value in args.large_error_gains):
        parser.error("large-error gains must be at least the base gain")
    return args


def main() -> int:
    return run(build_parser())


if __name__ == "__main__":
    raise SystemExit(main())
