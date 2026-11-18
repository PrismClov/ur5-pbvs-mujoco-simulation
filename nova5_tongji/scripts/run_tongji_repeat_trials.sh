#!/usr/bin/env bash
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python="$root/.venv/bin/python"
[[ -x "$python" ]] || python=python3

baseline="/home/clover/ur_ws/ibvs_rm65/runtime/tongji_pose_baseline_20261118.json"
output_dir="/home/clover/ur_ws/ibvs_rm65/runtime/tongji_repeat_each_from_baseline"
trials=3
duration=8
attempts_per_trial=2
execute=false
controller_args=(--ekf 12d --upright)

while (($#)); do
  case "$1" in
    --execute) execute=true; shift ;;
    --baseline) baseline="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --trials) trials="$2"; shift 2 ;;
    --duration) duration="$2"; shift 2 ;;
    --attempts-per-trial) attempts_per_trial="$2"; shift 2 ;;
    --) shift; controller_args+=("$@"); break ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if ((trials < 1)) || ((duration < 1)) || ((attempts_per_trial < 1)); then
  echo "trials, duration, and attempts must be positive" >&2
  exit 2
fi
if [[ "$execute" != true ]]; then
  printf 'Dry run: %d trials, %ds each, baseline=%s\n' \
    "$trials" "$duration" "$baseline"
  exit 0
fi

mkdir -p "$output_dir"
export PYTHONPATH="$root:$root/nova5_tongji/python:$root/nova5_visual_servo${PYTHONPATH:+:$PYTHONPATH}"

restore() {
  local label="$1"
  "$python" "$root/nova5_tongji/scripts/restore_tongji_pose.py" \
    --config "$root/nova5_tongji/field_runtime/config/tongji_direct_tracking.json" \
    --baseline "$baseline" \
    --output "$output_dir/${label}_restore.json" \
    --execute
  local status=$?
  if ((status == 0)); then
    # Give the Dobot dashboard service time to release the previous client.
    sleep 2
  fi
  return "$status"
}

restore "initial"
initial_restore_status=$?
if ((initial_restore_status != 0)); then
  echo "initial pose restore failed" >&2
  exit "$initial_restore_status"
fi

overall_status=0
for ((index = 1; index <= trials; index++)); do
  trial_passed=false
  for ((attempt = 1; attempt <= attempts_per_trial; attempt++)); do
    prefix="$(printf 'trial_%02d_attempt_%02d' "$index" "$attempt")"
    echo "Starting $prefix from captured baseline"
    bash "$root/nova5_tongji/scripts/run_tongji_controller.sh" \
      "${controller_args[@]}" \
      --execute \
      --duration "$duration" \
      --observation-log "$output_dir/${prefix}_observations.jsonl" \
      --output-log "$output_dir/${prefix}.jsonl"
    trial_status=$?
    # The controller closes its dashboard connection in cleanup. Let the robot
    # service release that client before opening the restore connection.
    sleep 2
    restore "${prefix}_after"
    restore_status=$?
    if ((restore_status != 0)); then
      echo "restore after $prefix failed" >&2
      exit "$restore_status"
    fi
    if ((trial_status == 0)); then
      trial_passed=true
      break
    fi
    echo "$prefix aborted; retrying after baseline restore" >&2
  done
  if [[ "$trial_passed" != true ]]; then
    overall_status=1
  fi
done

exit "$overall_status"
