#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$root/python:$root/../nova5_visual_servo${PYTHONPATH:+:$PYTHONPATH}"
# This node starts idle; its explicit ROS start service retains the existing
# calibration, timestamp, robot-status, feedback and workspace checks.
exec python3 -m nova5_tongji.ros_servo "$@"
