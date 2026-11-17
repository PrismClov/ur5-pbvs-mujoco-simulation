#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$root/python:$root/../nova5_visual_servo${PYTHONPATH:+:$PYTHONPATH}"
python3 -m unittest discover -s "$root/tests" -v
