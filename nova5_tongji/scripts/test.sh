#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$root/python:$root/../nova5_visual_servo${PYTHONPATH:+:$PYTHONPATH}"
python="$root/../.venv/bin/python"
if [[ ! -x "$python" ]]; then
  python=python3
fi
"$python" -m unittest discover -s "$root/tests" -v
