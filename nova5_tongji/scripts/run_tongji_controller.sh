#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$root:$root/nova5_tongji/python:$root/nova5_visual_servo${PYTHONPATH:+:$PYTHONPATH}"
python="$root/.venv/bin/python"
if [[ ! -x "$python" ]]; then
  python=python3
fi
# Small servo matrices share the VM with the camera and feedback threads.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec "$python" -m nova5_tongji.tongji_direct_tracking "$@"
