#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${TONGJI_LEGACY_PROJECT:?Set the path to the verified installed visual-servo runtime}"
export PYTHONPATH="$root/nova5_tongji/python:$root/nova5_visual_servo${PYTHONPATH:+:$PYTHONPATH}"
export TONGJI_TCP_MODE="${TONGJI_TCP_MODE:-shadow}"
exec python3 -m nova5_tongji.tcp_bridge "$@"
