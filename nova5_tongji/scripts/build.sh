#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
git -C "$root/.." submodule update --init --recursive -- nova5_tongji/third_party/sp_vision_25
cmake -S "$root" -B "$root/build" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$root/build" -j 2
