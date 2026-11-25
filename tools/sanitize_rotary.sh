#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."; source tools/env.sh; mkdir -p .runtime
exec flock .runtime/gpu-execution.lock bash -ec '
for check in memcheck racecheck synccheck; do
    printf "paired rotary: %s\n" "$check"
    compute-sanitizer --tool "$check" --target-processes application-only \
        --kernel-name kns=rotary_pair --error-exitcode 86 "$1" -m pytest -q tests/test_rotary.py -k pair_layout_and_rounding
done' _ "$PY"
