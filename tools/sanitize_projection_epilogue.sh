#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."; source tools/env.sh; mkdir -p .runtime
exec flock .runtime/gpu-execution.lock bash -ec '
for check in memcheck racecheck synccheck; do
  printf "ordered-int32-epilogue: %s\n" "$check"
  compute-sanitizer --tool "$check" --target-processes application-only --kernel-name kns=ordered_int32_epilogue --error-exitcode 86 "$1" -m pytest -q tests/test_projection_epilogue.py -k visible_exact_order_and_replay
done' _ "$PY"
