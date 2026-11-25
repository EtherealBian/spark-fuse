#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."; source tools/env.sh; mkdir -p .runtime
exec flock .runtime/gpu-execution.lock bash -ec '
for check in memcheck racecheck synccheck; do
  printf "packed-projection CUDA epilogue: %s\n" "$check"
  compute-sanitizer --tool "$check" --target-processes application-only --kernel-name kns=packed_int32_fanout --error-exitcode 86 "$1" -m pytest -q --disable-warnings tests/test_packed_projection.py -k "segmented_addressing or preserves_live_roles"
done' _ "$PY"
