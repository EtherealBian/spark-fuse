#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."; source tools/env.sh; mkdir -p .runtime
exec flock .runtime/gpu-execution.lock bash -ec '
for check in memcheck racecheck synccheck; do
  printf "segmented cache decode: %s\n" "$check"
  compute-sanitizer --tool "$check" --target-processes application-only --error-exitcode 86 "$1" -m pytest -q --disable-warnings tests/test_segmented_decode.py
done' _ "$PY"
