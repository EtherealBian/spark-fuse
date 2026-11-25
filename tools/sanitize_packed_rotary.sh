#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."; source tools/env.sh; mkdir -p .runtime
exec flock .runtime/gpu-execution.lock bash -ec '
for check in memcheck racecheck synccheck; do
  printf "packed_rotary_copy: %s\n" "$check"
  compute-sanitizer --tool "$check" --target-processes application-only --kernel-name kns=packed_rotary_copy --error-exitcode 86 "$1" -m pytest -q tests/test_packed_rotary.py -k live_aliases_and_rounding
done' _ "$PY"
