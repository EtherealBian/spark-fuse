#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."; source tools/env.sh; mkdir -p .runtime
exec flock .runtime/gpu-execution.lock bash -ec '
for check in memcheck racecheck synccheck; do
  printf "bounded packed rows: %s\n" "$check"
  compute-sanitizer --tool "$check" --target-processes application-only --kernel-name kns=packed_swiglu --error-exitcode 86 "$1" -m pytest -q tests/test_packed_rows.py -k bounded_packed_rows_dispatch
done' _ "$PY"
