#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."; source tools/env.sh; mkdir -p .runtime
exec flock .runtime/gpu-execution.lock bash -ec '
for check in memcheck racecheck synccheck; do
  printf "frozen-w8a8-rhs: %s; no kernel-name filter\n" "$check"
  compute-sanitizer --tool "$check" --target-processes application-only --error-exitcode 86 "$1" -m pytest -q "tests/test_quant_state.py::test_frozen_weight_content_staging_and_ownership[row-column_major-mixed]"
done' _ "$PY"
