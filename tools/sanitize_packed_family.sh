#!/usr/bin/env bash
# Run the actual shape-family request lifetimes and failed-admission cleanups, without kernel filters.
set -euo pipefail
cd "$(dirname "$0")/.."; source tools/env.sh; mkdir -p .runtime
exec flock .runtime/gpu-execution.lock bash -ec '
for check in memcheck racecheck synccheck; do
  printf "packed shape family: %s\n" "$check"
  compute-sanitizer --tool "$check" --target-processes application-only --error-exitcode 86 "$1" -m pytest -q --disable-warnings tests/test_packed_family.py
done' _ "$PY"
