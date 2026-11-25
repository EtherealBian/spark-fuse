#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."; source tools/env.sh; mkdir -p .runtime
exec flock .runtime/gpu-execution.lock bash -ec '
for check in memcheck racecheck synccheck; do
  printf "sliced_cache_update: %s\n" "$check"
  compute-sanitizer --tool "$check" --target-processes application-only --kernel-name kns=sliced_cache_update --error-exitcode 86 "$1" -m pytest -q tests/test_sliced_cache.py -k exact_strided_history
done' _ "$PY"
