#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."; source tools/env.sh; mkdir -p .runtime
exec flock .runtime/gpu-execution.lock bash -ec '
for check in memcheck racecheck synccheck; do
  printf "split-copy3: %s\n" "$check"
  compute-sanitizer --tool "$check" --target-processes application-only --kernel-name kns=split_copy3 --error-exitcode 86 "$1" -m pytest -q tests/test_split_copy.py -k copies_and_visible_aliases
done' _ "$PY"
