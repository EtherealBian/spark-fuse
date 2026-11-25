#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."; source tools/env.sh; mkdir -p .runtime
exec flock .runtime/gpu-execution.lock bash -ec '
for check in memcheck racecheck synccheck; do
    printf "paired head repeat: %s\n" "$check"
    compute-sanitizer --tool "$check" --target-processes application-only --kernel-name kns=paired_head_repeat --error-exitcode 86 "$1" -m pytest -q tests/test_head_repeat.py -k bits_views_and_expansion
done' _ "$PY"
