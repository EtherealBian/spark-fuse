#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."; source tools/env.sh; mkdir -p .runtime
exec flock .runtime/gpu-execution.lock bash -ec '
for check in memcheck racecheck synccheck; do
    printf "ordered mean reductions: %s\n" "$check"
    compute-sanitizer --tool "$check" --target-processes application-only \
        --kernel-name kns=fused_mean_tree --error-exitcode 86 "$1" -m pytest -q tests/test_mean_tree.py
done' _ "$PY"
