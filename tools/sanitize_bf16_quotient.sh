#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."; source tools/env.sh; mkdir -p .runtime
exec flock .runtime/gpu-execution.lock bash -ec '
for check in memcheck racecheck synccheck; do
  printf "guarded BF16 quotient: %s\n" "$check"
  compute-sanitizer --tool "$check" --target-processes application-only --kernel-name kns=fused --error-exitcode 86 "$1" -m pytest -q tests/test_bf16_score_guard.py -k guarded_stream_offsets_and_nonfinite
done' _ "$PY"
