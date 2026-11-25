#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."; source tools/env.sh; mkdir -p .runtime
exec flock .runtime/gpu-execution.lock bash -ec '
for check in memcheck racecheck synccheck; do
  printf "residual-token-quant: %s\n" "$check"
  compute-sanitizer --tool "$check" --target-processes application-only --kernel-name kns=residual_token_quant --error-exitcode 86 "$1" -m pytest -q tests/test_token_quant.py -k current_stream_and_visible_outputs
done' _ "$PY"
