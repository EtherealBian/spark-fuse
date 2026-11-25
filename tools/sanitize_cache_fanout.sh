#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."; source tools/env.sh; mkdir -p .runtime
exec flock .runtime/gpu-execution.lock bash -ec '
for check in memcheck racecheck synccheck; do
  printf "four-output KV cache fanout: %s\n" "$check"
  compute-sanitizer --tool "$check" --target-processes application-only --kernel-name kns=cache_fanout --error-exitcode 86 "$1" -m pytest -q tests/test_cache_fanout.py -k fanout_bits_aliases_streams
done' _ "$PY"
