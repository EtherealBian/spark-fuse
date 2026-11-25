#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/..";source tools/env.sh; mkdir -p .runtime
exec flock .runtime/gpu-execution.lock bash -c '
set -eo pipefail
for tool in memcheck racecheck synccheck; do
    compute-sanitizer --tool "$tool" --target-processes application-only --error-exitcode 86 --kernel-name kns=_Z5fused "$1" tools/sanitize.py
done' _ "$PY"
