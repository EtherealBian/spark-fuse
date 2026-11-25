#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."; source tools/env.sh; mkdir -p .runtime
# Serialize against campaign workers so correctness tests do not distort microbenchmarks.
exec flock .runtime/gpu-execution.lock "$PY" -m pytest -q "$@"
