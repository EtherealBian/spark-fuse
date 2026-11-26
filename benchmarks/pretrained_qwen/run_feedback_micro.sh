#!/usr/bin/env bash
# From the repository root. No model weights are used in this isolated GPU test.
set -euo pipefail
: "${OUT:?set OUT to a new directory}" "${PYTHON:=python3}"
D=$(cd -- "$(dirname -- "$0")" && pwd); source tools/env.sh
mkdir -p .runtime; exec 9>.runtime/gpu-execution.lock; flock -x 9
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; mkdir -- "$OUT"
"$PYTHON" -B "$D/measure_feedback.py" "$OUT"
python3 -B -E -S "$D/audit_trials.py" "$OUT/raw.json" | cmp - "$OUT/summary.json"
