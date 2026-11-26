#!/usr/bin/env bash
# Run on an idle GPU; the two model instances never coexist.
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/env.sh
: "${PYTHON:=python3}" "${WEIGHTS:?set WEIGHTS to a pinned local checkpoint directory}"
: "${WORK:=.runtime/pretrained-qwen-32b/run}"; : "${REPORTS:=$WORK/reports}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$WORK" "$REPORTS" .runtime
exec 9>.runtime/gpu-execution.lock; flock -x 9
B=benchmarks/pretrained_qwen
for k in eager resident; do
  "$PYTHON" -u "$B/scale32b.py" --weights "$WEIGHTS" --work "$WORK/references" --backend "$k" --output "$REPORTS/qwen32b-$k.json"
done
"$PYTHON" "$B/audit32b.py" "$REPORTS" > "$WORK/summary.tmp"
mv "$WORK/summary.tmp" "$REPORTS/qwen32b-summary.json"
