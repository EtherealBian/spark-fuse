#!/usr/bin/env bash
# Each backend gets its own process/allocator. Use a dedicated, idle GPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/env.sh
: "${PYTHON:=python3}"; : "${WEIGHTS:?set WEIGHTS to the pinned checkpoint directory}"
: "${WORK:=.runtime/pretrained-qwen-7b/run}"; : "${REPORTS:=$WORK/reports}"
mkdir -p "$WORK" "$REPORTS"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHINDUCTOR_COMPILE_THREADS=4
export TORCHINDUCTOR_CACHE_DIR="$WORK/inductor" TRITON_CACHE_DIR="$WORK/triton"
B=benchmarks/pretrained_qwen
"$PYTHON" "$B/checkpoint7b.py" "$WEIGHTS"
err=0
for nk in 128:eager 128:resident 128:inductor 2048:eager 2048:inductor 2048:resident; do
 n=${nk%:*}; k=${nk#*:}; echo "start $n $k"
 if ! "$PYTHON" -u "$B/scale7b.py" --weights "$WEIGHTS" --work "$WORK/references" --length "$n" --backend "$k" --output "$REPORTS/qwen7b-$n-$k.json" > "$WORK/$n-$k.log" 2>&1; then
    echo "failed $n $k; see $WORK/$n-$k.log" >&2; err=1
 else tail -1 "$WORK/$n-$k.log"; fi
done
if ((err)); then exit 1; fi
"$PYTHON" -I -c 'import sys,runpy; sys.path.insert(0,"benchmarks/pretrained_qwen"); sys.argv=["audit7b.py",sys.argv[1]];runpy.run_module("audit7b",run_name="__main__")' "$REPORTS" > "$WORK/summary.tmp"
mv "$WORK/summary.tmp" "$REPORTS/qwen7b-summary.json"
