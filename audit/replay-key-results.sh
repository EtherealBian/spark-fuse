#!/usr/bin/env bash
# Rebuild each report from the archived evidence; never start a server or fetch weights.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
unset PYTHONHOME PYTHONOPTIMIZE PYTHONINSPECT PYTHONSAFEPATH
export LC_ALL=C PYTHONPATH="$PWD/benchmarks/quant_serving"
B=benchmarks/quant_serving
check() {
  local expected=$1
  shift
  python3 -B -S "$@" | cmp - "$expected"
  printf 'OK  %s\n' "$expected"
}
R=reports/qwen32b-fork
check "$R/summary.json" benchmarks/pretrained_qwen/audit_fork32.py bundle "$R" reports
check "$R/paired-costs.json" audit/prefix_pair_costs.py "$R/raw.json"
R=reports/qwen72b/fifo/live; REF=reports/qwen72b/assignment
export PYTHONPATH="$PWD/$B/fifo72:$PWD/$B"
check "$R/summary.json" "$B/fifo72/live/live72_review.py" "$R" "$REF"
check "$R/wave-ledger.json" "$B/fifo72/live/wave_ledger.py" "$R" "$REF"
export PYTHONPATH="$PWD/$B"
REF=reports/qwen235b; R=$REF/natural-stream; LOCK=$B/qwen235b-q2k.lock.json
check "$R/summary.json" "$B/stream235/review_natural.py" "$LOCK" "$R" "$REF"
check "$R/delivery-ledger.json" "$B/stream235/delivery_ledger.py" "$LOCK" "$R" "$REF"
REF=reports/glm45-355b; R=$REF/paired
check "$R/summary.json" "$B/paired355/review_pair355.py" "$B/glm45-udiq1s.lock.json" "$R" "$REF" "$REF/tokenizer-contract.json"
check "$R/occupancy-ledger.json" "$B/paired355/occupancy_ledger.py" "$R" "$REF"
