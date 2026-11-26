#!/usr/bin/env bash
# Run from repository root with a fresh OUT. Both arms reuse the same model, not the same KV.
set -euo pipefail
: "${WEIGHTS:?provide the pinned 32B checkpoint}" "${OUT:?provide a fresh result directory}"
D=$(cd -- "$(dirname -- "$0")" && pwd); B=$(realpath benchmarks/pretrained_qwen); : "${PYTHON:=python3}" "${REPORTS:=reports}"; source tools/env.sh
export PYTHONPATH="$B:$PYTHONPATH" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false SPARKFUSE_GUARD_HANDSHAKE=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p .runtime; exec 9>.runtime/gpu-execution.lock; flock -x 9; mkdir "$OUT"; [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
python3 -B - "$OUT/admission.json" <<'ADMIT'
import json,sys,shutil
from pathlib import Path
avail=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024; disk=shutil.disk_usage('.').free
ok=avail>=84*2**30 and disk>=2*2**30
Path(sys.argv[1]).write_text(json.dumps(dict(available_bytes=avail,required_bytes=84*2**30,free_disk_bytes=disk,required_disk_bytes=2*2**30,admitted=ok))+'\n');sys.exit(0 if ok else 75)
ADMIT
"$PYTHON" -Bu "$D/measure_native32.py" --weights "$WEIGHTS" --reports "$REPORTS" --out "$OUT" & child=$!; watcher=''
cleanup(){ if [[ -n "$watcher" ]]; then kill -TERM "$watcher" 2>/dev/null || :; wait "$watcher" 2>/dev/null || :; fi; if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || :; wait "$child" 2>/dev/null || :; fi; };trap cleanup EXIT;trap 'exit 143' TERM;trap 'exit 130' INT
python3 -B "$D/physical_guard.py" watch "$child" "$(command -v "$PYTHON")" "$OUT/memory-guard.json" & watcher=$!;wait "$child";child=''; wait "$watcher";watcher='';python3 -B "$D/physical_guard.py" check "$OUT/memory-guard.json"
python3 -B "$D/native_audit32.py" "$OUT/raw.json" "$REPORTS" > "$OUT/summary.partial"; cmp "$OUT/summary.partial" "$OUT/timing.json";mv "$OUT/summary.partial" "$OUT/summary.json"
