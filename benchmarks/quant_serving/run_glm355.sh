#!/usr/bin/env bash
# Run at repository root. The pre-transfer receipt must have been captured before fetching weights.
set -euo pipefail
: "${SERVER:?provide the pinned llama-server}" "${WEIGHTS:?provide the two-shard directory}" "${OUT:?provide a new output directory}" "${TRANSFER_RECEIPT:?provide the original transfer admission JSON}"
D=$(cd -- "$(dirname -- "$0")" && pwd); B=$(realpath benchmarks/quant_serving); export PYTHONPATH="$B"; LOCK="$D/glm45-udiq1s.lock.json"; REF="$D/glm45-budget.json"
[[ -x "$SERVER" && -d "$WEIGHTS" && -f "$TRANSFER_RECEIPT" ]]; mkdir -p .runtime; mkdir "$OUT"
exec 9>.runtime/gpu-execution.lock; flock -x 9; test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
python3 -B "$D/audit_glm355.py" launch "$LOCK" "$WEIGHTS" --budget "$REF" > "$OUT/admission.json"
cp -- "$TRANSFER_RECEIPT" "$OUT/admission-transfer.json"
PYTHONPATH="$D:$B" python3 -B - "$LOCK" "$REF" "$OUT" <<'GATE'
import json,sys
from pathlib import Path
from audit_glm355 import read_inputs,gate
lock,bud,path=map(Path,sys.argv[1:]);_,b=read_inputs(lock,bud)
for k,n in [('transfer','admission-transfer.json'),('launch','admission.json')]:gate(k,b,json.loads((path/n).read_bytes()))
GATE
python3 -B "$B/verify_lock.py" "$LOCK" "$WEIGHTS" > "$OUT/checkpoint.json"
python3 -B "$D/audit_glm355.py" inventory "$LOCK" "$WEIGHTS" > "$OUT/header-budget.json"; cmp -s "$OUT/header-budget.json" "$REF"
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()'); log="$OUT/server.private.log"; first="$WEIGHTS/GLM-4.5-UD-IQ1_S-00001-of-00002.gguf"
"$SERVER" --model "$first" --host 127.0.0.1 --port "$PORT" --parallel 1 --ctx-size 4096 --batch-size 512 --ubatch-size 128 --gpu-layers all --flash-attn on --cache-ram 0 --no-webui --no-context-shift > "$log" 2>&1 & pid=$!; watcher=''; stop(){ if [[ -n "$watcher" ]]; then kill -TERM "$watcher" 2>/dev/null || :; wait "$watcher" 2>/dev/null || :; fi; kill -TERM "$pid" 2>/dev/null || :; wait "$pid" 2>/dev/null || :; }; trap stop EXIT; trap 'exit 143' TERM; trap 'exit 130' INT
python3 -B "$B/physical_guard.py" watch "$pid" "$SERVER" "$OUT/memory-guard.json" & watcher=$!
ready=0; for _ in $(seq 1 240); do kill -0 "$pid" 2>/dev/null || { echo 'server exited before ready'; exit 1; }; if curl --noproxy '*' -fsS --max-time 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then ready=1;break;fi;sleep 3;done
[[ "$ready" == 1 ]]; grep -E 'offloaded 94/94 layers to GPU' "$log"; python3 -B "$B/runtime_receipt.py" --pid "$pid" --server "$SERVER" --log "$log" --layers 94 --slots 1 --ubatch 128 > "$OUT/runtime.json"; SPARKFUSE_EOG_IDS=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["eog_token_ids"])' "$OUT/runtime.json"); export SPARKFUSE_EOG_IDS PYTHONPATH="$B"
touch "$OUT/requests.started"
python3 -B "$D/measure_glm355.py" run --rows "$OUT/rows.json" --runtime "$OUT/runtime.json" --port "$PORT"; python3 -B "$D/measure_glm355.py" audit --rows "$OUT/rows.json" --runtime "$OUT/runtime.json" > "$OUT/summary.partial"; touch "$OUT/requests.finished"; wait "$watcher"; watcher=''; kill -0 "$pid"; python3 -B "$B/physical_guard.py" check "$OUT/memory-guard.json"; mv "$OUT/summary.partial" "$OUT/timing.json"; python3 -B "$D/audit_glm355.py" audit "$LOCK" "$OUT" > "$OUT/summary.partial"; mv "$OUT/summary.partial" "$OUT/summary.json"
