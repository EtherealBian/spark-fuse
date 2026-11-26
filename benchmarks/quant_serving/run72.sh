#!/usr/bin/env bash
# Run at repo root; caller supplies a pinned llama.cpp build and verified GGUF set.
set -euo pipefail
: "${SERVER:?set SERVER to llama-server}" "${WEIGHTS:?set WEIGHTS to the shard directory}" "${OUT:?set OUT to a new output directory}"
B=$(cd "$(dirname "$0")" && pwd); mkdir -p .runtime; mkdir "$OUT"  # never overwrite evidence
exec 9>.runtime/gpu-execution.lock; flock -x 9
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); if [[ -n "$busy" ]]; then echo 'GPU has another compute process, skipping'; exit 75; fi
python3 -B "$B/verify_lock.py" "$B/qwen72b-q4km.lock.json" "$WEIGHTS" > "$OUT/checkpoint.json"
case "${PANEL:-prefix}" in
    prefix|catalog) slots=1; ctx=4096 ;;
    routing) slots=2; ctx=8192 ;;
    *) echo 'PANEL must be prefix, catalog or routing'; exit 2 ;;
esac
PORT=$(python3 -c 'import socket; s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')
"$SERVER" --model "$WEIGHTS/qwen2.5-72b-instruct-q4_k_m-00001-of-00012.gguf" --host 127.0.0.1 --port "$PORT" --parallel "$slots" --ctx-size "$ctx" --batch-size 512 --ubatch-size 128 --gpu-layers all --flash-attn on --cache-ram 0 --no-webui --no-context-shift >"$OUT/server.private.log" 2>&1 &
pid=$!; trap 'kill -TERM "$pid" 2>/dev/null || :; wait "$pid" 2>/dev/null || :' EXIT; trap 'exit 143' TERM; trap 'exit 130' INT
ready=0; for _ in $(seq 1 240); do kill -0 "$pid" 2>/dev/null || { echo 'server exited'; exit 1; }; if curl --noproxy '*' -fsS --max-time 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then ready=1; break; fi; sleep 3; done
[[ "$ready" == 1 ]]; grep -E 'offloaded 81/81 layers to GPU' "$OUT/server.private.log"; grep -E 'NVIDIA GB10' "$OUT/server.private.log" | head -1
python3 -B "$B/runtime_receipt.py" --pid "$pid" --server "$SERVER" --log "$OUT/server.private.log" --layers 81 --slots "$slots" --ubatch 128 > "$OUT/runtime.json"
SPARKFUSE_EOG_IDS=$(python3 -c 'import json,sys;print(json.dumps(json.load(open(sys.argv[1]))["eog_token_ids"]))' "$OUT/runtime.json"); export SPARKFUSE_EOG_IDS
if [[ "${PANEL:-prefix}" == prefix ]]; then
    python3 -B "$B/prefix_replay.py" --port "$PORT" --output "$OUT/rows.json"
    python3 -B "$B/audit_replay.py" "$OUT/rows.json" > "$OUT/summary.json.partial"
elif [[ "$PANEL" == catalog ]]; then
    base=${SOURCE_DIR:-reports/qwen72b}
    source_rows=$(python3 -B "$B/audit_catalog.py" source "$OUT" "$base")
    python3 -B "$B/catalog_prefix.py" run --port "$PORT" --source "$source_rows" --rows "$OUT/rows.json"
    python3 -B "$B/audit_catalog.py" audit "$OUT" "$base" > "$OUT/summary.json.partial"
else
    python3 -B "$B/routed_trace.py" --port "$PORT" --output "$OUT/rows.json"
    python3 -B "$B/audit_routes.py" "$OUT/rows.json" "$OUT/trace-inputs.json" > "$OUT/summary.json.partial"
fi
mv "$OUT/summary.json.partial" "$OUT/summary.json"
