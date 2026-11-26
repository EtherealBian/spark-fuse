#!/usr/bin/env bash
set -euo pipefail
: "${SERVER:?set SERVER to the pinned CUDA llama-server}" "${MODEL:?set MODEL to the verified 120B GGUF file}" "${OUT:?set OUT to a new output directory}"
B=$(cd "$(dirname "$0")" && pwd); mkdir -p .runtime; mkdir "$OUT"
exec 9>.runtime/gpu-execution.lock; flock -x 9
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); [[ -z "$busy" ]] || { echo 'occupied GPU'; exit 75; }
python3 -B "$B/verify_lock.py" "$B/gpt-oss-120b-mxfp4.lock.json" "$MODEL" > "$OUT/checkpoint.json"
pid=''; end_server(){ if [[ -n "$pid" ]]; then kill -TERM "$pid" 2>/dev/null || :; wait "$pid" 2>/dev/null || :; pid=''; fi; }; trap end_server EXIT; trap 'exit 143' TERM; trap 'exit 130' INT
for block in 0 1 2 3; do
    ub=128; if [[ "$block" == 1 || "$block" == 2 ]]; then ub=512; fi
    port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()'); log="$OUT/block-$block.private.log"
    "$SERVER" --model "$MODEL" --host 127.0.0.1 --port "$port" --parallel 1 --ctx-size 4096 --batch-size 512 --ubatch-size "$ub" --gpu-layers all --flash-attn on --cache-ram 0 --no-webui --no-context-shift > "$log" 2>&1 & pid=$!
    ready=0; for _ in $(seq 1 240); do kill -0 "$pid" 2>/dev/null || exit 1; if curl --noproxy '*' -fsS --max-time 3 "http://127.0.0.1:$port/health" 2>/dev/null | grep -q '"ok"'; then ready=1; break; fi; sleep 3; done
    [[ "$ready" == 1 ]]; grep -Eq 'offloaded 37/37 layers to GPU' "$log"; grep -q 'NVIDIA GB10' "$log"
    python3 -B "$B/runtime_receipt.py" --pid "$pid" --server "$SERVER" --log "$log" --layers 37 --slots 1 --ubatch "$ub" > "$OUT/runtime-$block.json"
    SPARKFUSE_EOG_IDS=$(python3 -c 'import json,sys;print(json.dumps(json.load(open(sys.argv[1]))["eog_token_ids"]))' "$OUT/runtime-$block.json"); export SPARKFUSE_EOG_IDS
    python3 -B "$B/moe_blocks.py" measure --block "$block" --port "$port" --output "$OUT/block-$block.json"; end_server; echo "finished block $block microbatch $ub"
done
python3 -B "$B/moe_blocks.py" audit "$OUT" > "$OUT/summary.json.partial"; mv "$OUT/summary.json.partial" "$OUT/summary.json"
