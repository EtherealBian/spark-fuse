"""A full-model prefix snapshot must pay for itself within two branches; fail closed on numeric drift."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from time import monotonic, sleep, perf_counter_ns
import torch
import transformers
import accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer
from checkpoint32b import verify, MODEL, REV, PARAMETERS
from prefix_snapshot import PrefixFork
from audit_fork32 import (
    ARMS,
    LENS,
    STEPS,
    V,
    EAGER,
    RESIDENT,
    BYTES_PER_POSITION,
    need,
    refs,
    workload,
    order,
    sha,
    sources,
    summarize,
)
from resident import resident
from run import Runner, sync
from scale7b import inputs, storage_bytes
import scale32b


def save(p, x):
    f = p.with_name(p.name + ".partial")
    f.write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")
    os.replace(f, p)


def request(r, scope, ids, snap, record=False):
    if snap is None:
        r.cache.reset()
        p = 0
    else:
        p = snap.restore(r, scope, ids)
    n = len(ids)
    prompt = torch.tensor([ids[p:]], dtype=torch.long, device=r.pos.device)
    y = r.forward(prompt, r.pos[p:n])
    outputs = []
    obs = []
    for j in range(STEPS):
        token = y.argmax(-1, keepdim=True)
        outputs.append(token)
        if record:
            obs.append(y.clone())
        if j + 1 < STEPS:
            y = r.decode(token, r.pos[n + j : n + j + 1])
    result = torch.cat(outputs, dim=1).flatten().cpu().tolist()
    return result, torch.cat(obs).cpu() if record else None


def elapsed(fn):
    st = perf_counter_ns()
    v = fn()
    sync()
    return v, (perf_counter_ns() - st) / 1e6


def panel(r, raw, out, o, n, i):
    ids, _ = workload(o, n, i)
    p = n * 3 // 4
    scope = f"owner-{i}"
    prefix = ids[0][:p]

    def make():
        return PrefixFork(r, scope, prefix, V)

    c = dict(
        length=n,
        family=i,
        prefix=p,
        capacity=r.pos.numel(),
        inputs=[list(t) for t in ids],
        diagnostics=[],
        pairs=[],
    )
    raw["cases"].append(c)
    snap = make()
    c["snapshot_bytes"] = snap.storage_bytes
    need(snap.storage_bytes == p * BYTES_PER_POSITION, "wrong snapshot footprint")
    for a in ARMS:
        for k in (0, 1):
            v, s = request(r, scope, ids[k], None if a == "cold" else snap, True)
            c["diagnostics"].append(
                dict(
                    arm=a,
                    branch=k,
                    tokens=v,
                    shape=list(s.shape),
                    dtype=str(s.dtype),
                    finite=bool(torch.isfinite(s).all().item()),
                    logits_sha256=hashlib.sha256(
                        s.contiguous().view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                )
            )
            save(out / "raw.json", raw)
    del snap, s
    for j in range(4):
        arms, branches = order(j)
        pair = dict(pair=j, waves=[])
        c["pairs"].append(pair)
        for a in arms:
            sync()
            start = perf_counter_ns()
            snap, h = elapsed(make) if a == "snapshot" else (None, 0)
            ts = []
            for k in branches:
                (out_ids, _), ms = elapsed(
                    lambda snapshot=snap, branch=k: request(r, scope, ids[branch], snapshot)
                )
                ts.append(dict(branch=k, wall_ms=ms, ids=out_ids))
            del snap
            sync()
            end = perf_counter_ns()
            pair["waves"].append(dict(arm=a, prepare_ms=h, targets=ts, wave_ms=(end - start) / 1e6))
            save(out / "raw.json", raw)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    o = refs(a.reports)
    old = o[1]
    verify(a.weights)
    mta = dict(
        model=MODEL,
        revision=REV,
        parameters=PARAMETERS,
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        transformers=transformers.__version__,
        cuda=torch.version.cuda,
        accelerate=accelerate.__version__,
        attention="sdpa",
        dtype="bfloat16",
        source_sha256={k: sha(Path(scale32b.__file__).with_name(k)) for k in scale32b.SOURCES},
    )
    need(all(mta[k] == old[k] for k in mta), "baseline changed")
    torch.manual_seed(617)
    torch.set_num_threads(4)
    opts = dict(local_files_only=True, trust_remote_code=False)
    tok = AutoTokenizer.from_pretrained(a.weights, **opts)
    model = (
        AutoModelForCausalLM.from_pretrained(
            a.weights,
            **opts,
            use_safetensors=True,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .eval()
        .requires_grad_(False)
    )
    mta["layers"] = model.config.num_hidden_layers
    mta["all_cuda_bf16"] = all(
        t.is_cuda and t.dtype == torch.bfloat16 for t in model.parameters()
    ) and all(t.is_cuda for t in model.buffers())
    need(
        mta["layers"] == 64
        and mta["all_cuda_bf16"]
        and sum(t.numel() for t in model.parameters()) == PARAMETERS,
        "wrong model",
    )
    mta["model_storage_before"] = storage_bytes(model)
    mta["gate_source_sha256"] = resident(model)
    sync()
    mta["model_storage_after"] = storage_bytes(model)
    need(
        all(
            mta[k] == old[k]
            for k in ("model_storage_before", "model_storage_after", "gate_source_sha256")
        ),
        "resident placement changed",
    )
    raw = dict(
        schema="sparkfuse.fork32.v1",
        oracles=[EAGER, RESIDENT],
        generated=STEPS,
        meta=mta,
        experiment_source_sha256=sources(),
        cases=[],
    )
    save(a.out / "raw.json", raw)
    (a.out / "requests.started").touch()
    for n in LENS:
        gc.collect()
        torch.cuda.empty_cache()
        sync()
        r = Runner(model, model.config, n + 32, torch.device("cuda"), torch.bfloat16)
        warm = inputs(tok, n, 0)
        r.setup_replay(warm)
        sync()
        for i in (1, 2):
            ids, _ = workload(o, n, i)
            need(tuple(inputs(tok, n, i).tolist()[0]) == ids[0], "original tokenizer changed")
            panel(r, raw, a.out, o, n, i)
        del r
    save(a.out / "timing.json", summarize(raw, o))
    (a.out / "requests.finished").touch()
    if os.getenv("SPARKFUSE_GUARD_HANDSHAKE") == "1":
        deadline = monotonic() + 12
        while monotonic() < deadline:
            f = a.out / "memory-guard.json"
            if f.is_file() and json.loads(f.read_bytes())["reason"] == "workload-finished":
                break
            sleep(0.2)
        else:
            raise RuntimeError("guard did not acknowledge end")


if __name__ == "__main__":
    main()
