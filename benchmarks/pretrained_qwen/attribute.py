"""Teacher-forced cold/split-in-place/split-and-restore evidence from one frozen runner."""

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from checkpoint32b import verify, PARAMETERS
from prefix_snapshot import PrefixFork, banks
from run import Runner
from scale7b import storage_bytes, inputs
from resident import resident
from audit_fork32 import LENS, V, need, refs
from attribution_review import source_panel, ARMS, filebase, ident, fileid, save


def capture(r, ids, p, forced, arm, vocab):
    need(arm in ARMS and len(ids) > p > 0 and len(forced) > 0, "bad probe")

    def f(t, lo):
        return r.forward(
            torch.tensor([t], device=r.pos.device, dtype=torch.long), r.pos[lo : lo + len(t)]
        )

    if arm == "whole":
        r.cache.reset()
        y = f(ids, 0)
    elif arm == "inplace":
        r.cache.reset()
        f(ids[:p], 0)
        y = f(ids[p:], p)
    else:
        s = PrefixFork(r, "private-probe", ids[:p], vocab)
        # Deliberately stale destinations. Correct restore must overwrite every prefix AND tail.
        for x in banks(r):
            x.fill_(0.75)
        j = s.restore(r, "private-probe", ids)
        y = f(ids[j:], j)
    pos = torch.tensor([p - 1, p, len(ids) - 1], device=r.pos.device)
    # Physical cache at three pre-registered VALID token positions, before decode. Not all KV.
    kv = (
        torch.stack([x[0].index_select(1, pos).permute(1, 0, 2) for x in banks(r)])
        .contiguous()
        .cpu()
    )
    logits = []
    for j, t in enumerate(forced):
        logits.append(y.cpu().clone())
        if j + 1 < len(forced):
            y = r.decode(
                torch.tensor([[t]], dtype=torch.long, device=r.pos.device),
                r.pos[len(ids) + j : len(ids) + j + 1],
            )
    return torch.cat(logits, 0), kv


def store(out, key, t):
    need(
        t.dtype == torch.bfloat16 and bool(torch.isfinite(t).all()) and sys.byteorder == "little",
        "non-finite/BF16/byte order",
    )
    b = t.contiguous().view(torch.uint8).numpy().tobytes()
    path = out / (key + ".bf16")
    with path.open("xb") as f:
        f.write(b)
    return dict(shape=list(t.shape), bytes=len(b), sha256=hashlib.sha256(b).hexdigest())


@torch.inference_mode()
def main():
    a = argparse.ArgumentParser()
    a.add_argument("--weights", type=Path, required=True)
    a.add_argument("--reference", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    args = a.parse_args()
    out = args.out
    panel = source_panel(args.reference)
    old = refs(Path("reports"))[1]
    need(torch.cuda.get_device_name() == "NVIDIA GB10", "expected GB10")
    verify(args.weights)
    opts = dict(local_files_only=True, trust_remote_code=False)
    tok = AutoTokenizer.from_pretrained(args.weights, **opts)
    torch.manual_seed(617)
    torch.set_num_threads(4)
    m = (
        AutoModelForCausalLM.from_pretrained(
            args.weights,
            **opts,
            use_safetensors=True,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .eval()
        .requires_grad_(False)
    )
    need(
        m.config.num_hidden_layers == 64
        and sum(t.numel() for t in m.parameters()) == PARAMETERS
        and all(t.is_cuda and t.dtype == torch.bfloat16 for t in m.parameters())
        and all(t.is_cuda for t in m.buffers()),
        "wrong placement",
    )
    need(
        storage_bytes(m) == old["model_storage_before"]
        and resident(m) == old["gate_source_sha256"]
        and storage_bytes(m) == old["model_storage_after"],
        "resident bank differs",
    )
    mta = dict(
        model=old["model"],
        revision=old["revision"],
        parameters=PARAMETERS,
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        transformers=__import__("transformers").__version__,
    )
    need(all(v == old[k] for k, v in mta.items()), "environment changed")
    raw = dict(
        schema="sparkfuse.fork.attribution.v1",
        meta=mta,
        reference=ident(args.reference),
        sources={
            k: fileid(Path(__file__).with_name(k))
            for k in ["attribution_review.py", "attribute.py"]
        },
        records=[],
    )
    save(out / "index.json", raw)
    (out / "requests.started").touch()
    for n in LENS:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        r = Runner(m, m.config, n + 32, torch.device("cuda"), torch.bfloat16)
        r.setup_replay(inputs(tok, n, 0))
        torch.cuda.synchronize()
        for rep in (0, 1):
            cells = [x for x in panel if x[0] == n]
            cells = cells if not rep else cells[::-1]
            for key in cells:
                orig = panel[key]
                ids = tuple(orig["ids"])
                tokens = orig["whole"]["tokens"]
                for arm in ARMS if rep == 0 else ARMS[::-1]:
                    logit_rows, k = capture(r, ids, n * 3 // 4, tokens, arm, V)
                    base = filebase(key, rep, arm)
                    row = dict(
                        cell=list(key),
                        repeat=rep,
                        arm=arm,
                        logits=store(out, base + "-logits", logit_rows),
                        kv=store(out, base + "-kv", k),
                    )
                    raw["records"].append(row)
                    save(out / "index.json", raw)
        del r
    (out / "requests.finished").touch()
    end = time.monotonic() + 15
    while time.monotonic() < end:
        q = out / "memory-guard.json"
        if q.is_file() and json.loads(q.read_bytes()).get("reason") == "workload-finished":
            break
        time.sleep(0.2)
    else:
        raise RuntimeError("physical guard did not acknowledge finish")


if __name__ == "__main__":
    main()
