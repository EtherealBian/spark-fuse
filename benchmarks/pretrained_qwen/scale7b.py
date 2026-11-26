"""Single-process/single-backend Qwen2-7B BF16 pilot; each length starts fresh."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch._dynamo.utils import counters
from checkpoint7b import REPO, REV, verify
from metrics import p95
from resident import resident
from run import Runner, sync, timed

GEN = 128
# Separate graph warm-up (0) from two measured prompts. Fixed budget, not an EOS server.
TEXT = (
    ("The town has a school, a bridge and a library. ", "Write a detailed story about this town."),
    (
        "The observatory records the sky every night. ",
        "List 50 distinct numbered questions an astronomer could investigate, with one sentence per question.",
    ),
    (
        "A greenhouse uses water, sunlight and sensors to grow herbs. ",
        "List 50 distinct numbered experiments for this greenhouse, with one sentence per item.",
    ),
)


def inputs(tok, n, i):
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    head = enc("<|im_start|>user\n")
    body = enc(TEXT[i][0])
    tail = enc("\n" + TEXT[i][1] + "<|im_end|>\n<|im_start|>assistant\n")
    k = n - len(head) - len(tail)
    if k <= 0 or not body:
        raise ValueError("invalid prompt budget")
    return torch.tensor([head + (body * (1 + k // len(body)))[:k] + tail], device="cuda")


def trajectory(r, ids, *, record=False, teacher=None, steps=GEN):
    n = ids.size(1)
    r.cache.reset()
    sync()
    x = ids
    tokens, logits, times = [], [], []
    for step in range(steps):

        def advance():
            y = (
                r.forward(x, r.pos[:n])
                if step == 0
                else r.decode(x, r.pos[n + step - 1 : n + step])
            )
            return y, y.argmax(-1, keepdim=True)

        (y, token), ms = timed(advance)
        tokens.append(token)
        times.append(ms)
        if record:
            logits.append(y[0].detach().to(device="cpu", copy=True))
        x = token if teacher is None else teacher[step : step + 1].view(1, 1)
    return (
        torch.cat(tokens, dim=1).flatten().tolist(),
        times,
        torch.stack(logits) if record else None,
    )


def validate(actual, expected):
    a, b = actual.float(), expected.float()
    if (
        a.shape != b.shape
        or a.shape[0] != GEN
        or not torch.isfinite(a).all()
        or not torch.isfinite(b).all()
    ):
        raise ValueError("invalid logits")
    return {
        "max_abs_by_step": (a - b).abs().amax(1).tolist(),
        "cosine_by_step": torch.nn.functional.cosine_similarity(a, b, dim=1).tolist(),
    }


def storage_bytes(m):
    unique = {
        (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
        for t in (*m.parameters(), *m.buffers())
        if t.is_cuda
    }
    return sum(n for _, n in unique)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--work", required=True, type=Path, help="private eager reference logits")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--length", type=int, required=True, choices=(128, 2048))
    p.add_argument("--backend", required=True, choices=("eager", "resident", "inductor"))
    args = p.parse_args()
    verify(args.weights)
    args.work.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(617)
    tok = AutoTokenizer.from_pretrained(
        args.weights, local_files_only=True, trust_remote_code=False
    )
    t = perf_counter()
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.weights,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .eval()
        .requires_grad_(False)
    )
    sync()
    load_ms = (perf_counter() - t) * 1000
    cfg = model.config
    assert (cfg.model_type, cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size) == (
        "qwen2",
        28,
        3584,
        18944,
    )
    n, key = args.length, args.backend
    namehash = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    files = (
        "checkpoint7b.py",
        "resident.py",
        "scale7b.py",
        "projection.py",
        "run.py",
        "metrics.py",
    )
    out = dict(
        format="sparkfuse.qwen2-7b.pilot.v1",
        model=REPO,
        revision=REV,
        backend=key,
        length=n,
        generated=GEN,
        batch=1,
        dtype="bfloat16",
        attention="sdpa",
        cache_capacity=n + GEN,
        torch=str(torch.__version__),
        transformers=transformers.__version__,
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),
        source_sha256={s: namehash(Path(__file__).with_name(s)) for s in files},
        load_ms=load_ms,
        model_storage_before=storage_bytes(model),
        model_allocated_before=torch.cuda.memory_allocated(),
        validation={},
        windows=[],
    )
    fn = model
    t = perf_counter()
    if key == "resident":
        out["gate_source_sha256"] = resident(model)
    elif key == "inductor":
        fn = torch.compile(
            model, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}
        )
    gc.collect()
    sync()
    out.update(
        transform_ms=(perf_counter() - t) * 1000,
        model_storage_after=storage_bytes(model),
        model_allocated_after=torch.cuda.memory_allocated(),
        transform_peak_allocated=torch.cuda.max_memory_allocated(),
    )
    r = Runner(fn, cfg, n + GEN, torch.device("cuda"), torch.bfloat16)
    ids = [inputs(tok, n, i) for i in range(3)]

    def refpath(i):
        return args.work / f"{REV}-{n}-{i}.pt"

    if key == "eager":
        for i in (1, 2):
            gen, _, logits = trajectory(r, ids[i], record=True)
            torch.save(
                {"model": REV, "ids": ids[i].tolist()[0], "output": gen, "logits": logits},
                refpath(i),
            )
            del logits
    print("capture", key, n, flush=True)
    _, out["capture_ms"] = timed(lambda: r.setup_replay(ids[0]))
    trajectory(r, ids[0], steps=5)
    out["graphs_after_setup"] = int(counters["stats"]["unique_graphs"])
    for i in (1, 2):
        ref = torch.load(refpath(i), map_location="cpu", weights_only=True)
        assert ref["model"] == REV and ref["ids"] == ids[i].tolist()[0]
        free, _, _ = trajectory(r, ids[i])
        teacher = torch.tensor(ref["output"], device="cuda")
        _, _, y = trajectory(r, ids[i], teacher=teacher, record=True)
        v = validate(y, ref["logits"])
        v.update(input_ids=ref["ids"], reference=ref["output"], output=free)
        v["passed"] = (
            free == ref["output"]
            and min(v["cosine_by_step"]) >= 0.999
            and max(v["max_abs_by_step"]) <= 1.0
        )
        out["validation"][str(i)] = v
        del ref, y
    print("timing", key, n, [v["passed"] for v in out["validation"].values()], flush=True)
    out["setup_peak_allocated"] = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for i in (1, 2, 2, 1):
        tokens, times, _ = trajectory(r, ids[i])
        out["windows"].append(dict(prompt=i, output=tokens, ttft_ms=times[0], decode_ms=times[1:]))
    out.update(
        warm_peak_allocated=torch.cuda.max_memory_allocated(),
        warm_peak_reserved=torch.cuda.max_memory_reserved(),
        graphs_after_windows=int(counters["stats"]["unique_graphs"]),
    )
    tt = [v["ttft_ms"] for v in out["windows"]]
    d = [ms for v in out["windows"] for ms in v["decode_ms"]]
    out.update(
        ttft_p50_ms=median(tt),
        decode_p50_ms=median(d),
        decode_p95_ms=p95(d),
        eligible=all(v["passed"] for v in out["validation"].values())
        and all(
            v["output"] == out["validation"][str(v["prompt"])]["reference"] for v in out["windows"]
        )
        and out["graphs_after_windows"] == out["graphs_after_setup"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, allow_nan=False) + "\n")
    tmp.replace(args.output)
    print(
        "done",
        key,
        n,
        out["eligible"],
        out["ttft_p50_ms"],
        out["decode_p50_ms"],
        out["decode_p95_ms"],
        flush=True,
    )


if __name__ == "__main__":
    main()
