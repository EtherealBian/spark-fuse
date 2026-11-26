"""One full 32B BF16 checkpoint per process. Eager owns the non-replayed oracle."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import accelerate
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from checkpoint32b import MODEL, PARAMETERS, REV, verify
from resident import resident
from run import Runner, sync, timed
from scale7b import inputs, storage_bytes, trajectory

STEPS = 32
LENGTHS = (128, 2048)
SOURCES = (
    "scale32b.py",
    "checkpoint32b.py",
    "qwen32b.lock.json",
    "scale7b.py",
    "run.py",
    "resident.py",
    "projection.py",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(a, b):
    if a.shape != b.shape or a.dtype != b.dtype or a.shape != (STEPS, 152064):
        raise ValueError("bad logits shape or dtype")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("non-finite logits")
    return (a.float() - b.float()).abs().amax(1).tolist(), torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def measure(model, tok, n, key, work):
    torch.cuda.reset_peak_memory_stats()
    r = Runner(model, model.config, n + STEPS, torch.device("cuda"), torch.bfloat16)
    prompts = [inputs(tok, n, i) for i in (0, 1, 2)]

    def ref(i):
        return work / f"{REV}-{n}-{i}.pt"

    if key == "eager":
        for i in (1, 2):
            tokens, _, scores = trajectory(r, prompts[i], steps=STEPS, record=True)
            torch.save(
                dict(rev=REV, ids=prompts[i].tolist()[0], tokens=tokens, scores=scores), ref(i)
            )
            del scores
    _, capture = timed(lambda: r.setup_replay(prompts[0]))
    trajectory(r, prompts[0], steps=5)
    checks = []
    for i in (1, 2):
        oracle = torch.load(ref(i), weights_only=True, map_location="cpu")
        ids, target = prompts[i].tolist()[0], oracle["tokens"]
        if oracle["rev"] != REV or oracle["ids"] != ids or len(target) != STEPS:
            raise ValueError("reference mismatch")
        tokens, _, _ = trajectory(r, prompts[i], steps=STEPS)
        _, _, scores = trajectory(
            r, prompts[i], steps=STEPS, record=True, teacher=torch.tensor(target, device="cuda")
        )
        delta, identical = compare(scores, oracle["scores"])
        checks.append(
            dict(
                prompt=i,
                input_ids=ids,
                reference=target,
                free=tokens,
                max_abs_by_step=delta,
                logits_bitwise_equal=identical,
                score_dtype=str(scores.dtype),
                logits_sha256=hashlib.sha256(
                    scores.contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
                reference_logits_sha256=hashlib.sha256(
                    oracle["scores"].contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
            )
        )
    setup_peak = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    windows = []
    for i in (1, 2, 2, 1):
        out, times, _ = trajectory(r, prompts[i], steps=STEPS)
        windows.append(dict(prompt=i, tokens=out, ttft_ms=times[0], decode_ms=times[1:]))
    return dict(
        length=n,
        capacity=n + STEPS,
        capture_ms=capture,
        setup_peak_allocated=setup_peak,
        warm_peak_allocated=torch.cuda.max_memory_allocated(),
        warm_peak_reserved=torch.cuda.max_memory_reserved(),
        checks=checks,
        windows=windows,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", required=True, choices=("eager", "resident"))
    args = p.parse_args()
    verify(args.weights)
    args.work.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(617)
    torch.set_num_threads(4)
    opts = dict(local_files_only=True, trust_remote_code=False)
    tok = AutoTokenizer.from_pretrained(args.weights, **opts)
    model, load_ms = timed(
        lambda: AutoModelForCausalLM.from_pretrained(
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
    cfg = model.config
    if (cfg.model_type, cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size) != (
        "qwen2",
        64,
        5120,
        27648,
    ):
        raise ValueError("unexpected model shape")
    if not all(p.is_cuda and p.dtype == torch.bfloat16 for p in model.parameters()):
        raise ValueError("expected full CUDA BF16 parameters; no offload")
    if not all(b.is_cuda for b in model.buffers()):
        raise ValueError("unexpected offloaded buffer")
    if sum(p.numel() for p in model.parameters()) != PARAMETERS:
        raise ValueError("parameter count mismatch")
    out = dict(
        format="sparkfuse.qwen25-32b.v1",
        model=MODEL,
        revision=REV,
        backend=args.backend,
        parameters=PARAMETERS,
        generated=STEPS,
        batch=1,
        attention="sdpa",
        dtype="bfloat16",
        layers=cfg.num_hidden_layers,
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        transformers=transformers.__version__,
        accelerate=accelerate.__version__,
        cuda=torch.version.cuda,
        source_sha256={name: sha(Path(__file__).with_name(name)) for name in SOURCES},
        load_ms=load_ms,
        model_storage_before=storage_bytes(model),
        cases=[],
    )
    print("loaded", args.backend, out["model_storage_before"], flush=True)
    torch.cuda.reset_peak_memory_stats()
    if args.backend == "resident":
        out["gate_source_sha256"], out["transform_ms"] = timed(lambda: resident(model))
    out.update(
        model_storage_after=storage_bytes(model),
        transform_peak_allocated=torch.cuda.max_memory_allocated(),
    )
    for n in LENGTHS:
        gc.collect()
        torch.cuda.empty_cache()
        sync()
        row = measure(model, tok, n, args.backend, args.work)
        out["cases"].append(row)
        print(
            "measured",
            args.backend,
            n,
            [c["logits_bitwise_equal"] for c in row["checks"]],
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".tmp")
    temp.write_text(json.dumps(out, allow_nan=False) + "\n")
    temp.replace(args.output)


if __name__ == "__main__":
    main()
