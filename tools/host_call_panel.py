"""Paired host-rewrite ablation: the mixed graphs use exactly the same CUDA schedules."""

import argparse
import json
import random
from pathlib import Path
from statistics import median

import torch
import transformers
from transformers import BertConfig, T5Config
from transformers.models.bert.modeling_bert import BertLayer
from transformers.models.t5.modeling_t5 import T5DenseGatedActDense
from provenance import capture
from sparkfuse.host_calls import specialize_host_calls
from sparkfuse.measure import compare, device_time, host_time
from sparkfuse.partition import ordered_collectives, partition_export
from sparkfuse.tuning import backend_id


def model(case):
    if case == "bert":
        cfg = BertConfig(hidden_size=768, intermediate_size=3072, num_attention_heads=12)
        cfg._attn_implementation = "eager"
        m = BertLayer(cfg)
        for sub in m.modules():
            if isinstance(sub, torch.nn.LayerNorm):
                sub.weight.normal_(1, 0.15)
                sub.bias.normal_(0, 0.05)
        return m
    return T5DenseGatedActDense(
        T5Config(d_model=768, d_ff=3072, dropout_rate=0, feed_forward_proj=case)
    )


@torch.inference_mode()
def run(a):
    root = Path(__file__).resolve().parents[1]
    result = {
        "scope": "Single random-weight module, BF16, batch=1, width=768; fresh output allocation and per-call device synchronization. No CUDA Graph wrapping of the host measurements. Device numbers use separate hot replay. Not full-model or service latency.",
        "torch": str(torch.__version__),
        "device": torch.cuda.get_device_name(),
        "transformers": transformers.__version__,
        "case": a.case,
        "seq": a.seq,
        "weight_seed": a.weight_seed,
        "input_seeds": a.seeds,
        "compiler": backend_id(),
        "provenance": capture(root / ".runtime/host-calls/sources"),
    }
    torch.manual_seed(a.weight_seed)
    m = model(a.case).eval().to("cuda", torch.bfloat16)
    inputs = [
        torch.randn(
            1,
            a.seq,
            768,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            device="cuda",
            dtype=torch.bfloat16,
        )
        for seed in a.seeds
    ]
    ep = torch.export.export(m, (inputs[0],))
    ordinary, r0 = partition_export(ep, host_backend="native", schedule_policy=ordered_collectives)
    fast, r1 = partition_export(ep, host_backend="native", schedule_policy=ordered_collectives)
    assert r0 == r1
    x0 = ep.module()
    x1 = ep.module()
    paths = {"eager": m, "export": x0, "export_fast": x1, "mixed": ordinary, "mixed_fast": fast}
    result.update(
        regions=r0,
        rewrites={
            "mixed_fast": specialize_host_calls(fast),
            "export_fast": specialize_host_calls(x1),
        },
        mixed_fx_before=str(ordinary.graph),
        mixed_fx_after=str(fast.graph),
    )
    assert result["rewrites"]["mixed_fast"] and any("module" in r for r in r0)
    if not a.verify_only:
        paths["inductor"] = torch.compile(m, backend="inductor", fullgraph=True, dynamic=False)
    refs = [m(x) for x in inputs]
    rows = {}
    for name, fn in paths.items():
        row = rows[name] = {"samples": []}
        for seed, x, y in zip(a.seeds, inputs, refs):
            try:
                outputs = fn(x)
                errors = compare(
                    outputs,
                    y,
                    exact=tuple(range(len(torch.utils._pytree.tree_leaves(y))))
                    if name != "inductor"
                    else (),
                )
                row["samples"].append({"seed": seed, "errors": errors})
            except AssertionError as exc:
                row["samples"].append({"seed": seed, "failure": str(exc)})
        row["passed"] = all("errors" in s for s in row["samples"])
    result["backends"] = rows
    if a.verify_only:
        return result
    eligible = [k for k in paths if rows[k]["passed"]]
    rng = random.Random(16819)
    result["orders"] = []
    for i in range(9):
        order = eligible.copy()
        rng.shuffle(order)
        result["orders"].append(order)
        x = inputs[1 + i % (len(inputs) - 1)]
        for k in order:
            fn = lambda: paths[k](x)
            # Explicit warmup after each switch, before host sampling, identical for all paths.
            for _ in range(3):
                fn()
            h = host_time(fn, samples=31)
            d = device_time(fn, groups=5, unroll=8, replay=4)
            rows[k].setdefault("windows", []).append(
                {"seed": a.seeds[1 + i % (len(inputs) - 1)], "host": h, "device": d}
            )
    for k in eligible:
        for t in ["device", "host"]:
            rows[k][t + "_us"] = median(w[t]["median_us"] for w in rows[k]["windows"])
    result["host_ablation"] = {}
    for base, test in [("mixed", "mixed_fast"), ("export", "export_fast")]:
        if base not in eligible or test not in eligible:
            continue
        ratios = [
            b["host"]["median_us"] / f["host"]["median_us"]
            for b, f in zip(rows[base]["windows"], rows[test]["windows"])
        ]
        result["host_ablation"][test] = {
            "paired_ratios": ratios,
            "median_ratio": median(ratios),
            "wins": sum(r > 1 for r in ratios),
            "rounds": len(ratios),
        }
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case", choices=["bert", "gated-gelu", "gated-gelu_pytorch_tanh"], required=True
    )
    p.add_argument("--seq", type=int, required=True)
    p.add_argument("--weight-seed", type=int, default=29771)
    p.add_argument("--seeds", nargs="+", type=int, default=[701, 5497, 25843, 38629, 47221])
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.seq < 1 or len(a.seeds) < 3 or len(set(a.seeds)) != len(a.seeds):
        p.error("positive seq, at least three distinct input seeds required")
    d = run(a)
    a.output.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    print(
        a.case,
        a.seq,
        [(k, v["passed"], v.get("host_us")) for k, v in d["backends"].items()],
        flush=True,
    )
    print(d.get("host_ablation", {}), flush=True)
    raise SystemExit(
        0
        if all(d["backends"][k]["passed"] for k in ("export", "export_fast", "mixed", "mixed_fast"))
        else 2
    )
