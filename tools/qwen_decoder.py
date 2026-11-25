"""Paired GPU panel: can a fused residual/RMSNorm legally feed both sides of a projection?"""

import argparse
import json
import random
from pathlib import Path
from statistics import median

import torch
import transformers
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding

from kernel_trace import kernel_census
from provenance import capture
from sparkfuse.measure import compare, device_time, host_time
from sparkfuse.partition import ordered_collectives, partition_export
from sparkfuse.tuning import backend_id

WEIGHT_SEED = 30191
EXPORT_SEED = 30203
HELD_OUT = (30211, 30307, 30449, 30517, 30631, 30727)
TIMING_SEED = 30941


class Block(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = Qwen2DecoderLayer(config, 0)

    def forward(self, x, cos, sin, mask):
        return self.layer(
            x,
            attention_mask=mask,
            position_embeddings=(cos, sin),
            output_attentions=True,
            use_cache=False,
        )


def stats(actual, ref):
    assert actual.dtype == ref.dtype == torch.bfloat16 and actual.shape == ref.shape
    bits = actual.contiguous().view(torch.int16) != ref.contiguous().view(torch.int16)
    delta = (actual.float() - ref.float()).abs()
    finite = bool(delta.isfinite().all())
    try:
        compare(actual, ref)
        accepted = True
    except AssertionError as exc:
        accepted = False
        diagnostic = str(exc).strip()[:1200]
    return dict(
        shape=list(ref.shape),
        unequal_bits=int(bits.sum()),
        finite=finite,
        max_abs=float(delta.max()) if finite else None,
        tolerance_pass=accepted and finite,
        **({} if accepted else {"diagnostic": diagnostic}),
    )


def witness(model, gm, name, args):
    """Observe only the untimed calls; a hook must never replace an output."""
    saved = {}

    def ref_hook(mod, ins, out):
        saved["ref"] = (ins[0], out)

    def jit_hook(mod, ins, out):
        saved["region"] = tuple(out)

    a = model.layer.post_attention_layernorm.register_forward_hook(ref_hook)
    b = gm.get_submodule(name).register_forward_hook(jit_hook)
    try:
        model(*args)
        gm(*args)
    finally:
        a.remove()
        b.remove()
    return {
        key: stats(x, y)
        for key, x, y in zip(("residual", "normalized"), saved["region"], saved["ref"], strict=True)
    }


def rotary_witness(aten, mixed, rec, args):
    """Compare the actual Q/K values at the matching ATen nodes, before either matmul."""
    names = rec["live_outs"]
    saved = {}

    class Tap(torch.fx.Interpreter):
        def run_node(self, n):
            out = super().run_node(n)
            if n.name in names:
                saved[n.name] = out
            return out

    Tap(aten).run(*args)

    def hook(mod, ins, outs):
        saved["physical"] = outs

    handle = mixed.get_submodule(rec["module"]).register_forward_hook(hook)
    try:
        mixed(*args)
    finally:
        handle.remove()
    observed = {}
    for name, out, order, key in zip(
        names, saved["physical"], rec["physical_order"], ("query", "key"), strict=True
    ):
        logical = out.transpose(1, 2) if order == "BSHD" else out
        ref = saved[name]
        observed[key] = dict(
            **stats(logical, ref),
            actual_stride=list(logical.stride()),
            expected_stride=list(ref.stride()),
        )
    return observed


@torch.inference_mode()
def case(seq, verify_only, profile=False, paired=False, offset=0, emit_dir=None):
    cfg = Qwen2Config(
        hidden_size=1536,
        intermediate_size=8960,
        num_attention_heads=12,
        num_key_value_heads=2,
        num_hidden_layers=1,
        sliding_window=None,
        use_sliding_window=False,
        attention_dropout=0.0,
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(WEIGHT_SEED + offset)
    model = Block(cfg).eval()
    for norm in (model.layer.input_layernorm, model.layer.post_attention_layernorm):
        norm.weight.normal_(1, 0.18)
    model.to("cuda", torch.bfloat16)
    rotary = Qwen2RotaryEmbedding(cfg, device="cuda")

    def make(seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn(2, seq, 1536, generator=g, device="cuda", dtype=torch.bfloat16)
        pos = torch.arange(seq, device="cuda")[None, :] + torch.tensor([[0], [13]], device="cuda")
        c, s = rotary(x, pos)
        mask = torch.full((2, 1, seq, seq), float("-inf"), device="cuda", dtype=x.dtype).triu(1)
        mask[1, :, 1:, 0] = float("-inf")
        return x, c, s, mask

    ep = torch.export.export(model, make(EXPORT_SEED + offset))
    paths = {"eager": model, "exported_aten": ep.module()}
    graphs = {}
    variants = [("prior", False), ("split", True)] + ([("rotary", True)] if paired else [])
    for name, enable in variants:
        gm, rec = partition_export(
            ep,
            host_backend="native",
            schedule_policy=ordered_collectives,
            split_barriers=enable,
            rotary_pairs=name == "rotary",
        )
        gm.graph.lint()
        paths[name] = gm
        graphs[name] = dict(
            fx=str(gm.graph), records=rec, region_count=sum("module" in r for r in rec)
        )
    (hit,) = [
        r
        for r in graphs["split"]["records"]
        if "module" in r
        and r.get("split", {}).get("reason") == "fallback_barrier"
        and len(r["live_outs"]) == 2
    ]
    assert graphs["prior"]["region_count"] + 1 == graphs["split"]["region_count"]
    rope = None
    if paired:
        (rope,) = [r for r in graphs["rotary"]["records"] if r.get("kind") == "rotary_pair"]
        if emit_dir:
            emit_dir.mkdir(parents=True, exist_ok=True)
            (emit_dir / f"decoder-s{seq}.cu").write_text(
                paths["rotary"].get_submodule(rope["module"]).program.source
            )
    if not verify_only:
        paths["inductor"] = torch.compile(model, backend="inductor", fullgraph=True, dynamic=False)
    seeds = [s + offset for s in (*HELD_OUT, TIMING_SEED)]
    samples = {k: [] for k in paths}
    wires = []
    rotary_wires = []
    for seed in seeds:
        args = make(seed)
        ref = model(*args)
        assert (
            len(ref) == 2 and ref[0].shape == (2, seq, 1536) and ref[1].shape == (2, 12, seq, seq)
        )
        mask = args[-1].isneginf().expand_as(ref[1])
        assert bool(mask[0].any()) and bool(mask[1, :, 1:, 0].all())
        wire = witness(model, paths["split"], hit["module"], args)
        wires.append(dict(seed=seed, **wire))
        if paired:
            rotary_wires.append(
                dict(
                    seed=seed, **rotary_witness(paths["exported_aten"], paths["rotary"], rope, args)
                )
            )
        for name, fn in paths.items():
            row = dict(seed=seed)
            try:
                out = fn(*args)
                h, p = (stats(x, y) for x, y in zip(out, ref, strict=True))
                row.update(
                    hidden=h,
                    probabilities=p,
                    masked_nonzero=int(out[1][mask].count_nonzero()),
                    max_row_sum_error=float((out[1].float().sum(-1) - 1).abs().max()),
                )
                row["passed"] = (
                    h["tolerance_pass"]
                    and p["tolerance_pass"]
                    and row["masked_nonzero"] == 0
                    and row["max_row_sum_error"] <= 0.004
                    and (name == "inductor" or h["unequal_bits"] == p["unequal_bits"] == 0)
                )
            except Exception as exc:
                if name != "inductor":
                    raise
                row.update(passed=False, runtime_error=type(exc).__name__, message=str(exc)[:600])
            samples[name].append(row)
    rows = {k: dict(passed=all(r["passed"] for r in v), samples=v) for k, v in samples.items()}
    wires_pass = all(
        s[key]["unequal_bits"] == 0 and s[key]["finite"]
        for s in wires
        for key in ("residual", "normalized")
    )
    rope_pass = all(
        w[k]["unequal_bits"] == 0
        and w[k]["finite"]
        and w[k]["actual_stride"] == w[k]["expected_stride"]
        for w in rotary_wires
        for k in ("query", "key")
    )
    allowed = [
        k
        for k, v in rows.items()
        if v["passed"] and (k != "split" or wires_pass) and (k != "rotary" or rope_pass)
    ]
    orders = []
    if not verify_only:
        rng = random.Random(7103 + seq)
        for k in allowed:
            rows[k]["windows"] = []
        for i in range(7):
            order = allowed.copy()
            rng.shuffle(order)
            orders.append(order)
            for k in order:

                def f():
                    return paths[k](*args)

                rows[k]["windows"].append(
                    dict(
                        device=device_time(f, groups=7, unroll=4, replay=3),
                        host=host_time(f, samples=15),
                    )
                )
        for k in allowed:
            for metric in ("device", "host"):
                rows[k][metric + "_us"] = median(w[metric]["median_us"] for w in rows[k]["windows"])
    if profile:
        for name in allowed:
            rows[name]["profile"] = kernel_census(lambda: paths[name](*args))
    return dict(
        seq=seq,
        **({"rotary_witness": rotary_wires, "rotary_witness_pass": rope_pass} if paired else {}),
        exported_fx=str(ep.graph),
        graphs=graphs,
        residual_witness=wires,
        witnessed_module=hit["module"],
        witness_pass=wires_pass,
        orderings=orders,
        backends=rows,
        passed=wires_pass
        and (not paired or (rope_pass and rows["rotary"]["passed"]))
        and all(rows[k]["passed"] for k in ("eager", "exported_aten", "prior", "split")),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", type=int, nargs="+", default=[33])
    p.add_argument(
        "--profile",
        action="store_true",
        help="Record Kineto kernel counts separately, after measurements",
    )
    p.add_argument(
        "--rotary", action="store_true", help="Also compare a horizontally fused Q/K half-rotation"
    )
    p.add_argument("--emit-dir", type=Path, help="Save the generated rotary CUDA for each length")
    p.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Shift the weight, export, validation and timing seeds together",
    )
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if min(a.seq) < 2 or len(set(a.seq)) != len(a.seq):
        p.error("sequence lengths must be distinct and >=2")
    if a.emit_dir and not a.rotary:
        p.error("--emit-dir requires --rotary")
    root = Path(__file__).resolve().parents[1]
    doc = dict(
        scope="One actual randomly initialized Qwen2DecoderLayer; BF16 batch 2, H=1536, I=8960, 12 Q / 2 KV heads. Causal plus per-batch exclusion mask; position offsets 0/13. No KV cache, no token loop. RoPE table generation outside timed path; rotary application and all projections inside.",
        protocol="Seven shuffled windows on one held-out timing input, graph device and synchronized host measured separately. No timing for a backend failing any seed. Prior and split share native host + ordered schedules; only single-collective barrier splitting differs.",
        weight_seed=WEIGHT_SEED + a.seed_offset,
        export_seed=EXPORT_SEED + a.seed_offset,
        held_out_seeds=[s + a.seed_offset for s in HELD_OUT],
        timing_seed=TIMING_SEED + a.seed_offset,
        paired_rotary=a.rotary,
        torch=str(torch.__version__),
        transformers=transformers.__version__,
        compiler=backend_id(),
        device=torch.cuda.get_device_name(),
        verify_only=a.verify_only,
        profile=a.profile,
        provenance=capture(root / ".runtime/qwen-decoder-audit/sources"),
        cases=[],
    )
    for seq in a.seq:
        result = case(seq, a.verify_only, a.profile, a.rotary, a.seed_offset, a.emit_dir)
        doc["cases"].append(result)
        print(
            seq,
            "witness",
            result["witness_pass"],
            {
                k: (v["passed"], v.get("device_us"), v.get("host_us"))
                for k, v in result["backends"].items()
            },
            flush=True,
        )
    doc["passed"] = all(r["passed"] for r in doc["cases"])
    tmp = a.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    tmp.replace(a.output)
    raise SystemExit(0 if doc["passed"] else 2)
