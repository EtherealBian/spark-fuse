"""Measure fusion across a packed Q/K/V split and Q/K RoPE; leave GEMMs outside the pass."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
import transformers
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from mlp_projection_audit import compare, digest, measure
from provenance import capture
from qkv_projection_audit import capture_values
from qwen_cached_decode import suite
from qwen_packed_decode import run as layer_panel
from qwen_qkv_layout import Bank, bank_ep
from sparkfuse.partition import ordered_collectives, partition_export

ROOT = Path(__file__).resolve().parents[1]


class Epilogue(nn.Module):
    def __init__(self, widths, d):
        super().__init__()
        self.widths = widths
        self.d = d

    def forward(self, wide, c, s):
        q, k, v = (
            z.clone(memory_format=torch.contiguous_format) for z in wide.split(self.widths, -1)
        )
        q, k = (z.view(*wide.shape[:2], -1, self.d).transpose(1, 2) for z in (q, k))
        rq, rk = apply_rotary_pos_emb(q, k, c, s)
        return rq, rk, v


def build(e, mode, emit_dir, layer=False):
    kw = (
        dict(
            paired_appends=True,
            paired_head_repeats=True,
            ordered_scores="slots",
            schedule_policy=ordered_collectives,
        )
        if layer
        else {}
    )
    gm, rs = partition_export(
        e,
        host_backend="native",
        rotary_pairs=True,
        split_copies=mode in ("two_regions", "joint"),
        packed_rotary=mode == "joint",
        **kw,
    )
    gm.graph.lint()
    wanted = {
        "packed_rotary_copy": int(mode == "joint"),
        "split_copy3": int(mode == "two_regions"),
        "rotary_pair": int(mode in ("two_regions", "direct")),
    }
    assert all(sum(r.get("kind") == k for r in rs) == v for k, v in wanted.items()), (
        mode,
        [(r.get("kind"), r.get("module")) for r in rs],
    )
    for r in rs:
        if r.get("kind") == "packed_rotary_copy":
            src = gm.get_submodule(r["module"]).program.source
            h = sha256(src.encode()).hexdigest()
            assert h == r["cuda_sha256"]
            if emit_dir:
                emit_dir.mkdir(exist_ok=True, parents=True)
                (emit_dir / (h + ".cu")).write_text(src)
    return gm, dict(fx=str(gm.graph), records=rs)


def variants(m, bank_data, args, verify_only, emit_dir):
    params, widths = bank_data
    ep = torch.export.export(m, args)
    packed = bank_ep(ep, Bank(params, widths, "copies"), args)
    paths = {"eager": m}
    gs = {}
    for name, e in [("direct", ep), ("two_regions", packed), ("joint", packed)]:
        paths[name], gs[name] = build(e, name, emit_dir, layer=True)
    if not verify_only:
        paths["inductor"] = torch.compile(
            ep.module(), backend="inductor", fullgraph=True, dynamic=False
        )
    return paths, dict(original=str(ep.graph), packed=str(packed.graph), variants=gs)


def isolated(m, rope, params, widths, prefix, offset, verify_only, emit_dir):
    seeds = [offset + s for s in (60527, 60733, 60943)]
    traces = [capture_values(m, rope, prefix, s) for s in seeds]
    f = Epilogue(widths, m.layer.self_attn.head_dim)
    cases = []
    for stage, tag in enumerate(("prefill", "decode1", "decode2")):
        rows = []
        paths = {}
        graphs = {}
        for seed, steps in zip(seeds, traces, strict=True):
            x, raw, (c, s), _ = steps[stage]
            wide = F.linear(x, *params)
            args = (wide, c, s)
            ref = f(*args)
            before = [digest(z) for z in args]
            if not paths:
                e = torch.export.export(f, args)
                paths["eager"] = f
                for name in ("two_regions", "joint"):
                    paths[name], graphs[name] = build(e, name, emit_dir)
                if not verify_only:
                    paths["inductor"] = torch.compile(
                        e.module(), backend="inductor", fullgraph=True, dynamic=False
                    )
            methods = {}
            for n, fn in paths.items():
                try:
                    y = fn(*args)
                    ds = [compare(z, v) for z, v in zip(y, ref, strict=True)]
                    fresh = len({z.untyped_storage().data_ptr() for z in (*args, *y)}) == len(
                        {z.untyped_storage().data_ptr() for z in args}
                    ) + len(y)
                    immutable = [digest(z) for z in args] == before
                    methods[n] = dict(
                        tensors=ds,
                        fresh=fresh,
                        inputs_unchanged=immutable,
                        passed=fresh
                        and immutable
                        and all(
                            not d["unequal_bits"]
                            and d["stride_match"]
                            and d["layout"]["storage_offset"] == 0
                            for d in ds
                        ),
                    )
                except Exception as exc:
                    if n != "inductor":
                        raise
                    methods[n] = dict(passed=False, error=type(exc).__name__)
            rows.append(
                dict(
                    seed=seed,
                    input_phase_sha256=before,
                    output_sha256=[digest(z) for z in ref],
                    upstream_wide_projection_against_hf=[
                        compare(z, v) for z, v in zip(wide.split(widths, -1), raw, strict=True)
                    ],
                    methods=methods,
                )
            )
        allowed = [n for n in paths if all(row["methods"][n]["passed"] for row in rows)]
        assert {"eager", "two_regions", "joint"} <= set(allowed)
        timing = (
            measure(
                {n: (lambda n=n: paths[n](*args)) for n in allowed}, offset + prefix * 17 + stage
            )
            if not verify_only and stage != 2
            else {}
        )
        cases.append(
            dict(
                stage=tag,
                shape=list(wide.shape),
                graphs=graphs,
                rows=rows,
                timing_seed=seeds[-1] if timing else None,
                **timing,
            )
        )
    return cases


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--past", type=int, nargs="+", default=[127, 1024])
    p.add_argument("--seed-offset", type=int, default=80831)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--emit-dir", type=Path)
    a = p.parse_args()
    if min(a.past) < 2 or len(set(a.past)) != len(a.past):
        p.error("distinct prefixes >=2 required")
    m, rope = suite(a.seed_offset)
    ps = [getattr(m.layer.self_attn, n + "_proj") for n in "qkv"]
    widths = tuple(p.out_features for p in ps)
    params = tuple(
        nn.Parameter(torch.cat([getattr(p, k) for p in ps]), requires_grad=False)
        for k in ("weight", "bias")
    )
    out = dict(
        protocol="Opt-in epilogue over existing packed activations only; no GEMM rewrite. Three held-out real HF input/phase pairs per prefix and stage. All three epilogue outputs must match eager on identical wide inputs bit-for-bit including strides, offset, independent storage, and unmodified inputs before any timings. Record separately if the wide GEMM actually agrees with HF projections. Both compiled isolated paths run their full FX wrappers. Layer benchmark compares the same packed checkpoint/ATen graph with two regions or one, native host, identical other passes. Separate original HF direct hybrid/eager controls. Two state-carrying decode calls per held-out seed, each receives its own previous K/V. Existing tolerant Inductor whole-layer gate retained, timing withheld on any failure. 7 shuffled measurement windows; not end-to-end generation or pretrained accuracy.",
        torch=str(torch.__version__),
        transformers=transformers.__version__,
        device=torch.cuda.get_device_name(),
        weight_seed=60313 + a.seed_offset,
        export_seed=60401 + a.seed_offset,
        layer_seeds=[a.seed_offset + x for x in (60527, 60637, 60733, 60869, 60943)],
        checkpoint=dict(widths=widths, packed_sha256=[digest(t) for t in params]),
        verify_only=a.verify_only,
        provenance=capture(ROOT / ".runtime/packed-rotary/sources"),
        cases=[],
    )
    for prefix in a.past:
        iso = isolated(m, rope, params, widths, prefix, a.seed_offset, a.verify_only, a.emit_dir)
        case = layer_panel(
            m,
            (params, widths),
            rope,
            prefix,
            a.seed_offset,
            a.verify_only,
            a.emit_dir,
            variant_builder=variants,
        )
        case["isolated"] = iso
        out["cases"].append(case)
        print(
            prefix, case["passed"], case["median_us"], [v.get("median_us") for v in iso], flush=True
        )
    out["passed"] = all(k["passed"] for k in out["cases"])
    tmp = a.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, allow_nan=False, separators=(",", ":")) + "\n")
    tmp.replace(a.output)
    return 0 if out["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
