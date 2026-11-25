"""Packed SwiGLU on an actual Qwen2 decode layer, with a two-GEMM reference and cache carry."""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import random
from statistics import median

import torch
from torch import nn
import transformers
from kernel_trace import kernel_census
from mlp_projection_audit import digest
from provenance import capture
from qwen_cached_decode import check, suite, trajectory
from sparkfuse.measure import device_time, host_time
from sparkfuse.packed_swiglu import PackedSwiGLUProjection
from sparkfuse.partition import ordered_collectives, partition_export

ROOT = Path(__file__).resolve().parents[1]


class PackedMLP(nn.Module):
    def __init__(self, old):
        super().__init__()
        self.gate_up = PackedSwiGLUProjection.from_linears(old.gate_proj, old.up_proj)
        self.down_proj = old.down_proj

    def forward(self, x):
        return self.down_proj(self.gate_up(x))


def model_pair(offset):
    m, rope = suite(offset)
    p = deepcopy(m)
    p.layer.mlp = PackedMLP(p.layer.mlp)
    w = p.layer.mlp.gate_up.weight
    g, u = m.layer.mlp.gate_proj.weight, m.layer.mlp.up_proj.weight
    assert all(torch.equal(x, y) for x, y in zip(w.chunk(2, 0), (g, u), strict=True))
    assert sum(x.numel() for x in m.parameters()) == sum(x.numel() for x in p.parameters())
    mlps = list(p.layer.mlp.state_dict())
    assert mlps == ["gate_up.weight", "down_proj.weight"]
    return (
        m,
        p,
        rope,
        dict(
            old_projection_sha256=[digest(x) for x in (g, u)],
            packed_sha256=digest(w),
            packed_shape=list(w.shape),
            packed_parameter_bytes=w.numel() * w.element_size(),
            original_projection_bytes=sum(x.numel() * x.element_size() for x in (g, u)),
            converted_mlp_state_keys=mlps,
        ),
    )


def variants(orig, packed, args, verify_only, emit_dir):
    ep = torch.export.export(orig, args)
    pp = torch.export.export(packed, args)
    paths = {"eager": orig, "packed_eager": packed}
    kwargs = dict(
        rotary_pairs=True,
        paired_appends=True,
        paired_head_repeats=True,
        ordered_scores="slots",
        host_backend="native",
        schedule_policy=ordered_collectives,
    )
    summaries = {}
    for k, graph, enabled in [
        ("two_linear", ep, False),
        ("packed_framework_gate", pp, False),
        ("packed_cuda_gate", pp, True),
        ("packed_scalar_gate", pp, True),
    ]:
        g, rr = partition_export(
            graph,
            packed_gates=enabled,
            packed_gate_schedule="scalar" if k == "packed_scalar_gate" else "adaptive",
            **kwargs,
        )
        g.graph.lint()
        paths[k] = g
        hits = [r for r in rr if r.get("kind") == "packed_swiglu"]
        assert len(hits) == int(enabled)
        for r in hits:
            code = g.get_submodule(r["module"]).program.source
            assert sha256(code.encode()).hexdigest() == r["cuda_sha256"]
            if emit_dir:
                emit_dir.mkdir(parents=True, exist_ok=True)
                (emit_dir / (r["cuda_sha256"] + ".cu")).write_text(code)
        summaries[k] = dict(fx=str(g.graph), records=rr)
    if not verify_only:
        paths["inductor"] = torch.compile(
            ep.module(), backend="inductor", fullgraph=True, dynamic=False
        )
    return paths, dict(original=str(ep.graph), packed=str(pp.graph), variants=summaries)


def run(m, p, rope, prefix, offset, verify_only, emit_dir):
    export = trajectory(m, rope, prefix, 60401 + offset)
    stages = []
    graphs = []
    for args, _ in export:
        paths, g = variants(m, p, args, verify_only, emit_dir)
        stages.append(paths)
        graphs.append(g)
    seeds = [s + offset for s in (60527, 60637, 60733, 60869, 60943)]
    results = {k: dict(trajectories=[]) for k in stages[0]}
    for seed in seeds:
        steps = trajectory(m, rope, prefix, seed)
        for name, r in results.items():
            cache = steps[0][0][-2:]
            checks = []
            for stage, (a, ref) in zip(stages, steps, strict=True):
                args = (*a[:4], *cache)
                before = tuple(x.clone() for x in cache)
                try:
                    out = stage[name](*args)
                    row = check(out, ref, args, before, name != "inductor")
                    cache = out[-2:]
                except Exception as exc:
                    if name != "inductor":
                        raise
                    checks.append(dict(error=type(exc).__name__, passed=False))
                    break
                checks.append(row)
            r["trajectories"].append(
                dict(
                    seed=seed,
                    steps=checks,
                    passed=len(checks) == 2 and all(x["passed"] for x in checks),
                )
            )
    allowed = []
    for name, r in results.items():
        r["passed"] = all(t["passed"] for t in r["trajectories"])
        if r["passed"]:
            allowed.append(name)
    windows = []
    rng = random.Random(63791 + prefix)
    profiles = []
    medians = []
    if not verify_only:
        for i, (args, _) in enumerate(steps):
            paths = stages[i]
            ws = []
            for _ in range(7):
                order = allowed.copy()
                rng.shuffle(order)
                w = dict(order=order, device={}, host={})
                for name in order:

                    def f():
                        return paths[name](*args)

                    w["device"][name] = device_time(f, groups=7, unroll=4, replay=3)
                    w["host"][name] = host_time(f, samples=15)
                ws.append(w)
            windows.append(ws)
            medians.append(
                {
                    metric: {
                        k: median(median(w[metric][k]["samples_us"]) for w in ws) for k in allowed
                    }
                    for metric in ("device", "host")
                }
            )
            profiles.append({k: kernel_census(lambda: paths[k](*args)) for k in allowed})
    return dict(
        prefix=prefix,
        graphs=graphs,
        backends=results,
        windows=windows,
        median_us=medians,
        profiles=profiles,
        timing_inputs_sha256=[[digest(v) for v in a] for a, _ in steps],
        passed=all(r["passed"] for n, r in results.items() if n != "inductor"),
    )


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-offset", type=int, default=80831)
    ap.add_argument("--past", type=int, nargs="+", default=[31, 1024, 2047])
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--emit-dir", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if min(a.past) < 2 or len(set(a.past)) != len(a.past):
        ap.error("distinct prefixes >=2 required")
    m, p, rope, checkpoint = model_pair(a.seed_offset)
    d = dict(
        protocol="Actual randomly initialized BF16 Qwen2 layer. Explicit independent deployment copy changes the two bias-free MLP projections to one registered [gate;up] weight, not a hidden cache. Every other parameter has identical initial bits. Eager HF stateful prefill, five held-out trajectories, both steps carry their own output cache. All four results, strides, cache ownership, prefix and masks gated before any timing. Inductor uses its existing tolerance, other paths require bit equality. Same last-seed timing inputs, seven random-order windows; no parameter packing in the timed path. Single-layer forward, not generation or pretrained accuracy.",
        torch=str(torch.__version__),
        transformers=transformers.__version__,
        device=torch.cuda.get_device_name(),
        weight_seed=60313 + a.seed_offset,
        export_seed=60401 + a.seed_offset,
        seeds=[s + a.seed_offset for s in (60527, 60637, 60733, 60869, 60943)],
        timing_seed=60943 + a.seed_offset,
        checkpoint=checkpoint,
        verify_only=a.verify_only,
        provenance=capture(ROOT / ".runtime/packed-swiglu/sources"),
        cases=[],
    )
    for n in a.past:
        c = run(m, p, rope, n, a.seed_offset, a.verify_only, a.emit_dir)
        d["cases"].append(c)
        print(n, c["passed"], [v["device"] for v in c["median_us"]], flush=True)
    d["passed"] = all(c["passed"] for c in d["cases"])
    tmp = a.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, allow_nan=False, separators=(",", ":")) + "\n")
    tmp.replace(a.output)
    return 0 if d["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
