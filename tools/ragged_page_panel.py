"""Measure the exported per-row predicate, not dense attention on a precomputed mask."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import torch
from examples.paged_decode_example import PageGatherDecode
from examples.ragged_pages import RaggedGatherDecode, ragged_request, explicit_mask, PROFILES
from sparkfuse.paged_decode import compile_paged_decode
from paged_decode_panel import oracle
from decode_sdpa_panel import metrics
from length_decode_panel import counted_inductor
from ordered_score_panel import binary
from token_quant_panel import digest
from paired_timing import measure
from serving_clocks import build, clocks
from provenance import capture

CONFIGS = (
    (4, 8, 2, 64, 32, 31, 1, "bf16"),
    (4, 6, 2, 128, 64, 257, 4, "bf16"),
    (4, 8, 2, 64, 32, 4093, 4, "bf16"),
    (4, 8, 4, 128, 64, 4093, 1, "fp16"),
)
SEEDS = (4211831, 4227139)
TABLES = (
    "shuffled",
    "shared",
    "repeated",
    "endpoints",
    "shared",
    "shuffled",
    "repeated",
    "endpoints",
)


def independent_mask(xs):
    # Build the additive mask row-by-row on CPU without using the exported arange predicate.
    q, k, v, t, length, b = xs
    bias = b.cpu().expand(q.size(0), -1, -1, -1).clone()
    s = bias.size(-1)
    for i, n in enumerate(length.tolist()):
        bias[i, :, :, max(0, min(n, s)) :] = -torch.inf
    return q, k, v, t, bias


def case(cfg, seed, smoke):
    b, h, g, d, t, s, mask, dt = cfg
    m = RaggedGatherDecode(s)
    ex = ragged_request(cfg, seed, "mixed", "shuffled")
    au = ragged_request(cfg, seed + 137, "extremes", "endpoints")
    ep = torch.export.export(m, ex, strict=True)
    graph = str(ep.graph)
    base = PageGatherDecode(s)
    a, z = map(explicit_mask, (ex, au))
    flat = torch.export.export(base, a, strict=True)
    f, fs = build(lambda: compile_paged_decode(ep, ex, au))
    e, es = build(lambda: compile_paged_decode(flat, a, z))
    paths = {
        "eager": m,
        "explicit_mask_paged": lambda *xs: e.run_bounded(*explicit_mask(xs)),
        "ragged": f.run_bounded,
    }
    meta = {}
    bins = {}
    for n, func, setup in [("ragged", f, fs), ("explicit_mask_paged", e, es)]:
        ids = []
        for pg in func.programs:
            v = binary(pg)
            k = v["binary_sha256"]
            bins[k] = v
            ids.append(k)
        meta[n] = dict(setup=setup, binaries=ids, workspace_bytes=func.plan.scratch)
    torch._dynamo.reset()
    graphs = []
    if not smoke:
        paths["inductor_static"], graphs = counted_inductor(ep.module(), False)
    ps = PROFILES if not smoke else ("full", "tiny", "empty", "extremes")
    req = [ragged_request(cfg, seed + 2017 + j * 71, p, TABLES[j]) for j, p in enumerate(ps)]
    hashes = lambda: [[digest(x) for x in xs] for xs in req]
    inputs = hashes()
    refs = [oracle(independent_mask(xs), s) for xs in req]
    gold = [m(*xs) for xs in req]
    activity = []
    for p, xs in zip(ps, req, strict=True):
        raw = xs[4].tolist()
        valid = [max(0, min(n, s)) for n in raw]
        activity.append(
            dict(
                profile=p,
                raw_lengths=raw,
                live_tokens=valid,
                live_pages=sum((n + t - 1) // t for n in valid),
                capacity_pages=b * ((s + t - 1) // t),
            )
        )
    gates = {}
    dig = {}
    for n, fn in paths.items():
        ys = [fn(*xs) for xs in req]
        rows = [
            metrics(y, ref, base, xs) for y, ref, base, xs in zip(ys, refs, gold, req, strict=True)
        ]
        unique = len({y.untyped_storage().data_ptr() for y in ys}) == len(ys)
        zeros = [
            int(y.view(torch.int16)[xs[4] <= 0].count_nonzero())
            for y, xs in zip(ys, req, strict=True)
        ]
        admitted = (
            unique
            and not any(zeros)
            and all(z["structure"] and not z["outside_tolerance"] for z in rows)
        )
        gates[n] = dict(rows=rows, nonzero_empty_row_bits=zeros, all_live=unique, admitted=admitted)
        dig[n] = [r["sha256"] for r in rows]
        if n != "inductor_static":
            assert admitted, (cfg, n, zeros, rows)
    assert dig["ragged"] == dig["explicit_mask_paged"]
    warm = len(graphs)
    ok = {n: f for n, f in paths.items() if gates[n]["admitted"]}
    hot = {}
    alloc = {}
    rot = None
    if not smoke:
        for j in (0, 1, 2, 3):
            xs = req[j]
            v = {k: (lambda f=f: f(*xs)) for k, f in ok.items()}
            hot[ps[j]] = measure(v, seed + 59 * j)
            alloc[ps[j]] = {}
            for k, fn in v.items():
                y, z = build(fn)
                z["output_storage_bytes"] = y.untyped_storage().nbytes()
                alloc[ps[j]][k] = z
                del y
        rot = clocks(ok, req, seed + 8179)
    assert len(graphs) == warm and hashes() == inputs and str(ep.graph) == graph
    return dict(
        config=cfg,
        seed=seed,
        graph_sha256=sha256(graph.encode()).hexdigest(),
        variants=meta,
        binaries=bins,
        activity=activity,
        inputs=inputs,
        fp64=[digest(r) for r in refs],
        gates=gates,
        inductor_backend_calls=warm,
        hot=hot,
        allocation=alloc,
        rotating=rot,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    torch.set_num_threads(4)
    r = dict(
        smoke=a.smoke,
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        provenance=capture(a.output.parent / "sources"),
        cases=[],
    )
    with torch._dynamo.config.patch(recompile_limit=32):
        for cfg in CONFIGS[:1] + CONFIGS[2:3] if a.smoke else CONFIGS:
            for seed in (4257683,) if a.smoke else SEEDS:
                c = case(cfg, seed, a.smoke)
                r["cases"].append(c)
                print(cfg, seed, {n: v["admitted"] for n, v in c["gates"].items()}, flush=True)
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(r, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
