"""End-to-end read-only table-to-attention costs, with no pre-gathering in timed candidates."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import torch
from examples.length_decode import CanonicalDecode, export_length_decode
from examples.paged_decode_example import PageGatherDecode, request, materialize
from sparkfuse.paged_decode import compile_paged_decode
from sparkfuse.length_attention import compile_length_decode
from decode_sdpa_panel import cpu_reference, metrics
from length_decode_panel import counted_inductor
from ordered_score_panel import binary
from token_quant_panel import digest
from serving_clocks import build, clocks
from paired_timing import measure
from provenance import capture

CONFIGS = (
    (2, 8, 2, 64, 32, 31, 2, "bf16"),
    (2, 6, 2, 128, 64, 64, 2, "bf16"),
    (2, 8, 4, 64, 32, 513, 0, "bf16"),
    (2, 8, 2, 64, 32, 4093, 2, "bf16"),
    (2, 6, 2, 128, 64, 4093, 2, "bf16"),
    (1, 16, 4, 128, 64, 8193, 0, "fp16"),
)
SEEDS = (4093111, 4117903)
PATTERNS = ("shuffled", "shared", "repeated", "endpoints")


def oracle(xs, s):
    # Interpret logical page order independently on CPU; no index_select/permute or SDPA.
    q, k, v, ids, *mask = [x.cpu() for x in xs]
    p = k.size(0)
    maps = [[max(0, min(p - 1, n)) for n in row] for row in ids.tolist()]
    logical = [
        torch.stack([torch.cat([bank[n] for n in row], dim=1)[:, :s] for row in maps])
        for bank in (k, v)
    ]
    return cpu_reference((q, *logical, *mask), None)


def case(cfg, seed, smoke):
    b, h, g, d, t, s, mask, dtype = cfg
    ex = request(cfg, seed, "shuffled")
    au = request(cfg, seed + 13, "endpoints")
    m = PageGatherDecode(s)
    ep = torch.export.export(m, ex, strict=True)
    source = str(ep.graph)
    paths = {"eager": m}
    meta = {}
    bins = {}
    for n, w in [("paged_w1", 1), ("paged_w4", 4)]:
        z, setup = build(lambda: compile_paged_decode(ep, ex, au, warp_heads=w))
        paths[n] = z.run_bounded
        ids = []
        for pg in z.programs:
            a = binary(pg)
            k = a["binary_sha256"]
            bins[k] = a
            ids.append(k)
        meta[n] = dict(warps=w, binaries=ids, setup=setup, partial_workspace_bytes=z.plan.scratch)

    def dense_example(length, j):
        v = (*cfg[:5], length, *cfg[6:])
        return materialize(request(v, seed + j, "shared"), length)

    dx, da = dense_example(17, 127), dense_example(65, 313)
    de = export_length_decode(CanonicalDecode(), dx, upper=max(s, 129))
    dense, ds = build(
        lambda: compile_length_decode(de, dx, da, tile=t, warp_heads=4, host_backend="native")
    )
    paths["gather_then_warp"] = lambda *xs: dense.run_bounded(*materialize(xs, s))
    meta["gather_then_warp"] = dict(setup=ds, tile=t, warps=4)
    graphs = []
    torch._dynamo.reset()
    if not smoke:
        paths["inductor_static"], graphs = counted_inductor(ep.module(), False)
    order = PATTERNS if smoke else PATTERNS * 2
    req = [request(cfg, seed + 1009 + j * 79, n) for j, n in enumerate(order)]
    fingerprint = lambda: [[digest(x) for x in xs] for xs in req]
    inputs = fingerprint()
    fp64 = [oracle(xs, s) for xs in req]
    gold = [m(*xs) for xs in req]
    gates = {}
    hashes = {}
    for n, f in paths.items():
        results = [f(*xs) for xs in req]
        zs = [metrics(y, r, v, xs) for y, r, v, xs in zip(results, fp64, gold, req, strict=True)]
        fresh = len({v.untyped_storage().data_ptr() for v in results}) == len(results)
        admitted = fresh and all(v["structure"] and not v["outside_tolerance"] for v in zs)
        gates[n] = dict(rows=zs, all_live=fresh, admitted=admitted)
        hashes[n] = [z["sha256"] for z in zs]
        if n != "inductor_static":
            assert admitted, (cfg, n, zs)
    warm = len(graphs)
    approved = {n: f for n, f in paths.items() if gates[n]["admitted"]}
    hot = {}
    alloc = {}
    rotation = None
    fixed = req[0]
    if not smoke:
        hot = measure({n: (lambda f=f: f(*fixed)) for n, f in approved.items()}, seed + 701)
        rotation = clocks(approved, req, seed + 8023)
        for n, f in approved.items():
            y, z = build(lambda: f(*fixed))
            z["live_output_bytes"] = y.untyped_storage().nbytes()
            alloc[n] = z
            del y
    assert len(graphs) == warm and str(ep.graph) == source and inputs == fingerprint()
    return dict(
        config=cfg,
        seed=seed,
        patterns=order,
        graph_sha256=sha256(source.encode()).hexdigest(),
        inputs=inputs,
        fp64=[digest(r) for r in fp64],
        physical_pages=ex[1].shape[0],
        dense_kv_logical_bytes=2 * b * g * s * d * ex[0].element_size(),
        variants=meta,
        binaries=bins,
        gates=gates,
        matched_materialized_bits={
            n: sum(x == y for x, y in zip(v, hashes["gather_then_warp"], strict=True))
            for n, v in hashes.items()
        },
        inductor_backend_calls=warm,
        hot=hot,
        allocation=alloc,
        rotating=rotation,
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
        for cfg in CONFIGS[:1] + CONFIGS[3:4] if a.smoke else CONFIGS:
            for seed in (4143223,) if a.smoke else SEEDS:
                c = case(cfg, seed, a.smoke)
                r["cases"].append(c)
                print(cfg, seed, c["matched_materialized_bits"], flush=True)
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(r, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
