"""Isolate the host boundary: identical compiled device binaries, held-out pools and clocks."""

import argparse, json
from hashlib import sha256
from pathlib import Path
import torch
from examples.paged_decode_example import PageGatherDecode, request
from examples.ragged_pages import RaggedGatherDecode, ragged_request, PROFILES
from sparkfuse.paged_decode import compile_paged_decode
from paged_decode_panel import oracle, PATTERNS
from ragged_page_panel import independent_mask
from decode_sdpa_panel import metrics
from ordered_score_panel import binary
from token_quant_panel import digest
from serving_clocks import build, clocks
from paired_timing import measure
from provenance import capture

CONFIGS = (
    (False, (2, 8, 2, 64, 32, 31, 0, "fp16")),
    (False, (2, 6, 2, 128, 64, 4093, 1, "bf16")),
    (True, (4, 8, 2, 64, 32, 31, 4, "bf16")),
    (True, (4, 6, 2, 128, 64, 4093, 1, "fp16")),
)
SEEDS = (4309823, 4316429)


def case(ragged, cfg, seed, smoke):
    b, h, g, d, t, s, mask, dt = cfg
    m = (RaggedGatherDecode if ragged else PageGatherDecode)(s)

    def req(j, offset):
        key = seed + offset + 67 * j
        p = PATTERNS[j % 4]
        return ragged_request(cfg, key, PROFILES[j % 8], p) if ragged else request(cfg, key, p)

    a, au = req(0, 0), req(1, 17)
    ep = torch.export.export(m, a, strict=True)
    graph = str(ep.graph)
    fns = {}
    meta = {}
    bins = {}
    for n in ("split", "native"):
        f, setup = build(lambda: compile_paged_decode(ep, a, au, host_backend=n))
        ids = []
        for pg in f.programs:
            z = binary(pg)
            k = z["binary_sha256"]
            ids.append(k)
            bins[k] = z
        meta[n] = dict(
            setup=setup,
            binaries=ids,
            sources=[bins[k]["source_sha256"] for k in ids],
            scratch_bytes=f.plan.scratch,
        )
        fns[n] = f.run_bounded
    assert (
        meta["split"]["binaries"] == meta["native"]["binaries"]
        and meta["split"]["sources"] == meta["native"]["sources"]
    )
    xs = [req(j, 2071) for j in range(4 if smoke else 8)]
    fingerprint = lambda: [[digest(t) for t in r] for r in xs]
    inputs = fingerprint()
    ref = [oracle(independent_mask(r) if ragged else r, s) for r in xs]
    gold = [m(*r) for r in xs]
    gates = {}
    for n, fn in fns.items():
        ys = [fn(*r) for r in xs]
        rows = [metrics(y, o, g, r) for y, o, g, r in zip(ys, ref, gold, xs, strict=True)]
        fresh = len({y.untyped_storage().data_ptr() for y in ys}) == len(ys)
        assert fresh and all(z["structure"] and not z["outside_tolerance"] for z in rows)
        gates[n] = dict(all_live=fresh, rows=rows)
    assert [r["sha256"] for r in gates["split"]["rows"]] == [
        r["sha256"] for r in gates["native"]["rows"]
    ]
    hot = {}
    rotation = None
    alloc = {}
    if not smoke:
        fixed = {n: (lambda f=f: f(*xs[0])) for n, f in fns.items()}
        hot = measure(fixed, seed + 43)
        rotation = clocks(fns, xs, seed + 7109)
        for n, f in fixed.items():
            y, z = build(f)
            z["live_output_bytes"] = y.untyped_storage().nbytes()
            alloc[n] = z
            del y
    assert fingerprint() == inputs and str(ep.graph) == graph
    return dict(
        ragged=ragged,
        config=cfg,
        seed=seed,
        export_graph_sha256=sha256(graph.encode()).hexdigest(),
        variants=meta,
        binaries=bins,
        inputs=inputs,
        fp64=[digest(v) for v in ref],
        gates=gates,
        hot=hot,
        rotating=rotation,
        allocation=alloc,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", type=Path, required=True)
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
    for v, c in CONFIGS[::3] if a.smoke else CONFIGS:
        for seed in (4329163,) if a.smoke else SEEDS:
            r["cases"].append(case(v, c, seed, a.smoke))
            print(v, c, seed, flush=True)
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(r, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
