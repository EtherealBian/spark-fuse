"""Measure the cost of a warp-private split-KV partial; the merge and the host ABI are fixed."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from examples.length_decode import CanonicalDecode, export_length_decode
from decode_sdpa_panel import cpu_reference, inputs, metrics
from length_decode_panel import counted_inductor
from sparkfuse.length_attention import compile_length_decode
from paired_timing import measure
from serving_clocks import build, clocks
from ordered_score_panel import binary
from token_quant_panel import digest
from provenance import capture


SEEDS = (3921719, 3932693)
SIZES = (2, 31, 32, 33, 127, 513, 2049, 8193)
CONFIGS = (
    (2, 8, 2, 64, 2, "bf16"),
    (2, 6, 2, 128, 2, "bf16"),
    (1, 16, 4, 128, 0, "fp16"),
    (2, 8, 4, 64, 0, "bf16"),
)
SCHEDULES = {
    "cta32": (32, 0),
    "cta64": (64, 0),
    "warp1_t32": (32, 1),
    "warp4_t32": (32, 4),
    "warp1_t64": (64, 1),
    "warp4_t64": (64, 4),
}


def case(cfg, seed, smoke):
    b, h, g, d, mask, dt = cfg

    def make(s, j):
        xs = inputs((b, h, g, s, d, mask, dt, None), seed + j)
        return xs[:3] + ((xs[3].to(xs[0].dtype),) if mask else ())

    ex, au = make(17, 1), make(65, 13)
    m = CanonicalDecode()
    ep = export_length_decode(m, ex, upper=8193)
    paths = {"eager": m}
    descs = {}
    bins = {}
    for n, (tile, warps) in SCHEDULES.items():
        obj, setup = build(
            lambda: compile_length_decode(
                ep, ex, au, host_backend="native", tile=tile, warp_heads=warps
            )
        )
        paths[n] = obj.run_bounded
        entries = []
        for p in obj.paths:
            info = binary(SimpleNamespace(path=p, source=p.read_text()))
            k = info["binary_sha256"]
            bins[k] = info
            entries.append(k)
        descs[n] = dict(
            tile=tile,
            warps=warps,
            binaries=entries,
            setup=setup,
            peak_workspace_bytes=obj.plan.workspace_bytes(8193),
        )
    for tile in (32, 64):
        assert len({v["binaries"][1] for v in descs.values() if v["tile"] == tile}) == 1
    graphs = []
    torch._dynamo.reset()
    if not smoke:
        paths["inductor_dynamic"], graphs = counted_inductor(m, True)
    sizes = (2, 33, 127, 513) if smoke else SIZES
    req = [make(s, 101 + i * 569 + j * 31) for i, s in enumerate(sizes) for j in range(3)]
    ids = lambda: [[digest(x) for x in xs] for xs in req]
    hashes = ids()
    refs = [cpu_reference(xs, None) for xs in req]
    gold = [m(*xs) for xs in req]
    gates = {}
    for n, f in paths.items():
        ys = [f(*xs) for xs in req]
        zs = [metrics(y, r, e, xs) for y, r, e, xs in zip(ys, refs, gold, req, strict=True)]
        distinct = len({y.untyped_storage().data_ptr() for y in ys}) == len(ys)
        ok = distinct and all(z["structure"] and z["outside_tolerance"] == 0 for z in zs)
        gates[n] = dict(all_live=distinct, admitted=ok, requests=zs)
        assert n == "inductor_dynamic" or ok
    warm = len(graphs)
    allowed = {n: fn for n, fn in paths.items() if gates[n]["admitted"]}
    hot = {}
    mixed = None
    if not smoke:
        for i, s in enumerate(sizes):
            xs = req[i * 3 + 1]
            hot[str(s)] = measure(
                {n: (lambda f=f: f(*xs)) for n, f in allowed.items()}, seed + 101 * s
            )
        mixed = clocks(allowed, [req[i * 3 + 2] for i in range(8)], seed + 991)
    assert ids() == hashes and warm == len(graphs)
    return dict(
        config=cfg,
        seed=seed,
        sizes=sizes,
        qualification=[17, 65],
        inputs=hashes,
        fp64=[digest(t) for t in refs],
        variants=descs,
        binaries=bins,
        gates=gates,
        inductor_backend_calls=warm,
        hot=hot,
        mixed=mixed,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = dict(
        smoke=a.smoke,
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),
        provenance=capture(a.output.parent / "sources"),
        cases=[],
    )
    torch.set_num_threads(4)
    with torch._dynamo.config.patch(recompile_limit=32):
        for cfg in CONFIGS[:2] if a.smoke else CONFIGS:
            for seed in (3951413,) if a.smoke else SEEDS:
                c = case(cfg, seed, a.smoke)
                r["cases"].append(c)
                print(cfg, seed, {k: z["admitted"] for k, z in c["gates"].items()}, flush=True)
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(r, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
