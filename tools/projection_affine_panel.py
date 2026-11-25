"""One exported graph: separate, grouped, or packed biased narrow epilogues, with fixed precision."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import torch
from examples.quant_affine import QuantAffineFanout
from examples.token_quant import quantize
from paired_timing import measure
from provenance import capture
from serving_clocks import build, clocks
from sparkfuse.resident import capture_export
from token_quant_panel import compare, digest, storage

SPECS = (
    ("short", 17, 1536, (128, 64, 64), False),
    ("visible", 33, 1536, (512, 256, 256), True),
    ("prefill", 129, 3072, (1536, 256, 256), False),
)
SEEDS = (1661987, 1674131)
POLICIES = {
    "separate": (False, "column_major"),
    "grouped_affine": (True, "column_major"),
    "packed_affine": (True, "packed_fanout"),
}


def gate(fn, reqs, refs, weights):
    ys = [fn(*a) for a in reqs]
    ptr = [storage(x) for row in ys for x in row]
    forbidden = {storage(x) for a in reqs for x in a} | {storage(w) for w in weights}
    fresh = len(set(ptr)) == len(ptr) and forbidden.isdisjoint(ptr)
    geometry = [compare(y, r) for y, r in zip(ys, refs, strict=True)]
    container = all(type(y) is tuple for y in ys)
    return dict(
        fresh=fresh,
        container=container,
        outputs=geometry,
        admitted=fresh
        and container
        and all(v["geometry_match"] and v["different_bytes"] == 0 for row in geometry for v in row),
    )


def case(spec, seed, smoke):
    label, m, k, ns, debug = spec
    g = torch.Generator(device="cuda").manual_seed(seed + m + k)
    ws = [
        torch.randint(-128, 128, (n, k), device="cuda", generator=g, dtype=torch.int8).t()
        for n in ns
    ]
    cs = [torch.rand(1, n, device="cuda", generator=g) * 0.03 + 0.001 for n in ns]
    bs = [torch.randn(n, device="cuda", generator=g) for n in ns]
    dts = [torch.float16 if i % 2 and debug else torch.bfloat16 for i in range(len(ns))]
    model = QuantAffineFanout(ws, cs, bs, dts, debug_outputs=debug).eval()
    req = [
        tuple(torch.randn(m, k, device="cuda", generator=g, dtype=torch.bfloat16) for _ in range(2))
        for _ in range(10)
    ]
    ex, au, *held = req
    ep = torch.export.export(model, ex)
    refs = [model(*a) for a in held]
    weights = {n: digest(t) for n, t in model.named_buffers()}
    inputs = [[digest(v) for v in a] for a in req]
    cpu = [w.cpu().int() for w in ws]
    oracle = []
    for args, ref in zip(held, refs, strict=True):
        q, s = quantize(args[0] + args[1])
        signed = q.cpu().int()
        ac = [signed @ w for w in cpu]
        narrow = []
        for i, (a, c, b, dt) in enumerate(zip(ac, cs, bs, dts, strict=True)):
            v = a.cuda().float()
            r = v * s
            o = r * c
            biased = o + b
            y = biased.to(dt)
            narrow.append(digest(y))
            want = ref[3 + 6 * i : 3 + 6 * (i + 1)] if debug else (ref[i],)
            have = (a.cuda(), v, r, o, biased, y) if debug else (y,)
            assert all(
                t["geometry_match"] and t["different_bytes"] == 0 for t in compare(have, want)
            )
        oracle.append(
            dict(
                codes=digest(q), scales=digest(s), cpu_int32=[digest(v) for v in ac], narrow=narrow
            )
        )
    setups = {}
    gates = {}
    meta = {}
    profiles = None
    timing = None
    servers = None
    with ExitStack() as stack:
        once = dict(eager=model)
        serve = dict(eager=model)
        for name, (affine, layout) in POLICIES.items():
            opts = dict(
                host_backend="native",
                token_quant=True,
                group_epilogues=True,
                projection_epilogue=True,
                affine_epilogues=affine,
            )
            f, cost = build(
                lambda: capture_export(
                    ep,
                    ex,
                    au,
                    partition_options=opts,
                    int8_rhs_layout=layout,
                    request_backend="native",
                )
            )
            stack.enter_context(f)
            once[name] = f._fn
            serve[name] = f
            setups[name] = cost
            meta[name] = f.metadata
        if not smoke:
            for name, opt in [
                ("inductor", {}),
                ("cast_inductor", {"emulate_precision_casts": True}),
            ]:
                once[name] = serve[name] = torch.compile(
                    model, fullgraph=True, dynamic=False, options=opt
                )
        for name, fn in once.items():
            g1 = gate(fn, held, refs, model.buffers())
            g2 = gate(serve[name], held, refs, model.buffers()) if name in POLICIES else g1
            gates[name] = dict(
                ordinary=g1,
                resident=g2 if name in POLICIES else None,
                admitted=g1["admitted"] and g2["admitted"],
            )
            assert gates[name]["admitted"] or name in ("inductor", "cast_inductor")
        if not smoke:

            def fixed(fn):
                return lambda: fn(*held[-1])

            timing = measure(
                {name: fixed(fn) for name, fn in once.items() if gates[name]["admitted"]},
                seed + m + 309,
            )
            profiles = timing.pop("profiles")
            servers = clocks(
                {name: fn for name, fn in serve.items() if gates[name]["admitted"]},
                held,
                seed + m + 557,
            )
    assert weights == {n: digest(t) for n, t in model.named_buffers()} and inputs == [
        [digest(v) for v in a] for a in req
    ]
    return dict(
        label=label,
        m=m,
        k=k,
        widths=list(ns),
        narrow_dtypes=[str(t) for t in dts],
        debug_outputs=debug,
        seed=seed,
        weight_sha256=weights,
        input_sha256=inputs,
        ref_sha256=[[digest(v) for v in row] for row in refs],
        oracle=oracle,
        setup=setups,
        metadata=meta,
        gates=gates,
        ordinary_profiles=profiles,
        ordinary_timing=timing,
        serving=servers,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    out = dict(
        protocol="Same biased frozen signed Q/K/V, all RHS already column-major; separate old tails, grouped ordered affine tails (same three integer GEMMs), and a single packed library GEMM. Native host/IO fixed for all resident paths. Two admission inputs and eight held-out requests. Independent CPU INT32 accumulators feed the ordered FP32 and narrow oracle. Explicit narrow-only and debug-visible contracts. Seven shuffled device/host ordinary-forward windows versus separate complete-request clocks; exclude non-equivalent competitors. Random weights, not full attention.",
        smoke=a.smoke,
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        provenance=capture(a.output.parent / "sources"),
        cases=[],
    )
    for seed in (1687829,) if a.smoke else SEEDS:
        for spec in SPECS[:2] if a.smoke else SPECS:
            torch.compiler.reset()
            c = case(spec, seed, a.smoke)
            out["cases"].append(c)
            print(
                seed,
                c["label"],
                {n: p["count"] for n, p in (c["ordinary_profiles"] or {}).items()},
                (c["ordinary_timing"] or {}).get("median_us"),
                flush=True,
            )
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
