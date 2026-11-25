"""Paired frozen RHS layout study: unchanged signed GEMM, request ownership, end-to-end clocks."""

import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path

import torch
from examples.token_quant import QuantProjection
from paired_timing import measure
from provenance import capture
from serving_clocks import build, clocks
from sparkfuse.partition import partition_export
from sparkfuse.resident import capture_export
from token_quant_panel import compare, digest, storage, witness

SEEDS = (987143, 1002857)
SPECS = (
    ("h33", (33, 1536), 64, torch.float16, "row"),
    ("h33-control", (33, 1536), 64, torch.float16, "column"),
    ("b32", (2, 16, 1536), 128, torch.bfloat16, "row"),
    ("b65", (65, 3072), 256, torch.bfloat16, "row"),
)
OPTS = dict(host_backend="native", token_quant=True, projection_epilogue=True)


def gate(fn, reqs, refs, weights):
    before = {storage(t) for r in reqs for t in r} | {storage(w) for w in weights}
    ys = [fn(*r) for r in reqs]  # retain every visible result through all replays
    ps = [storage(y) for row in ys for y in row]
    fresh = len(ps) == len(set(ps)) and set(ps).isdisjoint(before)
    reports = [compare(y, ref) for y, ref in zip(ys, refs, strict=True)]
    return dict(
        fresh=fresh,
        outputs=reports,
        admitted=fresh
        and all(v["geometry_match"] and v["different_bytes"] == 0 for row in reports for v in row),
        code_witnesses=[witness(y, ref, *a) for y, ref, a in zip(ys, refs, reqs, strict=True)],
    )


def case(spec, seed, smoke):
    label, shape, n, dtype, source = spec
    k = shape[-1]
    # The h33 row and column control regenerate identical activations, weights and scales.
    g = torch.Generator(device="cuda").manual_seed(seed + 5 * n + k)
    w = torch.randint(-128, 128, (k, n), device="cuda", dtype=torch.int8, generator=g)
    wc = w.t().contiguous().t()
    scale = torch.rand(1, n, device="cuda", generator=g) * 0.03 + 0.001
    requests = [
        tuple(torch.randn(shape, device="cuda", dtype=dtype, generator=g) for _ in range(2))
        for _ in range(10)
    ]
    ex, audit, *reqs = requests
    inputs = [[digest(x) for x in req] for req in requests]
    assert (
        len({v for row in inputs for v in row}) == 20
        and len({storage(t) for r in requests for t in r}) == 20
    )
    model = QuantProjection(w if source == "row" else wc, scale).eval()
    weights = {name: digest(t) for name, t in model.named_buffers()}
    ep = torch.export.export(model, ex)
    raw = ep.module()
    refs = [model(*a) for a in reqs]
    cpu_w = w.cpu().int()
    oracle = []
    for ref in refs:
        z = ref[1].reshape(-1, k).cpu().int() @ cpu_w
        assert torch.equal(z, ref[3].cpu())
        oracle.append(digest(z))
    mixed, partition_cost = build(lambda: partition_export(ep, **OPTS))
    gm, recs = mixed
    assert [r.get("kind") for r in recs].count("residual_token_quant") == 1 and [
        r.get("kind") for r in recs
    ].count("ordered_int32_epilogue") == 1
    paths = {"eager": model, "aten": raw, "mixed": gm}
    setup = {"partition_mixed": partition_cost}
    meta = {}
    st = list(model.weight.stride())
    with ExitStack() as stack:
        for backend in ("mixed", "aten"):
            for layout in ("preserve", "column_major"):
                key = backend + "_" + layout
                exe, setup[key] = build(
                    lambda: capture_export(
                        ep,
                        ex,
                        audit,
                        backend=backend,
                        partition_options=OPTS if backend == "mixed" else None,
                        int8_rhs_layout=layout,
                    )
                )
                stack.enter_context(exe)
                paths[key] = exe
                md = exe.metadata
                (entry,) = md["frozen_int8_weights"]
                priv = getattr(exe._fn, entry["name"])
                actual = list(priv.stride())
                assert (
                    priv.data_ptr() != model.weight.data_ptr()
                    and actual
                    == entry["snapshot_stride"]
                    == (st if layout == "preserve" else [1, k])
                    and weights["weight"] == digest(priv)
                )
                assert md["staging_and_state_bytes"] == sum(
                    t.nbytes for t in (*ex, *model.buffers())
                )
                meta[key] = dict(
                    metadata=md,
                    actual_stride=actual,
                    actual_weight_sha256=digest(priv),
                    private_distinct=True,
                )
        paths.update(
            inductor=torch.compile(model, fullgraph=True, dynamic=False),
            cast_inductor=torch.compile(
                model, fullgraph=True, dynamic=False, options={"emulate_precision_casts": True}
            ),
        )
        gates = {}
        for name, f in paths.items():
            try:
                gates[name] = gate(f, reqs, refs, model.buffers())
            except (RuntimeError, NotImplementedError) as e:
                if name not in ("inductor", "cast_inductor"):
                    raise
                gates[name] = dict(admitted=False, error=type(e).__name__)
        admitted = {n: f for n, f in paths.items() if gates[n]["admitted"]}
        assert paths.keys() - {"inductor", "cast_inductor"} <= admitted.keys()
        timing = None if smoke else clocks(admitted, reqs, seed + n + 911)
    direct = None
    if not smoke and source == "row":
        # Graph-replay clocks are restricted to ordinary non-resident computations.
        rhs = {"row": w, "column": wc}
        q = refs[-1][1].reshape(-1, k)
        args = reqs[-1]
        fs = {}
        expected = refs[-1]
        for lay, weight in rhs.items():

            def mm(weight=weight):
                return torch._int_mm(q, weight)

            assert digest(mm()) == oracle[-1]
            fs["int_mm_" + lay] = mm
            alt = QuantProjection(weight, scale)
            mod, _ = partition_export(torch.export.export(alt, ex), **OPTS)
            out = mod(*args)
            assert all(
                e["geometry_match"] and e["different_bytes"] == 0 for e in compare(out, expected)
            )
            fs["projection_" + lay] = lambda mod=mod: mod(*args)
        direct = measure(fs, seed + n + 1871)
    assert (
        st == list(model.weight.stride())
        and weights == {name: digest(t) for name, t in model.named_buffers()}
        and inputs == [[digest(x) for x in r] for r in requests]
    )
    return dict(
        label=label,
        source_layout=source,
        shape=list(shape),
        channels=n,
        dtype=str(dtype),
        seed=seed,
        weight_sha256=weights,
        source_weight_stride=st,
        inputs_sha256=inputs,
        reference_sha256=[[digest(x) for x in y] for y in refs],
        cpu_int32_sha256=oracle,
        export_sha256=sha256(str(ep.graph).encode()).hexdigest(),
        records=recs,
        setup=setup,
        captures=meta,
        gates=gates,
        serving=timing,
        matched_device=direct,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    res = dict(
        protocol="Frozen signed W8A8 projection with observable residual, codes, scales, INT32 accumulator and dequantized output. Two qualification requests followed by eight new held-out requests, retained simultaneously. All bytes, strides, offsets and output independence gate serving clocks. Single synchronized setup per policy, warm artifact caches; includes partition/copy/capture/audits, is not pure packing time. Device factorial uses ordinary nonresident signed GEMM or whole fused projection on identical logical bytes. Seven shuffled windows, no outer graph on resident serving. Random weights; no calibration, model-quality or complete decoder inference claim.",
        smoke=a.smoke,
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),
        provenance=capture(a.output.parent / "sources"),
        cases=[],
    )
    for seed in (1051397,) if a.smoke else SEEDS:
        for spec in SPECS[:2] if a.smoke else SPECS:
            torch.compiler.reset()
            c = case(spec, seed, a.smoke)
            res["cases"].append(c)
            print(
                seed,
                c["label"],
                c["matched_device"]["median_us"]["device"] if c["matched_device"] else "checked",
                flush=True,
            )
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
