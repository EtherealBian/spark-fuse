"""Frozen sibling projections: layouts matched, all three public accumulators retained."""

import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
from math import prod
from pathlib import Path

import torch
from examples.quant_fanout import QuantFanout
from group_epilogue_panel import SPECS
from paired_timing import measure
from provenance import capture
from quant_state_panel import gate
from serving_clocks import build, clocks
from sparkfuse.resident import capture_export
from token_quant_panel import digest

SEEDS = (1280471, 1294903)
OPTS = dict(host_backend="native", token_quant=True, group_epilogues=True)


def case(spec, seed, smoke):
    label, shape, widths, dt = spec
    k = shape[-1]
    m = prod(shape[:-1])
    gen = torch.Generator(device="cuda").manual_seed(seed + m)
    # Both baselines already use GB10's faster column-major layout, on identical weight bytes.
    ws = [
        torch.randint(-128, 128, (n, k), device="cuda", generator=gen, dtype=torch.int8).t()
        for n in widths
    ]
    cs = [torch.rand((1, n), device="cuda", generator=gen) * 0.03 + 0.001 for n in widths]
    model = QuantFanout(ws, cs).eval()
    requests = [
        tuple(torch.randn(shape, device="cuda", generator=gen, dtype=dt) for _ in range(2))
        for _ in range(10)
    ]
    ex, audit, *reqs = requests
    ep = torch.export.export(model, ex)
    raw = ep.module()
    refs = [model(*r) for r in reqs]
    weights = {n: digest(t) for n, t in model.named_buffers()}
    inputs = [[digest(t) for t in r] for r in requests]
    cpu = [w.cpu().int() for w in ws]
    oracle = []
    for ys in refs:
        codes = ys[1].reshape(-1, k).cpu().int()
        ys32 = [codes @ w for w in cpu]
        assert all(torch.equal(a.cpu(), b) for a, b in zip(ys[3::2], ys32, strict=True))
        oracle.append([digest(t) for t in ys32])
    paths = dict(eager=model, aten=raw)
    meta = {}
    setup = {}
    cap = sum(t.nbytes for t in (*ex, *model.buffers()))
    with ExitStack() as stack:
        for name, policy in [("grouped", "column_major"), ("packed", "packed_fanout")]:
            exe, setup[name] = build(
                lambda: capture_export(
                    ep,
                    ex,
                    audit,
                    max_static_bytes=cap,
                    int8_rhs_layout=policy,
                    partition_options=OPTS,
                )
            )
            stack.enter_context(exe)
            paths[name + "_once"] = (
                exe._fn
            )  # owned snapshot, single ordinary call outside resident replay
            paths[name + "_resident"] = exe
            md = exe.metadata
            bufs = [t for t in exe._fn.buffers() if t.dtype == torch.int8]
            assert md["staging_and_state_bytes"] == cap
            if name == "packed":
                assert len(bufs) == 1 and bufs[0].stride() == (1, k)
                views = bufs[0].split(widths, dim=1)
            else:
                views = [getattr(exe._fn, "weight_" + str(j)) for j in range(len(widths))]
            assert all(
                torch.equal(w, v)
                and w.untyped_storage().data_ptr() != v.untyped_storage().data_ptr()
                for w, v in zip(ws, views, strict=True)
            )
            meta[name] = dict(
                metadata=md,
                private_rhs_count=len(bufs),
                actual_rhs_strides=[list(t.stride()) for t in bufs],
                logical_band_sha256=[digest(t) for t in views],
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
            except (RuntimeError, NotImplementedError) as exc:
                if name not in ("inductor", "cast_inductor"):
                    raise
                gates[name] = dict(admitted=False, error=type(exc).__name__)
        admit = {n: f for n, f in paths.items() if gates[n]["admitted"]}
        assert paths.keys() - {"inductor", "cast_inductor"} <= admit.keys()
        ordinary = {
            n: (lambda f=f: f(*reqs[-1])) for n, f in admit.items() if not n.endswith("_resident")
        }
        direct = None if smoke else measure(ordinary, seed + m + 127)
        if direct:
            for n, count, marker in [
                ("packed_once", 3, "packed_int32_fanout"),
                ("grouped_once", 5, "grouped_int32_epilogue"),
            ]:
                p = direct["profiles"][n]
                assert (
                    p["count"] == count
                    and sum(v for k, v in p["kernel_names"].items() if marker in k) == 1
                )
        serve = None if smoke else clocks(admit, reqs, seed + m + 229)
    assert weights == {n: digest(t) for n, t in model.named_buffers()} and inputs == [
        [digest(t) for t in r] for r in requests
    ]
    return dict(
        label=label,
        shape=list(shape),
        widths=list(widths),
        dtype=str(dt),
        seed=seed,
        inputs_sha256=inputs,
        source_weights_sha256=weights,
        reference_sha256=[[digest(t) for t in y] for y in refs],
        cpu_int32_sha256=oracle,
        export_sha256=sha256(str(ep.graph).encode()).hexdigest(),
        setup=setup,
        captures=meta,
        gates=gates,
        ordinary=direct,
        serving=serve,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    result = dict(
        protocol="Three static signed Q/K/V-like projections (generated weights). Same already-column-major bytes. Two setup requests and eight independent held-out requests, all nine outputs retained, three independent INT32 CPU oracles. Only exact, fresh, geometry-matched paths timed in seven shuffled windows. Ordinary GPU-event graph replay excludes request staging; resident serving includes input copies, graph replay, fresh output copies and synchronization. Warm-cache setup measurement includes qualification, not just packing. Not a model-quality or full-attention benchmark.",
        smoke=a.smoke,
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        provenance=capture(a.output.parent / "sources"),
        cases=[],
    )
    for seed in (1301381,) if a.smoke else SEEDS:
        for spec in SPECS[:1] if a.smoke else SPECS:
            torch.compiler.reset()
            c = case(spec, seed, a.smoke)
            result["cases"].append(c)
            print(
                seed,
                spec[0],
                c["ordinary"]["median_us"]["device"] if c["ordinary"] else "checked",
                flush=True,
            )
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
