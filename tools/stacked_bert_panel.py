"""Multiple real BertLayers in one static export; only Q/K/V linears are quantized."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import torch
import transformers
from bert_w8a8_panel import models, oracle
from examples.bert_quant import ObservableStack
from paired_timing import measure
from projection_affine_panel import gate
from provenance import capture
from serving_clocks import build, clocks
from sparkfuse.ir import Unsupported
from sparkfuse.resident import capture_export
from token_quant_panel import digest

SPECS = ((2, 17, False), (2, 33, True), (2, 129, False), (4, 33, True))
SEEDS = (1981121, 1993243)


def resident(ep, ex, au, layout, *, share=True):
    return capture_export(
        ep,
        ex,
        au,
        request_backend="native",
        int8_rhs_layout=layout,
        partition_options=dict(
            token_quant=True,
            activation_quant=True,
            share_activation_quant=share,
            affine_epilogues=True,
            group_epilogues=True,
            projection_epilogue=True,
            host_backend="native",
            min_ops=100000,
        ),
    )


def staged(parts):
    def run(*args):
        out = []
        for b in parts:
            hidden, probs = b(*args)
            out.extend((hidden, probs))
            args = (hidden,)
        return tuple(out)

    return run


def case(spec, seed, smoke):
    depth, m, residual = spec
    layers = [models(seed + 131 * i)[1] for i in range(depth)]
    model = ObservableStack(layers).eval()
    rng = torch.Generator(device="cuda").manual_seed(seed + m * depth + 23)
    req = [
        tuple(
            torch.randn(1, m, 768, dtype=torch.bfloat16, device="cuda", generator=rng)
            for _ in range(1 + int(residual))
        )
        for _ in range(10)
    ]
    ex, au, *held = req
    ep = torch.export.export(model, ex)
    fx = str(ep.graph)
    states = {n: digest(t) for n, t in model.state_dict().items()}
    inputs = [[digest(t) for t in row] for row in req]
    refs = [model(*a) for a in held]
    # Audit first and last held-out activations of EVERY layer against CPU integer GEMMs.
    audits = [0, 7]
    quant = []
    for level, layer in enumerate(layers):
        acts = [held[i] if level == 0 else (refs[i][2 * (level - 1)],) for i in audits]
        quant.append(oracle(layer, acts))
    try:
        with resident(ep, ex, au, "packed_fanout"):
            pass
    except Unsupported as e:
        refusal = str(e)
    else:
        raise AssertionError("legacy single-group policy unexpectedly accepted stack")
    build_errors = {}
    gates = {}
    setup = {}
    meta = {}
    layer_meta = []
    ordinary = serving = None
    with ExitStack() as ctx:
        paths = {"eager": model}
        requests = {"eager": model}
        for name, layout, share in [
            ("independent", "column_major", False),
            ("shared", "column_major", True),
            ("packed_groups", "packed_groups", True),
        ]:
            f, cost = build(lambda: resident(ep, ex, au, layout, share=share))
            ctx.enter_context(f)
            paths[name] = f._fn
            requests[name] = f
            setup[name] = cost
            meta[name] = f.metadata
        lhs = [ex, au]
        parts = []
        setup["layerwise_stages"] = []
        for layer in layers:
            sub = torch.export.export(layer, lhs[0])
            f, cost = build(lambda: resident(sub, *lhs, "packed_fanout"))
            ctx.enter_context(f)
            parts.append(f)
            layer_meta.append(f.metadata)
            setup["layerwise_stages"].append(cost)
            lhs = [(layer(*a)[0],) for a in lhs]
        requests["layerwise_packed"] = staged(parts)
        if not smoke:
            for name, opts in [
                ("inductor", {}),
                ("cast_inductor", {"emulate_precision_casts": True}),
            ]:
                f = torch.compile(model, fullgraph=True, dynamic=False, options=opts)
                try:
                    f(*ex)
                except (
                    torch._dynamo.exc.BackendCompilerFailed,
                    torch._dynamo.exc.Unsupported,
                ) as e:
                    build_errors[name] = type(e).__name__
                    continue
                paths[name] = requests[name] = f
        for name, f in requests.items():
            g = gate(f, held, refs, model.state_dict().values())
            q = gate(paths[name], held, refs, model.state_dict().values()) if name in meta else None
            gates[name] = dict(
                request=g, ordinary=q, admitted=g["admitted"] and (q is None or q["admitted"])
            )
            assert gates[name]["admitted"] or name in ("inductor", "cast_inductor")
        if not smoke:
            ordinary = measure(
                {
                    name: (lambda f=f: f(*held[-1]))
                    for name, f in paths.items()
                    if gates[name]["admitted"]
                },
                seed + m + depth * 17,
            )
            serving = clocks(
                {name: f for name, f in requests.items() if gates[name]["admitted"]},
                held,
                seed + m + depth * 31,
            )
    assert (
        states == {n: digest(t) for n, t in model.state_dict().items()}
        and inputs == [[digest(t) for t in row] for row in req]
        and fx == str(ep.graph)
    )
    records = [
        r for r in meta["packed_groups"]["regions"] if r.get("kind") == "packed_int32_projection"
    ]
    assert len(records) == depth and all(r["shape"] == [m, 768, 2304] for r in records)
    assert len({w["name"] for r in records for w in r["source_weights"]}) == depth * 3
    assert len({v["staging_and_state_bytes"] for v in meta.values()}) == 1
    return dict(
        depth=depth,
        m=m,
        residual=residual,
        seed=seed,
        original_fx=fx,
        source_state_sha256=states,
        input_sha256=inputs,
        ref_sha256=[[digest(t) for t in out] for out in refs],
        oracle_request_indices=audits,
        qkv_cpu_oracle_by_layer=quant,
        legacy_refusal=refusal,
        build_failures=build_errors,
        metadata=meta,
        layerwise_metadata=layer_meta,
        setup=setup,
        gates=gates,
        ordinary=ordinary,
        serving=serving,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = dict(
        protocol="Distinct random BF16 BertLayers, batch 1, no dropout or mask. Each Q/K/V uses per-token signed W8A8; FFN and attention stay in PyTorch. Public results are the hidden and attention probabilities after EVERY layer. Two admission inputs plus eight held-out independent requests. CPU INT32 oracle for each Q/K/V on the first and last held-out activations per layer. Rotating full-request latency and queue8 use seven shuffled windows, separate from graph-replay and synchronized ordinary forward. Layerwise calls use the same packed policy separately, including first-layer residual inside capture; not an ordinary-forward arm. Setup excludes export/compile of the source FX graph. No timing of byte-ineligible backends.",
        smoke=a.smoke,
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        transformers=transformers.__version__,
        provenance=capture(a.output.parent / "sources"),
        cases=[],
    )
    for seed in (2001731,) if a.smoke else SEEDS:
        for spec in SPECS[1:2] if a.smoke else SPECS:
            torch.compiler.reset()
            r = case(spec, seed, a.smoke)
            report["cases"].append(r)
            print(
                seed,
                spec,
                {n: g["admitted"] for n, g in r["gates"].items()},
                {
                    n: round(v["rotating_sync_us"], 2)
                    for n, v in (r["serving"] or {}).get("median_us", {}).items()
                },
                flush=True,
            )
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
