"""A real BF16 BertLayer with only its three Q/K/V linears replaced by independently exported W8A8 linears."""

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import torch
import transformers
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertLayer
from examples.bert_quant import EncoderBlock, replace_qkv
from examples.token_quant import quantize
from paired_timing import measure
from projection_affine_panel import gate
from provenance import capture
from serving_clocks import build, clocks
from sparkfuse.resident import capture_export
from token_quant_panel import compare, digest

SPECS = ((17, False), (33, True), (129, False))
SEEDS = (1731131, 1744213)
POLICIES = {
    "independent": (False, "column_major"),
    "shared": (True, "column_major"),
    "packed": (True, "packed_fanout"),
}


def models(seed):
    torch.manual_seed(seed)
    cfg = BertConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    cfg._attn_implementation = "eager"
    dense = BertLayer(cfg).eval().to(device="cuda", dtype=torch.bfloat16)
    return EncoderBlock(dense).eval(), EncoderBlock(replace_qkv(deepcopy(dense))).eval()


def oracle(model, reqs):
    attn = model.block.attention.self
    linears = [attn.query, attn.key, attn.value]
    cpu = [lin.rhs.cpu().int() for lin in linears]
    result = []
    for inputs in reqs:
        x = inputs[0] if len(inputs) == 1 else inputs[0] + inputs[1]
        q, s = quantize(x)
        signed = q.reshape(-1, x.shape[-1]).cpu().int()
        one = []
        for linear, w in zip(linears, cpu, strict=True):
            a = signed @ w
            y = (
                ((a.cuda().float() * s.reshape(-1, 1)) * linear.channel + linear.bias)
                .to(x.dtype)
                .reshape_as(x)
            )
            item = compare((linear(x),), (y,))[0]
            assert item["different_bytes"] == 0 and item["geometry_match"]
            one.append(dict(cpu_int32=digest(a), narrow=digest(y)))
        result.append(dict(activation=digest(x), codes=digest(q), scale=digest(s), branches=one))
    return result


def case(m, res, seed, smoke):
    dense, model = models(seed)
    g = torch.Generator(device="cuda").manual_seed(seed + m + 293)
    req = [
        tuple(
            torch.randn(1, m, 768, generator=g, device="cuda", dtype=torch.bfloat16)
            for _ in range(2 if res else 1)
        )
        for _ in range(10)
    ]
    ex, au, *held = req
    ep = torch.export.export(model, ex)
    refs = [model(*a) for a in held]
    quant = oracle(model, held)
    drift = []
    for args, ref in zip(held, refs, strict=True):
        d = dense(*args)
        drift.append(
            [
                dict(
                    max_abs=(x.float() - y.float()).abs().max().item(),
                    rms=(x.float() - y.float()).square().mean().sqrt().item(),
                )
                for x, y in zip(d, ref, strict=True)
            ]
        )
    state = {n: digest(t) for n, t in model.state_dict().items()}
    inputs = [[digest(t) for t in row] for row in req]
    gates = {}
    meta = {}
    setup = {}
    build_failures = {}
    ts = profiles = serving = None
    with ExitStack() as stack:
        paths = {"eager": model}
        serve = {"eager": model}
        for name, (share, layout) in POLICIES.items():
            opts = dict(
                token_quant=True,
                activation_quant=True,
                share_activation_quant=share,
                affine_epilogues=True,
                group_epilogues=True,
                projection_epilogue=True,
                host_backend="native",
                min_ops=100000,
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
            paths[name] = f._fn
            serve[name] = f
            meta[name] = f.metadata
            setup[name] = cost
        if not smoke:
            for name, opt in [
                ("inductor", {}),
                ("cast_inductor", {"emulate_precision_casts": True}),
            ]:
                f = torch.compile(model, fullgraph=True, dynamic=False, options=opt)
                try:
                    f(*ex)
                except (
                    torch._dynamo.exc.BackendCompilerFailed,
                    torch._dynamo.exc.Unsupported,
                ) as e:
                    build_failures[name] = type(e).__name__
                    continue
                paths[name] = serve[name] = f
        for n, f in paths.items():
            ordinary = gate(f, held, refs, model.state_dict().values())
            resident = (
                gate(serve[n], held, refs, model.state_dict().values()) if n in POLICIES else None
            )
            gates[n] = dict(
                ordinary=ordinary,
                resident=resident,
                admitted=ordinary["admitted"] and (resident is None or resident["admitted"]),
            )
            assert gates[n]["admitted"] or n in ("inductor", "cast_inductor")
        if not smoke:

            def fixed(f):
                return lambda: f(*held[-1])

            ts = measure(
                {n: fixed(f) for n, f in paths.items() if gates[n]["admitted"]}, seed + m + 373
            )
            profiles = ts.pop("profiles")
            serving = clocks(
                {n: f for n, f in serve.items() if gates[n]["admitted"]}, held, seed + m + 499
            )
    assert state == {n: digest(t) for n, t in model.state_dict().items()} and inputs == [
        [digest(t) for t in row] for row in req
    ]
    return dict(
        seed=seed,
        m=m,
        residual=res,
        state_sha256=state,
        inputs_sha256=inputs,
        ref_sha256=[[digest(t) for t in row] for row in refs],
        oracle=quant,
        dense_bf16_drift=drift,
        exported_fx=str(ep.graph),
        gates=gates,
        metadata=meta,
        setup=setup,
        build_failures=build_failures,
        ordinary_profiles=profiles,
        ordinary_timing=ts,
        serving=serving,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    out = dict(
        protocol="Actual Hugging Face BertLayer, batch 1, BF16, random weights, dropout off, eager self-attention/no mask. Replace the three Q/K/V nn.Linear modules (not the attention or FFN) with independent dynamic symmetric signed W8A8 linears. Compare only against quantized eager; report dense model drift separately. Eight held-out requests; CPU INT32 Q/K/V oracle plus hidden-state and attention-probability byte gates, geometry, ownership. All three compiled arms use same native host and explicit min_ops=100000 to keep generic arithmetic in the framework. Seven shuffled ordinary and full-request windows; no timing of rejected baselines.",
        smoke=a.smoke,
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        transformers=transformers.__version__,
        provenance=capture(a.output.parent / "sources"),
        cases=[],
    )
    for seed in (1752911,) if a.smoke else SEEDS:
        for m, res in SPECS[:2] if a.smoke else SPECS:
            torch.compiler.reset()
            c = case(m, res, seed, a.smoke)
            out["cases"].append(c)
            print(
                seed,
                m,
                {n: p["count"] for n, p in (c["ordinary_profiles"] or {}).items()},
                {n: g["admitted"] for n, g in c["gates"].items()},
                (c["ordinary_timing"] or {}).get("median_us"),
                flush=True,
            )
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
