"""Measured ablation of the INT32 dequantization tail, isolated and across a framework GEMM."""

import argparse
import json
from math import prod
from pathlib import Path

import torch
from torch import nn
from examples.token_quant import QuantProjection
from paired_timing import measure
from provenance import capture
from sparkfuse.partition import partition_export
from token_quant_panel import compare, digest, storage, witness

SPECS = (
    ("b16", (2, 16, 1536), 128, torch.bfloat16),
    ("h33", (33, 1536), 64, torch.float16),
    ("b64", (2, 32, 3072), 256, torch.bfloat16),
)
SEEDS = (903757, 928043)


class Tail(nn.Module):
    def forward(self, acc, row, channel):
        return ((acc.float() * row) * channel,)


def analyze(fs, m, requests, scope):
    refs = [m(*args) for args in requests]
    gates = {}
    passed = {}
    before = [[digest(x) for x in args] for args in requests]
    weights = {k: digest(t) for k, t in m.named_buffers()}
    for name, fn in fs.items():
        try:
            ys = [fn(*a) for a in requests]
            ps = [storage(t) for y in ys for t in y]
            other = {storage(t) for a in requests for t in a} | {storage(t) for t in m.buffers()}
            fresh = len(ps) == len(set(ps)) and not (set(ps) & other)
            out = [compare(y, r) for y, r in zip(ys, refs, strict=True)]
            ok = fresh and all(
                e["geometry_match"] and e["different_bytes"] == 0 for v in out for e in v
            )
            gates[name] = dict(admitted=ok, fresh=fresh, outputs=out)
            if scope == "projection":
                gates[name]["code_witnesses"] = [
                    witness(y, r, *a) for y, r, a in zip(ys, refs, requests, strict=True)
                ]
            if ok:
                passed[name] = lambda fn=fn: fn(*requests[-1])
        except (RuntimeError, NotImplementedError) as e:
            if name not in ("inductor", "cast_inductor"):
                raise
            gates[name] = dict(admitted=False, error=type(e).__name__)
        assert before == [[digest(x) for x in a] for a in requests] and weights == {
            k: digest(t) for k, t in m.named_buffers()
        }
    return gates, passed, refs, before, weights


def run_case(spec, seed, scope, smoke):
    label, sh, n, dt = spec
    k = sh[-1]
    m = prod(sh[:-1])
    rng = torch.Generator(device="cuda").manual_seed(seed + n + len(sh))
    args = []

    def rand(shape, off, kind):
        nf = prod(shape) + off
        if kind is torch.int32:
            t = torch.randint(-(2**31), 2**31 - 1, (nf,), device="cuda", generator=rng, dtype=kind)
        else:
            t = torch.randn(nf, device="cuda", generator=rng, dtype=kind)
        return t[off:].view(shape)

    if scope == "projection":
        col = label != "h33"
        w = torch.randint(
            -127, 128, (n, k) if col else (k, n), device="cuda", generator=rng, dtype=torch.int8
        )
        w = w.t() if col else w
        chan = torch.rand(1, n, generator=rng, device="cuda") * 0.03 + 0.001
        model = QuantProjection(w, chan)
        for off in (0, 1, 2, 3):
            args.append(tuple(rand(sh, off, dt) for _ in range(2)))
    else:
        model = Tail()
        for off in (0, 1, 2, 3):
            args.append(
                (
                    rand((m, n), off, torch.int32),
                    rand((m, 1), off, torch.float32),
                    rand((1, n), off, torch.float32),
                )
            )
    ep = torch.export.export(model, args[0])
    modes = {"mixed": {}, "tail": {"projection_epilogue": True}}
    if scope == "projection":
        modes.update(
            quant={"token_quant": True}, both={"projection_epilogue": True, "token_quant": True}
        )
    fns = {"eager": model, "aten": ep.module()}
    records = {}
    for key, opts in modes.items():
        fn, rr = partition_export(ep, host_backend="native", **opts)
        fns[key] = fn
        records[key] = rr
        assert sum(r.get("kind") == "residual_token_quant" for r in rr) == int(
            bool(opts.get("token_quant"))
        )
        assert sum(r.get("kind") == "ordered_int32_epilogue" for r in rr) == int(
            bool(opts.get("projection_epilogue"))
        )
    fns.update(
        inductor=torch.compile(model, fullgraph=True, dynamic=False),
        cast_inductor=torch.compile(
            model, fullgraph=True, dynamic=False, options={"emulate_precision_casts": True}
        ),
    )
    gates, passed, refs, before, weights = analyze(fns, model, args, scope)
    assert {"eager", "aten", "mixed", "tail"} <= passed.keys()
    oracle = []
    if scope == "projection":
        assert {"quant", "both"} <= passed.keys() and k * 127**2 < 2**31
        wc = w.cpu().int()
        for r in refs:
            z = r[1].reshape(-1, k).cpu().int() @ wc
            assert torch.equal(z, r[3].cpu())
            oracle.append(digest(z))
    order_seed = seed + n + (199 if scope == "projection" else 797)
    timing = None if smoke else measure(passed, order_seed)
    if timing:
        for key in ["tail", "both"] if scope == "projection" else ["tail"]:
            assert (
                sum(
                    v
                    for name, v in timing["profiles"][key]["kernel_names"].items()
                    if "ordered_int32_epilogue" in name
                )
                == 1
            )
    return dict(
        label=label,
        scope=scope,
        seed=seed,
        shape=list(sh),
        channels=n,
        narrow_dtype=str(dt),
        geometry=[[[list(x.shape), list(x.stride()), x.storage_offset()] for x in a] for a in args],
        inputs_sha256=before,
        weights_sha256=weights,
        reference_sha256=[[digest(t) for t in y] for y in refs],
        cpu_accumulator_sha256=oracle,
        gates=gates,
        records=records,
        timing_seed=order_seed,
        timing=timing,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    d = dict(
        protocol="Static FP32 dequantization y=(float_rn(acc_int32)*row_f32)*channel_f32. Isolated uses full-range signed accumulator; projection uses an unchanged framework signed int8 GEMM. Four independently allocated offsets per seed with outputs retained. Bytes, shape/strides, storage identity, and independent CPU INT32 accumulator gate before timing fourth request. Seven shuffled windows of warm device graph replay and separately synchronized host calls. Random weights without calibration or accuracy claims.",
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),
        smoke=a.smoke,
        provenance=capture(a.output.parent / "sources"),
        cases=[],
    )
    for seed in (934861,) if a.smoke else SEEDS:
        for spec in SPECS[:1] if a.smoke else SPECS:
            for scope in ("isolated", "projection"):
                c = run_case(spec, seed, scope, a.smoke)
                d["cases"].append(c)
                print(
                    seed,
                    spec[0],
                    scope,
                    c["timing"]["median_us"]["device"] if c["timing"] else "passed",
                    flush=True,
                )
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
