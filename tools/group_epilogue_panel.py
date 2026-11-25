"""Matched separate versus grouped dequantization for three independent integer projections."""

import argparse
from hashlib import sha256
import json
from math import prod
from pathlib import Path

import torch
from torch import nn
from examples.quant_fanout import QuantFanout
from paired_timing import measure
from provenance import capture
from sparkfuse.partition import partition_export
from token_quant_panel import compare, digest, storage, witness

SPECS = (
    ("h17", (17, 1536), (128, 64, 64), torch.float16),
    ("b33", (1, 33, 1536), (512, 256, 256), torch.bfloat16),
    ("b129", (129, 3072), (1536, 256, 256), torch.bfloat16),
)
SEEDS = (1191169, 1201331)


class Tail(nn.Module):
    def forward(self, row, a, ca, b, cb, c, cc):
        return (a.float() * row) * ca, (b.float() * row) * cb, (c.float() * row) * cc


def case(spec, seed, scope, smoke):
    label, shape, widths, dt = spec
    k = shape[-1]
    m = prod(shape[:-1])
    g = torch.Generator(device="cuda").manual_seed(seed + m)
    req = []
    oracle = []
    if scope == "fanout":
        ws = [
            torch.randint(-128, 128, (n, k), device="cuda", generator=g, dtype=torch.int8).t()
            for n in widths
        ]
        cs = [torch.rand((1, n), device="cuda", generator=g) * 0.03 + 0.001 for n in widths]
        model = QuantFanout(ws, cs).eval()
        for off in (0, 1, 2, 3):
            req.append(
                tuple(
                    torch.randn(prod(shape) + off, device="cuda", generator=g, dtype=dt)[off:].view(
                        shape
                    )
                    for _ in range(2)
                )
            )
    else:
        model = Tail().eval()
        for off in (0, 1, 2, 3):
            row = torch.randn(m + off, device="cuda", generator=g)[off:, None]
            a = [row]
            for j, n in enumerate(widths):
                x = torch.randint(
                    -(2**31),
                    2**31 - 1,
                    (m * n + off,),
                    device="cuda",
                    generator=g,
                    dtype=torch.int32,
                )[off:].view(m, n)
                c = torch.randn(n + off, device="cuda", generator=g)[off:]
                a += [x, c if j == 1 else c[None, :]]
            req.append(tuple(a))
    inp = [[digest(t) for t in a] for a in req]
    weights = {n: digest(w) for n, w in model.named_buffers()}
    refs = [model(*r) for r in req]
    ep = torch.export.export(model, req[0])
    gm = ep.module()
    fns = dict(eager=model, aten=gm)
    records = {}
    for name, opts in [
        ("separate", dict(projection_epilogue=True)),
        ("grouped", dict(projection_epilogue=True, group_epilogues=True)),
    ]:
        fns[name], rr = partition_export(
            ep, host_backend="native", token_quant=scope == "fanout", **opts
        )
        records[name] = rr
        kinds = [v.get("kind") for v in rr]
        assert (
            kinds.count("residual_token_quant") == int(scope == "fanout")
            and kinds.count("ordered_int32_epilogue") == (3 if name == "separate" else 0)
            and kinds.count("grouped_int32_epilogue") == int(name == "grouped")
        )
    fns.update(
        inductor=torch.compile(model, fullgraph=True, dynamic=False),
        cast_inductor=torch.compile(
            model, fullgraph=True, dynamic=False, options={"emulate_precision_casts": True}
        ),
    )
    gates = {}
    admit = {}
    forbidden = {storage(t) for a in req for t in a} | {storage(t) for t in model.buffers()}
    if scope == "fanout":
        wc = [w.cpu().int() for w in ws]
        for ref in refs:
            lhs = ref[1].reshape(-1, k).cpu().int()
            outs = [lhs @ v for v in wc]
            assert all(torch.equal(x.cpu(), y) for x, y in zip(ref[3::2], outs, strict=True))
            oracle.append([digest(t) for t in outs])
    for name, fn in fns.items():
        try:
            ys = [fn(*a) for a in req]
            ptrs = [storage(t) for y in ys for t in y]
            fresh = len(set(ptrs)) == len(ptrs) and set(ptrs).isdisjoint(forbidden)
            rows = [compare(y, e) for y, e in zip(ys, refs, strict=True)]
            ok = fresh and all(
                v["geometry_match"] and v["different_bytes"] == 0 for row in rows for v in row
            )
            gates[name] = dict(admitted=ok, fresh=fresh, outputs=rows)
            if scope == "fanout":
                gates[name]["code_witnesses"] = [
                    witness(y, e, *r) for r, y, e in zip(req, ys, refs, strict=True)
                ]
            if ok:
                admit[name] = lambda fn=fn: fn(*req[-1])
        except (RuntimeError, NotImplementedError) as e:
            if name not in ("inductor", "cast_inductor"):
                raise
            gates[name] = dict(admitted=False, error=type(e).__name__)
    assert set(fns) - {"inductor", "cast_inductor"} <= admit.keys()
    assert inp == [[digest(t) for t in a] for a in req] and weights == {
        n: digest(w) for n, w in model.named_buffers()
    }
    tseed = seed + m + (231 if scope == "fanout" else 971)
    t = None if smoke else measure(admit, tseed)
    if t:
        p = t["profiles"]
        assert p["grouped"]["count"] == (5 if scope == "fanout" else 1) and p["separate"][
            "count"
        ] == (7 if scope == "fanout" else 3)
    return dict(
        label=label,
        scope=scope,
        seed=seed,
        shape=list(shape),
        widths=list(widths),
        dtype=str(dt),
        export_sha256=sha256(str(ep.graph).encode()).hexdigest(),
        weights_sha256=weights,
        inputs_sha256=inp,
        reference_sha256=[[digest(x) for x in y] for y in refs],
        cpu_int32_sha256=oracle,
        timing_seed=tseed,
        records=records,
        gates=gates,
        timing=t,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    result = dict(
        protocol="Static three-branch projection or isolated three tails. First common per-token row scale, then separate per-channel scales; distinct visible outputs. Four retained requests with offsets 0/1/2/3. Signed GEMMs keep separate, column-major weights and the same PyTorch implementation in both programs. Correctness and alias/layout gate before seven shuffled device/host timing windows. Generated coefficients; not attention or model-quality measurement.",
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        smoke=a.smoke,
        provenance=capture(a.output.parent / "sources"),
        cases=[],
    )
    for seed in (1220141,) if a.smoke else SEEDS:
        for spec in SPECS[:1] if a.smoke else SPECS:
            for scope in ("tails", "fanout"):
                torch.compiler.reset()
                c = case(spec, seed, scope, a.smoke)
                result["cases"].append(c)
                print(
                    seed,
                    spec[0],
                    scope,
                    c["timing"]["median_us"]["device"] if c["timing"] else "checked",
                    flush=True,
                )
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
