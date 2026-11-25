"""Preregistered byte-exact comparison of a residual quantizer and integer projection."""

import argparse
from hashlib import sha256
import json
from math import prod
from pathlib import Path
import re
import time

import torch
from examples.token_quant import ResidualQuant, QuantProjection
from paired_timing import measure
from provenance import capture
from sparkfuse.partition import partition_export

CASES = [
    ("odd", (17, 1025), torch.float16, 0),
    ("hidden", (33, 1536), torch.bfloat16, 0),
    ("wide", (1, 65, 8160), torch.bfloat16, 0),
    ("linear3d", (2, 16, 1536), torch.bfloat16, 128),
    ("linear2d", (33, 1536), torch.float16, 64),
]
SEEDS = (279239, 317419)
NAMES = ("eager", "aten", "mixed", "inductor", "cast_inductor", "fused")


def raw(t):
    return t.detach().contiguous().reshape(-1).view(torch.uint8)


def digest(t):
    return sha256(raw(t).cpu().numpy().tobytes()).hexdigest()


def meta(t):
    return [str(t.dtype), str(t.device), list(t.shape), list(t.stride()), t.storage_offset()]


def compare(xs, ys):
    assert len(xs) == len(ys)
    return [
        dict(
            dtype=str(y.dtype),
            geometry_match=meta(x) == meta(y),
            different_bytes=int((raw(x) != raw(y)).sum())
            if x.dtype == y.dtype and x.shape == y.shape
            else None,
        )
        for x, y in zip(xs, ys, strict=True)
    ]


def storage(t):
    return t.untyped_storage().data_ptr()


def witness(y, ref, x, r):
    # Recompute the standardized quotient from observable outputs, not compiler internals.
    # Coincidence with a reciprocal candidate is diagnostic, not a causal attribution.
    a = (y[1] != ref[1]).reshape(-1).nonzero()
    if not a.numel():
        return None
    i = int(a[0, 0])
    row = i // x.shape[-1]
    z = ref[0].reshape(-1)[i].float()
    s = ref[2].reshape(-1)[row]
    v = z / s
    via = z * s.reciprocal()
    measured = y[0].reshape(-1)[i].float() / y[2].reshape(-1)[row]

    def word(t):
        return hex(int(t.contiguous().view(torch.int32)) & 0xFFFFFFFF)

    return dict(
        flat_index=i,
        input_x=float(x.reshape(-1)[i]),
        input_r=float(r.reshape(-1)[i]),
        residual=float(z),
        scale_f32_bits=word(s),
        observed_scale_bits=word(y[2].reshape(-1)[row]),
        eager_quotient=float(v),
        eager_quotient_bits=word(v),
        reciprocal_product_bits=word(via),
        observed_boundary_recomputed_quotient_bits=word(measured),
        eager_code=int(ref[1].reshape(-1)[i]),
        actual_code=int(y[1].reshape(-1)[i]),
        round_even_from_div=int(v.round()),
        round_even_from_product=int(via.round()),
    )


def encoding_sweep():
    g = torch.Generator(device="cuda").manual_seed(759223)
    audit = []
    for dt in (torch.float16, torch.bfloat16):
        bits = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16).view(dt)
        for order, vals in [
            ("ordered", bits),
            ("shuffled", bits[torch.randperm(65536, generator=g, device="cuda")]),
            ("finite_only", bits[bits.isfinite()]),
        ]:
            w = 257
            n = vals.numel()
            x = torch.zeros((n + w - 1) // w * w, device="cuda", dtype=dt)
            x[:n].copy_(vals)
            x = x.view(-1, w)
            m = ResidualQuant()
            gm, rec = partition_export(
                torch.export.export(m, (x, torch.zeros_like(x))),
                host_backend="native",
                token_quant=True,
            )
            assert sum(k.get("kind") == "residual_token_quant" for k in rec) == 1
            diffs = {}
            for name, b in [
                ("zero", torch.zeros_like(x)),
                ("opposite", -x),
                ("self", x),
                ("random", torch.randn(x.shape, device=x.device, dtype=dt, generator=g)),
            ]:
                out = gm(x, b)
                ref = m(x, b)
                ex = compare(out, ref)
                assert all(e["different_bytes"] == 0 and e["geometry_match"] for e in ex)
                diffs[name] = dict(
                    different_bytes=[e["different_bytes"] for e in ex],
                    reference_sha256=[digest(y) for y in ref],
                )
            audit.append(
                dict(
                    dtype=str(dt),
                    order=order,
                    count=n,
                    width=w,
                    input_sha256=digest(x),
                    cases=diffs,
                )
            )
    return audit


def case(spec, seed, smoke):
    label, shape, dt, n = spec
    k = shape[-1]
    rng = torch.Generator(device="cuda").manual_seed(seed + sum(map(ord, label)))
    requests = []
    for off in (0, 1, 3):
        requests.append(
            tuple(
                torch.randn(prod(shape) + off, device="cuda", dtype=dt, generator=rng)[off:].view(
                    shape
                )
                for _ in range(2)
            )
        )
    if n:
        w = torch.randint(
            -127,
            128,
            (n, k) if label == "linear3d" else (k, n),
            device="cuda",
            dtype=torch.int8,
            generator=rng,
        )
        w = w.t() if label == "linear3d" else w
        cs = torch.rand(1, n, generator=rng, device="cuda", dtype=torch.float32) * 0.03 + 0.001
        m = QuantProjection(w, cs)
    else:
        m = ResidualQuant()
    ep = torch.export.export(m, requests[0])
    aten = ep.module()
    mix, mr = partition_export(ep, host_backend="native")
    fuse, fr = partition_export(ep, host_backend="native", token_quant=True)
    sites = [v for v in fr if v.get("kind") == "residual_token_quant"]
    assert len(sites) == 1 and sites[0]["output_roles"] == ["residual", "codes", "scale"]
    kernels = {name: getattr(fuse, sites[0]["module"]).program.source for name in ("candidate",)}
    assert sha256(kernels["candidate"].encode()).hexdigest() == sites[0]["cuda_sha256"]
    refs = [m(*a) for a in requests]
    independent = []
    if n:
        assert k * 127**2 < 2**31
        wc = w.cpu().int()
        for ref in refs:
            accum = ref[1].reshape(-1, k).cpu().int() @ wc
            assert torch.equal(accum, ref[3].cpu())
            independent.append(digest(accum))
    allpaths = dict(
        eager=m,
        aten=aten,
        mixed=mix,
        fused=fuse,
        inductor=torch.compile(m, fullgraph=True, dynamic=False),
        cast_inductor=torch.compile(
            m, fullgraph=True, dynamic=False, options={"emulate_precision_casts": True}
        ),
    )
    guards = {}
    admitted = {}
    inputs0 = [[digest(t) for t in a] for a in requests]
    weights = {k: digest(t) for k, t in m.named_buffers()}
    for name in NAMES:
        fn = allpaths[name]
        try:
            t0 = time.perf_counter_ns()
            first = fn(*requests[0])
            torch.cuda.synchronize()
            first_us = (time.perf_counter_ns() - t0) / 1000
            del first
            # Keep every result alive. Later requests must not alias or overwrite the first ones.
            held = [fn(*a) for a in requests]
            outs = [t for h in held for t in h]
            outside = {storage(t) for a in requests for t in a} | {storage(t) for t in m.buffers()}
            ps = [storage(t) for t in outs]
            fresh = len(ps) == len(set(ps)) and not (set(ps) & outside)
            gates = [compare(y, r) for y, r in zip(held, refs, strict=True)]
            ok = fresh and all(
                c["different_bytes"] == 0 and c["geometry_match"] for cs0 in gates for c in cs0
            )
            guards[name] = dict(
                admitted=ok,
                fresh_and_retained=fresh,
                outputs=gates,
                wrapper_first_call_us=first_us,
                first_code_witness=[
                    witness(y, r, *a) for y, r, a in zip(held, refs, requests, strict=True)
                ],
            )
            if ok:
                admitted[name] = lambda fn=fn: fn(*requests[-1])
        except (RuntimeError, ValueError, NotImplementedError) as e:
            if name not in ("inductor", "cast_inductor"):
                raise
            # Avoid exporting machine-local paths or compiler cache dumps.
            guards[name] = dict(
                admitted=False,
                error=type(e).__name__,
                message=re.sub(r"/[^\s:]+", "<path>", str(e).splitlines()[0])[:300],
            )
        assert inputs0 == [[digest(t) for t in a] for a in requests] and weights == {
            k: digest(t) for k, t in m.named_buffers()
        }
    assert {"eager", "aten", "fused"} <= admitted.keys(), guards
    result = None if smoke else measure(admitted, seed + shape[-1] + n)
    if result:
        assert (
            sum(
                v
                for k, v in result["profiles"]["fused"]["kernel_names"].items()
                if "residual_token_quant" in k
            )
            == 1
        )
    return dict(
        label=label,
        shape=shape,
        dtype=str(dt),
        channels=n,
        seed=seed,
        inputs_sha256=inputs0,
        input_offsets=[a[0].storage_offset() for a in requests],
        parameters_sha256=weights,
        weight_stride=list(w.stride()) if n else None,
        references_sha256=[[digest(t) for t in h] for h in refs],
        cpu_int32_oracle=independent,
        mixed_regions=mr,
        fused_regions=fr,
        gates=guards,
        timing=result,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    d = dict(
        protocol="Static residual add in FP16/BF16; FP32 absmax over the last dimension; signed INT8 codes, FP32 scale. Two independent seeds, three separately allocated offsets (0,1,3) held live per seed/shape. Gate every output byte, offset, stride, shape, dtype, device and fresh-storage relation before timing the third request. Warm device CUDA graph and synchronized single-call host timers reported separately across seven shuffled windows. First-call time includes lazy compilation if any, excludes prebuilt export/partition. Projection includes ATen int_mm then FP32 dequantization; its INT32 accumulators are separately checked on CPU. No checkpoint or inference-quality claim.",
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),
        smoke=a.smoke,
        provenance=capture(a.output.parent / "sources"),
        encodings=[] if a.smoke else encoding_sweep(),
        cases=[],
    )
    for seed in (723553,) if a.smoke else SEEDS:
        for spec in [CASES[3]] if a.smoke else CASES:
            row = case(spec, seed, a.smoke)
            d["cases"].append(row)
            print(
                seed,
                spec[0],
                {k: v["admitted"] for k, v in row["gates"].items()},
                row["timing"]["median_us"]["device"] if row["timing"] else {},
                flush=True,
            )
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
