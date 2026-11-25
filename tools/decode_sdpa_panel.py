"""Qualify a whole ATen decode export against independent CPU FP64 math before timing."""

import argparse
from dataclasses import asdict
import json
import math
import random
from statistics import median
from sparkfuse.measure import host_time
from pathlib import Path
import torch
from examples.single_decode import SingleDecode
from sparkfuse.decode_attention import compile_decode_attention
from ordered_score_panel import binary
from paired_timing import measure
from token_quant_panel import digest
from provenance import capture

# Fixed schedules, no holdout-based dispatch; includes long-context negative controls.
SPECS = (
    (1, 16, 4, 31, 64, 0, "bf16", None),
    (1, 16, 4, 257, 64, 0, "fp16", None),
    (2, 12, 3, 1025, 128, 2, "bf16", None),
    (2, 8, 2, 513, 64, 1, "fp16", 0.1875),
    (1, 32, 8, 2049, 128, 0, "bf16", None),
    (1, 16, 4, 8193, 64, 0, "bf16", None),
    (1, 8, 8, 129, 64, 0, "fp16", None),
)
SEEDS = (2181223, 2193271)


def inputs(spec, seed):
    b, h, g, s, d, mask, dt, _ = spec
    rng = torch.Generator(device="cuda").manual_seed(seed)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[dt]
    req = [
        torch.randn(sh, device="cuda", dtype=dtype, generator=rng)
        for sh in ((b, h, 1, d), (b, g, s, d), (b, g, s, d))
    ]
    if mask:
        m = torch.randn(mask, 1, 1, s, device="cuda", generator=rng)
        m[..., ::7] = -torch.inf
        if mask > 1:
            m[0].fill_(-torch.inf)
        req.append(m)
    return tuple(req)


def cpu_reference(req, scale):
    # Independent per-KV-head einsum, CPU only; no SDPA or dense KV-head repeat.
    q, k, v = [x.to(device="cpu", dtype=torch.float64) for x in req[:3]]
    b, h, _, d = q.shape
    g = k.shape[1]
    q = q.reshape(b, g, h // g, d)
    z = torch.einsum("bgrd,bgsd->bgrs", q, k) * (scale if scale is not None else 1 / math.sqrt(d))
    if len(req) == 4:
        z += req[3].cpu().double()
    all_masked = torch.isneginf(z).all(-1, keepdim=True)
    p = z.masked_fill(all_masked, 0.0).softmax(-1).masked_fill(all_masked, 0.0)
    return torch.einsum("bgrs,bgsd->bgrd", p, v).reshape(b, h, 1, d)


def metrics(y, ref, expected, args):
    e = (y.cpu().double() - ref).abs()
    dt = y.dtype
    atol, rtol = (0.004, 0.016) if dt == torch.bfloat16 else (0.0006, 0.003)
    structure = (
        y.shape == expected.shape
        and y.stride() == expected.stride()
        and y.device == expected.device
        and dt == expected.dtype
        and y.storage_offset() == 0
        and y._base is None
        and y.untyped_storage().nbytes() == y.numel() * y.element_size()
        and all(x.untyped_storage().data_ptr() != y.untyped_storage().data_ptr() for x in args)
    )
    return dict(
        max_abs=float(e.max()),
        rmse=float(e.square().mean().sqrt()),
        atol=atol,
        rtol=rtol,
        outside_tolerance=int((~(e <= atol + rtol * ref.abs())).sum()),
        output_stride=list(y.stride()),
        structure=structure,
        sha256=digest(y),
    )


def case(spec, seed, smoke):
    b, h, g, s, d, mask, dt, scale = spec
    req = [inputs(spec, seed + i * 131) for i in range(5)]
    ex, au, *held = req
    model = SingleDecode(h != g, scale).eval()
    ep = torch.export.export(model, ex)
    graph = str(ep.graph)
    digests = [[digest(x) for x in xs] for xs in req]
    refs = [cpu_reference(xs, scale) for xs in held]
    expect = [model(*xs) for xs in held]
    paths = {"eager": model}
    meta = {}
    fail = {}
    for tile in (32,) if smoke else (32, 64, 128):
        f = compile_decode_attention(ep, ex, au, tile=tile)
        name = "tile" + str(tile)
        if tile == 32:
            default = f
        paths[name] = f.run_bounded
        meta[name] = dict(plan=asdict(f.plan), kernels=[binary(p) for p in f.programs])
        torch.testing.assert_close(
            f(*held[0]), f.run_bounded(*held[0]), rtol=0, atol=0, check_stride=True
        )
    if not smoke:
        f = torch.compile(model, fullgraph=True, dynamic=False)
        try:
            f(*ex)
        except (torch._dynamo.exc.BackendCompilerFailed, torch._dynamo.exc.Unsupported) as e:
            fail["inductor"] = type(e).__name__
        else:
            paths["inductor"] = f
    gates = {}
    for name, f in paths.items():
        outs = [f(*xs) for xs in held]
        unique = len({x.untyped_storage().data_ptr() for x in outs}) == len(outs)
        m = [metrics(y, r, e, xs) for y, r, e, xs in zip(outs, refs, expect, held, strict=True)]
        passed = unique and all(z["structure"] and z["outside_tolerance"] == 0 for z in m)
        gates[name] = dict(admitted=passed, independent_storage=unique, requests=m)
        if name != "inductor":
            assert passed, (name, m)
    clocks = (
        measure(
            {k: (lambda f=f: f(*held[-1])) for k, f in paths.items() if gates[k]["admitted"]},
            seed + s,
        )
        if not smoke
        else None
    )
    checked_cost = None
    if not smoke:
        # Domain scans synchronize, so this comparison deliberately uses only host timing.
        rng = random.Random(seed + 41 * s)
        windows = []
        callers = {"checked": default, "bounded": default.run_bounded}
        for _ in range(5):
            order = list(callers)
            rng.shuffle(order)
            samples = {}
            for name in order:
                fn = callers[name]
                samples[name] = host_time(lambda: fn(*held[-1]), samples=11)
            windows.append(dict(order=order, host=samples))
        checked_cost = dict(
            tile=32,
            windows=windows,
            median_us={k: median(w["host"][k]["median_us"] for w in windows) for k in callers},
        )
    assert digests == [[digest(x) for x in xs] for xs in req] and graph == str(ep.graph)
    return dict(
        spec=spec,
        seed=seed,
        export_graph=graph,
        input_sha256=digests,
        cpu_fp64_sha256=[digest(x) for x in refs],
        gates=gates,
        build_failures=fail,
        generated=meta,
        timing=clocks,
        checked_host=checked_cost,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(4)
    r = dict(
        smoke=a.smoke,
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),
        provenance=capture(a.output.parent / "sources"),
        domain="abs(q,k,v)<=64; mask finite abs<=10000 or -inf; fast run_bounded timings exclude numerical scanning",
        cases=[],
    )
    for s in SPECS[2:3] if a.smoke else SPECS:
        for seed in (2199917,) if a.smoke else SEEDS:
            c = case(s, seed, a.smoke)
            r["cases"].append(c)
            print(
                s,
                seed,
                {k: z["admitted"] for k, z in c["gates"].items()},
                c["timing"]["median_us"] if c["timing"] else "",
                flush=True,
            )
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(r, indent=2, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
