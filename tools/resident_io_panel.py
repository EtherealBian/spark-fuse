"""Single host-call resident request I/O: unchanged graph and visible allocations."""

import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path

import torch
from torch import nn
from examples.quant_fanout import QuantFanout
from kernel_trace import kernel_census
from provenance import capture
from serving_clocks import build, clocks
from sparkfuse.resident import capture_export
from token_quant_panel import compare, digest, storage

SEEDS = (1481849, 1495241)
SPECS = (
    ("qkv-short", 17, 1536, (128, 64, 64)),
    ("qkv-prefill", 129, 3072, (1536, 256, 256)),
    ("one-output", 17, 256, ()),
)


class One(nn.Module):
    def forward(self, x, r):
        return (x + r) * r


def leaves(out):
    return out if isinstance(out, tuple) else (out,)


def case(spec, seed, smoke):
    label, m, k, ns = spec
    g = torch.Generator(device="cuda").manual_seed(seed + m + k)
    ws = [
        torch.randint(-128, 128, (n, k), dtype=torch.int8, device="cuda", generator=g).t()
        for n in ns
    ]
    cs = [torch.rand(1, n, device="cuda", generator=g) * 0.03 + 0.001 for n in ns]
    model = QuantFanout(ws, cs).eval() if ns else One().eval()
    req = [
        tuple(torch.randn(m, k, device="cuda", generator=g, dtype=torch.bfloat16) for _ in range(2))
        for _ in range(10)
    ]
    ex, audit, *held = req
    w = {n: digest(v) for n, v in model.named_buffers()}
    inputs = [[digest(v) for v in a] for a in req]
    ep = torch.export.export(model, ex)
    refs = [model(*a) for a in held]
    oracle = []
    if ns:
        wc = [t.cpu().int() for t in ws]
        for y in refs:
            q = y[1].reshape(-1, k).cpu().int()
            v = [q @ c for c in wc]
            assert all(torch.equal(x.cpu(), z) for x, z in zip(y[3::2], v, strict=True))
            oracle.append([digest(t) for t in v])
    paths = {"eager": model}
    setup = {}
    meta = {}
    gates = {}
    profiles = {}
    opts = (
        dict(
            int8_rhs_layout="packed_fanout",
            partition_options=dict(token_quant=True, group_epilogues=True, host_backend="native"),
        )
        if ns
        else dict(backend="aten")
    )
    forbidden = {storage(v) for a in req for v in a} | {storage(v) for v in model.buffers()}
    with ExitStack() as stack:
        for mode in ("python", "native"):
            exe, setup[mode] = build(
                lambda: capture_export(ep, ex, audit, request_backend=mode, **opts)
            )
            stack.enter_context(exe)
            paths[mode] = exe
            meta[mode] = exe.metadata
            if not smoke:
                profiles[mode] = kernel_census(lambda: exe._fn(*held[-1]))
        assert {k: v for k, v in meta["python"].items() if k != "request_backend"} == {
            k: v for k, v in meta["native"].items() if k != "request_backend"
        }
        if not smoke:
            assert profiles["python"] == profiles["native"]
        for name, fn in paths.items():
            ys = [fn(*a) for a in held]
            vs = [compare(leaves(y), leaves(r)) for y, r in zip(ys, refs, strict=True)]
            ptrs = [storage(v) for y in ys for v in leaves(y)]
            fresh = len(ptrs) == len(set(ptrs)) and forbidden.isdisjoint(ptrs)
            containers = all(type(y) is type(r) for y, r in zip(ys, refs, strict=True))
            assert (
                fresh
                and containers
                and all(
                    v["geometry_match"] and v["different_bytes"] == 0 for row in vs for v in row
                )
            )
            gates[name] = dict(fresh=fresh, container_match=containers, outputs=vs)
        timing = clocks(paths, held, seed + m + k + 463) if not smoke else None
    assert inputs == [[digest(v) for v in a] for a in req] and w == {
        n: digest(v) for n, v in model.named_buffers()
    }
    return dict(
        label=label,
        m=m,
        k=k,
        widths=list(ns),
        seed=seed,
        inputs_sha256=inputs,
        weights_sha256=w,
        export_sha256=sha256(str(ep.graph).encode()).hexdigest(),
        reference_sha256=[[digest(t) for t in leaves(y)] for y in refs],
        cpu_int32_sha256=oracle,
        setup=setup,
        metadata=meta,
        ordinary_graph_profiles=profiles,
        gates=gates,
        serving=timing,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    result = dict(
        protocol="Matched host I/O policies on identical CUDA graphs with the same request contract: input staging and stream lifetime, graph replay, fresh independent outputs. Eight held-out requests after two qualification samples, including CPU INT32 oracles for packed QKV. Profiles are ordinary forwards of the frozen graph outside serving clocks. Seven shuffled windows: synchronized fixed/rotating, submit-only and eight outstanding requests. The one-output ATen control uses a different workload and is not a QKV speed comparison. Warm-cache setup and allocated memory reported separately.",
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        smoke=a.smoke,
        provenance=capture(a.output.parent / "sources"),
        cases=[],
    )
    for seed in (1506559,) if a.smoke else SEEDS:
        for spec in SPECS[::2] if a.smoke else SPECS:
            torch.compiler.reset()
            c = case(spec, seed, a.smoke)
            result["cases"].append(c)
            print(
                seed, spec[0], c["serving"]["median_us"] if c["serving"] else "checked", flush=True
            )
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
