"""Matched caller-owned output transports: ATen, individual/batched driver copies, or one CUDA grid."""

import argparse
from contextlib import ExitStack, nullcontext
import ctypes
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from unittest.mock import patch

import torch
from output_slot_panel import allocator, profile
from projection_affine_panel import SPECS, SEEDS, workload
from provenance import capture
from serving_clocks import build, clocks
from sparkfuse.resident import capture_export
from sparkfuse.runtime import build as build_cuda
from token_quant_panel import compare, digest


def control(outputs, device, *, batch):
    # Experimental controls only. Same generated-launch ABI, validation, stream tracking and version bumps.
    # Batches are valid here because all source and destination ranges are disjoint; use a non-null stream.
    sizes = [t.nbytes for t in outputs]
    n = len(sizes)
    body = (
        """cudaMemcpyAttributes attr{}; attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  size_t first=0;
  return int(cudaMemcpyBatchAsync(ptr, src, sizes, count, &attr, &first, 1, stream));"""
        if batch
        else """for(int i=0;i<count;++i){
    auto e=cudaMemcpyAsync(ptr[i],src[i],sizes[i],cudaMemcpyDeviceToDevice,stream);
    if(e!=cudaSuccess) return int(e);
  }
  return int(cudaSuccess);"""
    )
    src = (
        """#include <cuda_runtime.h>
extern "C" int launch(void** ptr,int count,void* stream_ptr){
  if(count != COUNT || !stream_ptr) return int(cudaErrorInvalidValue);
  const size_t sizes[COUNT]={SIZES}; const void* src[COUNT];
  for(int i=0;i<count;++i) src[i]=ptr[count+i];
  cudaStream_t stream=static_cast<cudaStream_t>(stream_ptr);
  BODY
}
""".replace("COUNT", str(n))
        .replace("SIZES", ",".join(f"{v}ULL" for v in sizes))
        .replace("BODY", body)
    )
    lib, _ = build_cuda(src, device)
    return (
        lib,
        ctypes.cast(lib.launch, ctypes.c_void_p).value,
        dict(
            control="batch" if batch else "raw",
            output_bytes=sizes,
            cuda_sha256=sha256(src.encode()).hexdigest(),
        ),
    )


def case(spec, seed, smoke, side, oracle):
    label, m, *_ = spec
    model, req, *_ = workload(spec, seed)
    ex, au, *held = req
    ep = torch.export.export(model, ex)
    refs = [model(*r) for r in held]
    hashes = [[digest(t) for t in row] for row in refs]
    inputs = [[digest(t) for t in row] for row in req]
    if oracle:
        rec = next(c for c in oracle["cases"] if c["seed"] == seed and c["label"] == label)
        assert hashes == rec["ref_sha256"] and inputs == rec["input_sha256"]
    opts = dict(
        token_quant=True,
        group_epilogues=True,
        projection_epilogue=True,
        affine_epilogues=True,
        host_backend="native",
    )
    names = ["aten", "cuda"] + (["raw", "batch"] if side and not smoke else [])
    Random(seed + m).shuffle(names)
    setup = {}
    meta = {}
    gates = {}
    allocated = {}
    traces = {}
    timing = None
    with ExitStack() as st:
        handles = {}
        fns = {}
        for n in names:

            def prepare(out, dev, name=n):
                return control(out, dev, batch=name == "batch")

            with (
                patch("sparkfuse.slot_copy.prepare", prepare)
                if n in ("raw", "batch")
                else nullcontext()
            ):
                h, setup[n] = build(
                    lambda: capture_export(
                        ep,
                        ex,
                        au,
                        partition_options=opts,
                        int8_rhs_layout="packed_fanout",
                        request_backend="native",
                        slot_copy_backend="aten" if n == "aten" else "cuda",
                    )
                )
            handles[n] = st.enter_context(h)
            fns[n] = h.run_into
            meta[n] = h.metadata
        with torch.inference_mode(False):
            slots, ring = build(lambda: [tuple(torch.empty_like(t) for t in v) for v in refs])
            requests = list(zip(held, slots, strict=True))
        for n, f in fns.items():
            # Poison the common ring separately before each arm: a no-op write must not pass a previous arm's outputs.
            for group in slots:
                for t in group:
                    t.fill_(79)
            versions = [t._version for group in slots for t in group]
            old = handles[n](*held[-1])
            out = [f(*r) for r in requests]
            assert all(x is y for x, y in zip(out, slots, strict=True))
            assert [v + 1 for v in versions] == [t._version for group in slots for t in group]
            assert len({t.data_ptr() for group in [old, *slots] for t in group}) == (
                1 + len(slots)
            ) * len(old)
            gates[n] = [compare(row, ref) for row, ref in zip(out, refs, strict=True)]
            assert all(
                t["different_bytes"] == 0 and t["geometry_match"] for row in gates[n] for t in row
            )
            assert [digest(t) for t in f(held[0], slots[-1])] == hashes[0] and [
                digest(t) for t in old
            ] == hashes[-1]
            allocated[n] = allocator(f, requests)
            assert allocated[n] == dict(new_cuda_allocations=0, retained_cuda_bytes=0)
        if not smoke:
            traces = {n: profile(lambda: f(*requests[0])) for n, f in fns.items()}
            timing = clocks(fns, requests, seed + m + 953 + int(side))
    return dict(
        seed=seed,
        label=label,
        m=m,
        stream="explicit" if side else "default",
        outputs=len(refs[0]),
        capture_order=names,
        setup=setup,
        metadata=meta,
        input_sha256=inputs,
        ref_sha256=hashes,
        gates=gates,
        versions_incremented_once=True,
        retained_fresh_result_checked=True,
        allocator_8=allocated,
        ring_setup=ring,
        profile=traces,
        clocks=timing,
    )


def singleton(n, seed, smoke):
    """One framework ReLU and one result: the copy-engine path is a serious large-copy baseline."""
    g = torch.Generator(device="cuda").manual_seed(seed + n)
    req = [(torch.randn(n, device="cuda", generator=g),) for _ in range(10)]
    ex, au, *held = req
    model = torch.nn.ReLU().eval()
    ep = torch.export.export(model, ex)
    refs = [model(*r) for r in held]
    with ExitStack() as st:
        handles = {}
        setup = {}
        gates = {}
        order = ["aten", "cuda"]
        Random(seed + n).shuffle(order)
        for k in order:
            f, setup[k] = build(
                lambda: capture_export(
                    ep, ex, au, backend="aten", request_backend="native", slot_copy_backend=k
                )
            )
            handles[k] = st.enter_context(f)
        slots = [torch.empty_like(t) for t in refs]
        requests = list(zip(held, slots, strict=True))
        fns = {k: f.run_into for k, f in handles.items()}
        for k, f in fns.items():
            for t in slots:
                t.fill_(-799)
            ys = [f(*r) for r in requests]
            assert all(x is y for x, y in zip(ys, slots, strict=True))
            gates[k] = [compare((x,), (y,))[0] for x, y in zip(ys, refs, strict=True)]
            assert all(t["different_bytes"] == 0 and t["geometry_match"] for t in gates[k])
        t = None if smoke else clocks(fns, requests, seed + n + 101)
    return dict(
        seed=seed,
        elements=n,
        bytes=4 * n,
        stream="default",
        ref_sha256=[digest(x) for x in refs],
        capture_order=order,
        setup=setup,
        gates=gates,
        clocks=t,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    oracle = (
        None
        if a.smoke
        else json.loads(
            (
                Path(__file__).resolve().parents[1] / "reports/projection-affine-gb10.json"
            ).read_bytes()
        )
    )
    out = dict(
        protocol="All arms: matched captured projection, same private packed weight layout, identical eight output slots, preflight and lifetime policy. Compare separate ATen copies to one fixed-size scheduled byte-copy kernel. Explicit stream additionally includes direct CUDA single and batched memcpy controls. CPU signed-accumulator oracle digests inherited from matched archived affine case. Seven shuffled windows include all host, staging, graph, transport, and synchronization costs. Setup precedes the clock; compiler caches may already be warm. No equivalence claimed between streams or across different runs.",
        smoke=a.smoke,
        oracle_source_tree=oracle["provenance"]["source_tree_sha256"] if oracle else None,
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        provenance=capture(a.output.parent / "sources"),
        cases=[],
    )
    for seed in (2083193,) if a.smoke else SEEDS:
        for spec in SPECS[:2] if a.smoke else SPECS:
            for side in [True] if a.smoke else [False, True]:
                with torch.cuda.stream(
                    torch.cuda.Stream() if side else torch.cuda.default_stream()
                ):
                    torch.compiler.reset()
                    c = case(spec, seed, a.smoke, side, oracle)
                    out["cases"].append(c)
                    print(
                        seed,
                        c["label"],
                        c["stream"],
                        {
                            n: round(d["rotating_sync_us"], 2)
                            for n, d in (c["clocks"] or {}).get("median_us", {}).items()
                        },
                        flush=True,
                    )
    with torch.cuda.stream(torch.cuda.default_stream()):
        out["single_output_controls"] = []
        for seed in (2083193,) if a.smoke else (1869841, 1869943):
            for n in (67,) if a.smoke else (67, 65537, 4194305):
                torch.compiler.reset()
                c = singleton(n, seed, a.smoke)
                out["single_output_controls"].append(c)
                print(
                    "single",
                    seed,
                    n,
                    {
                        k: round(v["rotating_sync_us"], 2)
                        for k, v in (c["clocks"] or {}).get("median_us", {}).items()
                    },
                    flush=True,
                )
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
