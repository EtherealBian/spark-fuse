"""Independent vs shared frozen packed Q/K/V at three explicitly admitted lengths."""

import argparse
from contextlib import ExitStack
import gc
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import torch
from examples.quant_fanout import QuantFanout
from provenance import capture
from quant_state_panel import gate
from serving_clocks import clocks
from sparkfuse.family import ShapeFamily, _key, capture_family
from sparkfuse.resident import capture_export
from token_quant_panel import digest, storage

ROWS = (17, 33, 129)
SEQ = (17, 33, 129, 33, 17, 129, 17, 129)
K = 1536
NS = (512, 256, 256)
SEEDS = (1556219, 1569401)
OPTS = dict(
    int8_rhs_layout="packed_fanout",
    request_backend="native",
    partition_options=dict(host_backend="native", token_quant=True, group_epilogues=True),
)


def make(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ws = [
        torch.randint(-128, 128, (n, K), dtype=torch.int8, device="cuda", generator=g).t()
        for n in NS
    ]
    cs = [torch.rand(1, n, device="cuda", generator=g) * 0.03 + 0.001 for n in NS]

    def request(m):
        return tuple(
            torch.randn(m, K, dtype=torch.bfloat16, device="cuda", generator=g) for _ in range(2)
        )

    return (
        QuantFanout(ws, cs).eval(),
        [(request(m), request(m)) for m in ROWS],
        [request(m) for m in SEQ],
    )


def provision(kind, model, pairs):
    if kind == "shared":
        return capture_family(model, pairs, **OPTS)
    with ExitStack() as stack:
        runs = {
            _key(ex): stack.enter_context(
                capture_export(torch.export.export(model, ex), ex, au, **OPTS)
            )
            for ex, au in pairs
        }
        # Route through precisely the same shape/dtype lookup, stream and result ownership guards.
        return ShapeFamily(runs, stack.pop_all(), {})


def inventory(f, model):
    private = {}
    rhs = set()
    staged = set()
    byshape = []
    source = {storage(v) for v in model.buffers()}
    for h in f._handles.values():
        b = dict(h._fn.named_buffers())
        p = b["_sparkfuse_packed_rhs"]
        ptr = storage(p)
        rhs.add(ptr)
        assert tuple(p.stride()) == (1, K) and p._base is None and ptr not in source
        assert all(
            torch.equal(w, getattr(model, f"weight_{i}")) for i, w in enumerate(p.split(NS, dim=1))
        )
        private.update({storage(v): v.untyped_storage().nbytes() for v in b.values()})
        assert set(private).isdisjoint(source)
        for t in h._static:
            assert storage(t) not in staged
            staged.add(storage(t))
        t = h.metadata
        (r,) = (v for v in t["regions"] if v.get("kind") == "packed_int32_projection")
        byshape.append(
            dict(
                shape=r["shape"],
                source_weights=r["source_weights"],
                cuda_sha256=r["cuda_sha256"],
                staging=t["input_bytes"],
                state=t["staging_and_state_bytes"] - t["input_bytes"],
            )
        )
    state = sum(private.values())
    stage = sum(e["staging"] for e in byshape)
    return dict(
        private_state_stores=len(private),
        private_packed_rhs=len(rhs),
        private_packed_rhs_bytes=len(rhs) * K * sum(NS),
        state_bytes=state,
        input_staging_bytes=stage,
        input_stores=len(staged),
        static_bytes=stage + state,
        members=byshape,
    )


def settled():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


@torch.inference_mode()
def worker(kind, seed):
    model, pairs, held = make(seed)
    for ex, au in pairs:
        with capture_export(torch.export.export(model, ex), ex, au, **OPTS) as h:
            h(*au)
    del h
    base = settled()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter_ns()
    with provision(kind, model, pairs) as f:
        torch.cuda.synchronize()
        wall = (time.perf_counter_ns() - start) / 1e6
        retained = settled()
        peak = torch.cuda.max_memory_allocated()
        inv = inventory(f, model)
        if kind == "shared":
            assert inv["static_bytes"] == f.metadata["static_bytes"]
        for a in held:
            assert all(torch.equal(x, y) for x, y in zip(f(*a), model(*a), strict=True))
    del f
    after = settled()
    return dict(
        kind=kind,
        weight_sha256={n: digest(t) for n, t in model.named_buffers()},
        baseline_allocated=base,
        retained_delta=retained - base,
        peak_build_delta=peak - base,
        after_close_delta=after - base,
        warm_setup_ms=wall,
        inventory=inv,
    )


@torch.inference_mode()
def case(seed, smoke, temp):
    model, pairs, held = make(seed)
    refs = [model(*a) for a in held]
    weights = {n: digest(t) for n, t in model.named_buffers()}
    inputs = [[digest(v) for v in a] for a in held]
    admission = [[[digest(v) for v in a] for a in pair] for pair in pairs]
    cpu = [getattr(model, f"weight_{i}").cpu().int() for i in range(len(NS))]
    oracle = []
    for row in refs:
        q = row[1].cpu().int()
        acc = [q @ w for w in cpu]
        assert all(torch.equal(a, b.cpu()) for a, b in zip(acc, row[3::2], strict=True))
        oracle.append([digest(v) for v in acc])
    allocations = {}
    layouts = {}
    decisions = {}
    timing = None
    if not smoke:
        for kind in ("independent", "shared"):
            dest = temp / f"{kind}-{seed}.json"
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--worker",
                    kind,
                    "--seed",
                    str(seed),
                    "--output",
                    str(dest),
                ],
                check=True,
            )
            allocations[kind] = json.loads(dest.read_bytes())
            assert weights == allocations[kind]["weight_sha256"]
    with ExitStack() as stack:
        fns = {"eager": model}
        for kind in ("independent", "shared"):
            f = fns[kind] = stack.enter_context(provision(kind, model, pairs))
            layouts[kind] = inventory(f, model)
            if not smoke:
                assert layouts[kind] == allocations[kind]["inventory"]
        assert layouts["independent"]["members"] == layouts["shared"]["members"]
        if not smoke:
            for name, opts in [
                ("inductor", {}),
                ("cast_inductor", {"emulate_precision_casts": True}),
            ]:
                fns[name] = torch.compile(model, fullgraph=True, dynamic=False, options=opts)
        for name, fn in fns.items():
            decisions[name] = gate(fn, held, refs, model.buffers())
            if name in ("eager", "independent", "shared"):
                assert decisions[name]["admitted"]
        if not smoke:
            timing = clocks(
                {k: v for k, v in fns.items() if decisions[k]["admitted"]}, held, seed + 811
            )
    assert weights == {n: digest(t) for n, t in model.named_buffers()} and inputs == [
        [digest(v) for v in a] for a in held
    ]
    return dict(
        seed=seed,
        weights_sha256=weights,
        admission_sha256=admission,
        held_inputs_sha256=inputs,
        held_rows=list(SEQ),
        reference_sha256=[[digest(v) for v in row] for row in refs],
        cpu_int32_sha256=oracle,
        inventory=layouts,
        isolated_allocator=allocations,
        gates=decisions,
        serving=timing,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--worker", choices=["independent", "shared"])
    p.add_argument("--seed", type=int)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.worker:
        out = worker(a.worker, a.seed)
    else:
        out = dict(
            protocol="Matched native resident request paths at three fixed token lengths, identical packed code and router. Independent captures own separate packed weights and scales; shared family verifies them before reuse. Allocator deltas are measured in separately warmed fresh processes; static bytes count only physical private model state and input staging. Eight held-out requests, two qualification requests per length. Seven shuffled whole-request windows; timings exclude inexact contenders. No whole-model or device-kernel speedup implied.",
            smoke=a.smoke,
            device=torch.cuda.get_device_name(),
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            k=K,
            widths=list(NS),
            rows=list(ROWS),
            provenance=capture(a.output.parent / "sources"),
            cases=[],
        )
        with TemporaryDirectory() as tmp:
            for seed in (1575923,) if a.smoke else SEEDS:
                torch.compiler.reset()
                c = case(seed, a.smoke, Path(tmp))
                out["cases"].append(c)
                print(
                    seed,
                    {p: i["static_bytes"] for p, i in c["inventory"].items()},
                    c["serving"]["median_us"] if c["serving"] else "smoke",
                    flush=True,
                )
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
