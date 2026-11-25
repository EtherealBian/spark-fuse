"""Same captured graph and held-out inputs: fresh owned results vs explicit caller-owned ring slots."""

import argparse
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from projection_affine_panel import SPECS, SEEDS, workload
from provenance import capture
from serving_clocks import build, clocks
from sparkfuse.resident import capture_export
from token_quant_panel import compare, digest


def profile(f):
    f()
    torch.cuda.synchronize()
    with TemporaryDirectory() as d:
        p = Path(d) / "trace.json"
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]
        ) as pr:
            res = f()
            torch.cuda.synchronize()
        pr.export_chrome_trace(str(p))
        del res
        es = json.loads(p.read_text())["traceEvents"]

    def counts(cat):
        return dict(
            sorted(
                Counter(e["name"] for e in es if e.get("ph") == "X" and e.get("cat") == cat).items()
            )
        )

    return {
        cat: counts(cat)
        for cat in ("cpu_op", "cuda_runtime", "cuda_driver", "gpu_memcpy", "kernel")
    }


def allocator(fn, batches):
    # Keep all eight outputs alive across the sample. Exclude ring creation from the hot path.
    stream = torch.cuda.current_stream()
    stream.synchronize()

    def counters():
        d = torch.cuda.memory_stats()
        return [d[k] for k in ("allocation.all.allocated", "allocated_bytes.all.current")]

    before = counters()
    vs = [fn(*v) for v in batches]
    stream.synchronize()
    after = counters()
    del vs
    return dict(new_cuda_allocations=after[0] - before[0], retained_cuda_bytes=after[1] - before[1])


def case(spec, seed, smoke, archive):
    label, m, k, ns, debug = spec
    model, req, _, _, _, _ = workload(spec, seed)
    ex, au, *held = req
    ep = torch.export.export(model, ex)
    opts = dict(
        token_quant=True,
        group_epilogues=True,
        projection_epilogue=True,
        affine_epilogues=True,
        host_backend="native",
    )
    refs = [model(*r) for r in held]
    hashes = [[digest(t) for t in row] for row in refs]
    signed = None
    if not smoke:
        record = next(c for c in archive["cases"] if c["seed"] == seed and c["label"] == label)
        assert (
            hashes == record["ref_sha256"]
            and [[digest(t) for t in row] for row in req] == record["input_sha256"]
        )
        signed = archive["provenance"]["source_tree_sha256"]
    with capture_export(
        ep,
        ex,
        au,
        partition_options=opts,
        int8_rhs_layout="packed_fanout",
        request_backend="native",
    ) as run:
        slots, cost = build(lambda: [tuple(torch.empty_like(t) for t in row) for row in refs])
        batches = list(zip(held, slots, strict=True))

        def fresh(a, slot):
            return run(*a)

        def into(a, slot):
            return run.run_into(a, slot)

        fns = {"fresh": fresh, "run_into": into}
        old = [fresh(*v) for v in batches]
        current = [into(*v) for v in batches]
        assert all(x is y for x, y in zip(current, slots, strict=True))
        ps = [t.data_ptr() for a in old + slots for t in a]
        assert len(ps) == len(set(ps))
        gates = {
            n: [compare(x, y) for x, y in zip(xs, refs, strict=True)]
            for n, xs in [("fresh", old), ("run_into", slots)]
        }
        assert all(
            t["different_bytes"] == 0 and t["geometry_match"]
            for ys in gates.values()
            for y in ys
            for t in y
        )
        z = into(held[-1], slots[0])
        assert (
            z is slots[0]
            and [digest(t) for t in z] == hashes[-1]
            and [digest(t) for t in old[0]] == hashes[0]
        )
        a = {n: allocator(f, batches) for n, f in fns.items()}
        assert a["fresh"]["new_cuda_allocations"] == 8 * len(refs[0]) and a["run_into"] == dict(
            new_cuda_allocations=0, retained_cuda_bytes=0
        )
        if not smoke:
            profiles = {n: profile(lambda: f(*batches[0])) for n, f in fns.items()}
            timing = clocks(fns, batches, seed + m + 731)
        else:
            profiles = timing = None
    return dict(
        seed=seed,
        label=label,
        m=m,
        k=k,
        widths=ns,
        outputs=len(refs[0]),
        oracle_source_tree=signed,
        ref_sha256=hashes,
        gates=gates,
        independent_ring_slots=True,
        explicit_reuse_overwrites=True,
        ring_setup=cost,
        ring_logical_bytes=sum(t.nbytes for s in slots for t in s),
        allocator_8=a,
        profile=profiles,
        clocks=timing,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    archive = (
        None
        if a.smoke
        else json.loads(
            (
                Path(__file__).resolve().parents[1] / "reports/projection-affine-gb10.json"
            ).read_bytes()
        )
    )
    out = dict(
        protocol="Identical resident native graph; fresh allocations vs explicit caller-owned ring of eight preallocated output tuples. Slot construction outside clocks, memory and construction measured separately. Same queue depth, synchronization and identical wrapper arguments. Compare complete requests, not device algorithm or default ownership. Two seeds, eight held-out requests per case checked against eager and previously archived CPU-accumulator oracle digests.",
        smoke=a.smoke,
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        provenance=capture(a.output.parent / "sources"),
        cases=[],
    )
    for seed in (1687829,) if a.smoke else SEEDS:
        for spec in SPECS[:2] if a.smoke else SPECS:
            torch.compiler.reset()
            c = case(spec, seed, a.smoke, archive)
            out["cases"].append(c)
            print(
                seed, c["label"], c["allocator_8"], (c["clocks"] or {}).get("median_us"), flush=True
            )
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
