"""Fresh output ownership: independent ATen clones versus independent allocations and one CUDA transport."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from random import Random

import torch
from output_slot_panel import allocator, profile
from projection_affine_panel import SEEDS, SPECS, workload
from provenance import capture
from serving_clocks import build, clocks
from sparkfuse.contextual import _exact
from sparkfuse.resident import capture_export
from token_quant_panel import digest


def leaves(y):
    return y if type(y) is tuple else (y,)


def assess(model, req, opts, seed, smoke, oracle=None):
    ep = torch.export.export(model, req[0])
    held = req[2:]
    refs = [model(*r) for r in held]
    hashes = [[digest(t) for t in leaves(y)] for y in refs]
    ins = [[digest(t) for t in row] for row in req]
    weights = list(model.state_dict().values())
    state = [digest(t) for t in weights]
    count = len(leaves(refs[0]))
    order = ["aten", "cuda"]
    Random(seed).shuffle(order)
    meta = {}
    setup = {}
    alloc = {}
    traces = {}
    gates = {}
    if oracle:
        assert hashes == oracle["ref_sha256"] and ins == oracle["input_sha256"]
    with ExitStack() as st:
        arms = {}
        for name in order:
            h, cost = build(
                lambda: capture_export(
                    ep, *req[:2], **opts, request_backend="native", fresh_copy_backend=name
                )
            )
            arms[name] = st.enter_context(h)
            meta[name] = h.metadata
            setup[name] = cost
        # Additional fresh-output control: allocate each destination in Python and pass it to run_into.
        # Deliberately keep this separate from the no-allocation preallocated slot API.
        ring = st.enter_context(
            capture_export(ep, *req[:2], **opts, request_backend="native", slot_copy_backend="cuda")
        )

        def allocate(*r):
            dst = tuple(torch.empty_like(t) for t in leaves(refs[0]))
            return ring.run_into(r, dst if type(refs[0]) is tuple else dst[0])

        arms["explicit_allocate"] = allocate
        retained = []
        for n, fn in arms.items():
            ys = [fn(*row) for row in held]
            flat = [t for y in ys for t in leaves(y)]
            ptr = {t.untyped_storage().data_ptr() for t in flat}
            forbidden = {t.untyped_storage().data_ptr() for row in req for t in row} | {
                w.untyped_storage().data_ptr() for w in weights
            }
            assert len(ptr) == len(flat) and ptr.isdisjoint(forbidden)
            gates[n] = [
                _exact(y, ref, "unchanged independent result")
                for y, ref in zip(ys, refs, strict=True)
            ]
            assert all(
                t.storage_offset() == 0 and t.untyped_storage().nbytes() == t.nbytes for t in flat
            )
            retained.append(ys[0])
            del flat, ys
            alloc[n] = allocator(fn, held)
            assert alloc[n]["new_cuda_allocations"] == 8 * count
            traces[n] = None if smoke else profile(lambda: fn(*held[0]))
        assert (
            len({t.untyped_storage().data_ptr() for y in retained for t in leaves(y)})
            == len(arms) * count
        )
        timing = None if smoke else clocks(arms, held, seed)
    for y in retained:
        _exact(y, refs[0], "after serving and closing all handles")
    assert state == [digest(t) for t in weights] and ins == [
        [digest(t) for t in row] for row in req
    ]
    return dict(
        outputs=count,
        state_sha256=state,
        ref_sha256=hashes,
        input_sha256=ins,
        capture_order=order,
        setup=setup,
        metadata=meta,
        gates=gates,
        retained_after_close=True,
        allocations_8=alloc,
        profile=traces,
        clocks=timing,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
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
        protocol="All three arms return fresh independent storages on every call, with the same quantized projection or single ReLU. Compare ATen clone per output, native separate allocation + one CUDA grid, and explicitly allocating in Python before the existing CUDA run_into. Previous CPU int32 oracle input/output hashes verified. Retain 8 requests at once and compare every public output byte, stride, ownership and post-close contents. Seven shuffled windows for fixed/rotating and queue8; allocator counts exclude graph setup.",
        smoke=a.smoke,
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        provenance=capture(a.output.parent / "sources"),
        oracle_source_tree=oracle["provenance"]["source_tree_sha256"] if oracle else None,
        cases=[],
        controls=[],
    )
    opts = dict(
        partition_options=dict(
            token_quant=True,
            group_epilogues=True,
            projection_epilogue=True,
            affine_epilogues=True,
            host_backend="native",
        ),
        int8_rhs_layout="packed_fanout",
    )

    def show(r):
        print(
            r["seed"],
            r["label"],
            r["stream"],
            {
                k: [round(v["rotating_sync_us"], 2), round(v["queued8_us_per_forward"], 2)]
                for k, v in (r["clocks"] or {}).get("median_us", {}).items()
            },
            flush=True,
        )

    for seed in (2099263,) if a.smoke else SEEDS:
        for spec in SPECS[:2] if a.smoke else SPECS:
            label, m, *_ = spec
            for side in [True] if a.smoke else [False, True]:
                with torch.cuda.stream(
                    torch.cuda.Stream() if side else torch.cuda.default_stream()
                ):
                    torch.compiler.reset()
                    model, req, *_ = workload(spec, seed)
                    old = (
                        next(
                            c for c in oracle["cases"] if c["label"] == label and c["seed"] == seed
                        )
                        if oracle
                        else None
                    )
                    r = assess(model, req, opts, seed + m + 781 + int(side), a.smoke, old)
                    r.update(seed=seed, m=m, label=label, stream="explicit" if side else "default")
                    out["cases"].append(r)
                    show(r)
        for n in (257,) if a.smoke else (257, 65537, 4194305):
            torch.compiler.reset()
            g = torch.Generator(device="cuda").manual_seed(seed + n)
            req = [(torch.randn(n, device="cuda", generator=g),) for _ in range(10)]
            r = assess(torch.nn.ReLU().eval(), req, dict(backend="aten"), seed + n + 41, a.smoke)
            r.update(seed=seed, elements=n, label="single_relu", stream="default")
            out["controls"].append(r)
            show(r)
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
