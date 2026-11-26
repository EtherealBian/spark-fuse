"""Paired functional K/V scatters followed by reads. Time complete graphs including index creation."""

import argparse, json
from hashlib import sha256
from pathlib import Path
import torch
from examples.cache_slot import CacheSlot, request
from sparkfuse.slot_forward import compile_slot_forward
from length_decode_panel import counted_inductor
from token_quant_panel import digest
from inspect_reduction_sass import op_counts
import subprocess, re
from paired_timing import measure
from serving_clocks import build, clocks
from provenance import capture

CONFIGS = ((4, 2, 31, 64, "fp16"), (4, 2, 4093, 64, "bf16"))
VISIBLE = ((False, False), (True, False), (False, True), (True, True))
SEEDS = (4374611, 4381177)
PATTERNS = ("equal", "different", "mixed", "endpoints")


def binary(pg):
    so = pg.path.with_suffix(".so")
    raw = subprocess.check_output(["cuobjdump", "--dump-resource-usage", str(so)], text=True)
    ops = op_counts(subprocess.check_output(["cuobjdump", "--dump-sass", str(so)], text=True))
    units = {
        k: dict(registers=int(r), stack_bytes=int(st), shared_bytes=int(sh), local_bytes=int(lo))
        for k, r, st, sh, lo in re.findall(
            r"Function (\S+):\s+REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)", raw
        )
    }
    assert len(units) == 3 and set(units) == set(ops)
    return dict(
        source_sha256=sha256(pg.source.encode()).hexdigest(),
        binary_sha256=sha256(so.read_bytes()).hexdigest(),
        entries=units,
        sass=ops,
    )


def oracle(xs, vis):
    # Independent CPU integer storage model; do not call gather/scatter or route through CUDA.
    k, v, kt, vt = [x.cpu().view(torch.int16) for x in xs[:4]]
    w, r = [t.tolist() for t in xs[4:]]
    b, g, s, d = k.shape
    reads = []
    banks = []
    for old, new, keep in zip((k, v), (kt, vt), vis, strict=True):
        result = old.clone()
        y = torch.empty_like(new)
        for j, (i, l) in enumerate(zip(w, r, strict=True)):
            i = max(0, min(i, s - 1))
            l = max(0, min(l, s - 1))
            result[j, :, i].copy_(new[j, :, 0])
            y[j, :, 0].copy_(result[j, :, l])
        reads.append(y)
        if keep:
            banks.append(result)
    return tuple(reads + banks)


def case(cfg, vis, seed, smoke):
    b, g, s, d, dt = cfg
    m = CacheSlot(vis)
    a, au = [request(cfg, seed + 113 * j, p) for j, p in enumerate(("equal", "endpoints"))]
    ep = torch.export.export(m, a, strict=True)
    graph = str(ep.graph)
    f, setup = build(lambda: compile_slot_forward(ep, a, au))
    paths = {"eager": m, "slot_forward": f}
    graphs = []
    torch._dynamo.reset()
    if not smoke:
        paths["inductor_static"], graphs = counted_inductor(ep.module(), False)
    patterns = PATTERNS if smoke else PATTERNS * 2
    xs = [request(cfg, seed + 1747 + j * 71, p) for j, p in enumerate(patterns)]
    fp = lambda: [[digest(x) for x in row] for row in xs]
    inputs = fp()
    gold = [oracle(r, vis) for r in xs]
    refs = [[digest(y) for y in row] for row in gold]
    gates = {}
    for n, fn in paths.items():
        outputs = [fn(*r) for r in xs]
        rows = []
        live = [y.untyped_storage().data_ptr() for out in outputs for y in out]
        for out, ins, ref, hashes in zip(outputs, xs, gold, refs, strict=True):
            forbidden = {x.untyped_storage().data_ptr() for x in ins}
            got = [digest(y) for y in out]
            structure = len(out) == len(ref) and all(
                y.device == ins[0].device
                and y.dtype == ins[0].dtype
                and y.shape == t.shape
                and y.stride() == t.stride()
                and y.storage_offset() == 0
                and y.untyped_storage().nbytes() == y.numel() * 2
                and y._base is None
                and y.data_ptr() not in forbidden
                for y, t in zip(out, ref, strict=True)
            )
            rows.append(
                dict(
                    hashes=got,
                    matches=[x == y for x, y in zip(got, hashes, strict=True)],
                    structure=structure,
                )
            )
        fresh = len(set(live)) == len(live)
        ok = fresh and all(row["structure"] and all(row["matches"]) for row in rows)
        gates[n] = dict(all_live=fresh, rows=rows, admitted=ok)
        if n != "inductor_static":
            assert ok, (cfg, vis, n, rows)
    paths = {k: v for k, v in paths.items() if gates[k]["admitted"]}
    warm = len(graphs)
    hot = {}
    rot = None
    alloc = {}
    if not smoke:
        fixed = {n: (lambda f=f: f(*xs[0])) for n, f in paths.items()}
        hot = measure(fixed, seed + 59)
        rot = clocks(paths, xs, seed + 7817)
        for n, fn in fixed.items():
            y, z = build(fn)
            z["output_bytes"] = sum(o.untyped_storage().nbytes() for o in y)
            alloc[n] = z
            del y
    assert fp() == inputs and str(ep.graph) == graph and warm == len(graphs)
    return dict(
        config=cfg,
        visible=vis,
        seed=seed,
        patterns=patterns,
        setup=setup,
        binary=binary(f.program),
        export_graph_sha256=sha256(graph.encode()).hexdigest(),
        inputs=inputs,
        oracle=refs,
        gates=gates,
        inductor_backend_calls=warm,
        hot=hot,
        rotating=rot,
        allocation=alloc,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    torch.set_num_threads(4)
    r = dict(
        smoke=a.smoke,
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        provenance=capture(a.output.parent / "sources"),
        cases=[],
    )
    with torch._dynamo.config.patch(recompile_limit=32):
        for cfg in CONFIGS[:1] if a.smoke else CONFIGS:
            for vis in VISIBLE[1::2] if a.smoke else VISIBLE:
                for seed in (4392751,) if a.smoke else SEEDS:
                    c = case(cfg, vis, seed, a.smoke)
                    r["cases"].append(c)
                    print(
                        cfg,
                        vis,
                        seed,
                        {n: v["admitted"] for n, v in c["gates"].items()},
                        flush=True,
                    )
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(r, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
