"""Does a schedule imported from the public evidence work on new input allocations?"""

import argparse, json, sqlite3, tempfile
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from hashlib import sha256
import torch
from torch.utils import _pytree as tree
from examples.graphs import case as make
from sparkfuse.runtime import compile_export
from sparkfuse.ir import lower
from sparkfuse.tuning import backend_id, signature, compile_tuned
from sparkfuse.measure import compare
from import_campaign import import_bundle
from serving_clocks import build, clocks
from paired_timing import measure
from length_decode_panel import counted_inductor
from token_quant_panel import digest
from provenance import capture

# All ten eligible fixture keys, the measured default and a shape never in the fixture.
CASES = (
    ("resnorm", 1, 768, "float32"),
    ("resnorm", 32, 768, "bfloat16"),
    ("resnorm2", 1, 768, "float16"),
    ("resnorm2", 32, 768, "bfloat16"),
    ("mean", 1, 1536, "float16"),
    ("mean", 32, 768, "float16"),
    ("gate", 1, 768, "float16"),
    ("gate", 32, 1536, "float16"),
    ("decomp", 1, 768, "float32"),
    ("decomp", 32, 768, "float32"),
    ("gate", 48, 1024, "float32"),
    ("resnorm2", 7, 1009, "bfloat16"),
)


def one(cfg, db, seed, smoke):
    name, r, w, dt = cfg
    dtype = getattr(torch, dt)
    m, a = make(name, r, w, dtype, seed)
    ep = torch.export.export(m, a, strict=True)
    graph = str(ep.graph)
    key = signature(lower(ep), a[0].device)
    with closing(sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        row = conn.execute("select record from schedules where key=?", (key,)).fetchone()
    entry = json.loads(row[0]) if row else None
    default, ds = build(lambda: compile_export(ep))
    f, fs = build(lambda: compile_tuned(ep, db, a[0].device))
    selected = asdict(f.schedule)
    assert selected == (entry["selected"] if entry else asdict(default.schedule))
    assert bool(entry) == (cfg in CASES[:10])
    paths = {"eager": m, "default": default, "catalog": f}
    graphs = []
    torch._dynamo.reset()
    if not smoke:
        paths["inductor_static"], graphs = counted_inductor(m, False)
    xs = [make(name, r, w, dtype, seed + 4099 + 83 * j)[1] for j in range(2 if smoke else 8)]
    fp = lambda: [[digest(t) for t in x] for x in xs]
    inputs = fp()
    ref = [m(*x) for x in xs]
    exact = (1,) if name == "resnorm2" else ()
    gates = {}
    for n, fn in paths.items():
        checks = []
        for x, g in zip(xs, ref, strict=True):
            y = fn(*x)
            fy, yt = tree.tree_flatten(y)
            ft, gt = tree.tree_flatten(g)
            shape = yt == gt and all(
                v.shape == t.shape
                and v.stride() == t.stride()
                and v.dtype == t.dtype
                and v.device == t.device
                for v, t in zip(fy, ft, strict=True)
            )
            bits = all(
                torch.equal(
                    fy[j].view(torch.int16 if fy[j].element_size() == 2 else torch.int32),
                    ft[j].view(torch.int16 if ft[j].element_size() == 2 else torch.int32),
                )
                for j in exact
            )
            try:
                errors = compare(y, g, exact)
                valid = True
            except AssertionError:
                errors = None
                valid = False
            checks.append(
                dict(
                    shape=shape,
                    exact_bits=bits,
                    within_tolerance=valid,
                    errors=errors,
                    output_sha256=[digest(t) for t in fy],
                    reference_sha256=[digest(t) for t in ft],
                )
            )
        gates[n] = dict(
            checks=checks,
            admitted=all(q["shape"] and q["exact_bits"] and q["within_tolerance"] for q in checks),
        )
    assert gates["eager"]["admitted"]
    eligible = {n: v for n, v in paths.items() if gates[n]["admitted"]}
    warm = len(graphs)
    hot = {}
    rotation = None
    if not smoke:
        hot = measure({n: (lambda fn=fn: fn(*xs[0])) for n, fn in eligible.items()}, seed + 53)
        rotation = clocks(eligible, xs, seed + 8911)
    assert str(ep.graph) == graph and fp() == inputs and len(graphs) == warm
    return dict(
        config=cfg,
        seed=seed,
        signature=key,
        entry=entry,
        selected=selected,
        default=asdict(default.schedule),
        setup=dict(catalog=fs, default=ds),
        program_sha256={
            n: sha256(v.path.with_suffix(".so").read_bytes()).hexdigest()
            for n, v in [("catalog", f), ("default", default)]
        },
        inputs=inputs,
        gates=gates,
        inductor_backend_calls=warm,
        hot=hot,
        rotating=rotation,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--bundle",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reports/campaign-dispatch-sample.json.gz",
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    torch.set_num_threads(4)
    r = dict(
        smoke=a.smoke,
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),
        provenance=capture(a.output.parent / "sources"),
        cases=[],
    )
    with tempfile.TemporaryDirectory() as t, torch._dynamo.config.patch(recompile_limit=64):
        db = Path(t) / "catalog.sqlite"
        r["catalog"] = import_bundle(a.bundle, db, backend_id())
        before = sha256(db.read_bytes()).hexdigest()
        for j, c in enumerate((CASES[3], CASES[-1]) if a.smoke else CASES):
            q = one(c, db, (4531013 if a.smoke else 4500251) + 101 * j, a.smoke)
            r["cases"].append(q)
            print(c, {n: g["admitted"] for n, g in q["gates"].items()}, flush=True)
        assert sha256(db.read_bytes()).hexdigest() == before
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(r, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
