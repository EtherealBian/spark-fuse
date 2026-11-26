"""Same symbolic export, same measured CUDA, switch exact leading sizes inside one native call."""

import argparse, json, tempfile
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import torch
from torch.utils import _pytree as tree
from examples.graphs import case as make
from sparkfuse.catalog_rows import compile_catalog_rows
from sparkfuse.dynamic import DynamicRows
from sparkfuse.measure import compare
from sparkfuse.native import native_dispatch
from sparkfuse.tuning import backend_id, compile_tuned, signature
from sparkfuse.ir import lower
from import_campaign import import_bundle
from serving_clocks import build, clocks
from paired_timing import measure
from length_decode_panel import counted_inductor
from token_quant_panel import digest
from provenance import capture

CONFIGS = (
    ("decomp", 768, "float32"),
    ("resnorm2", 768, "bfloat16"),
    ("gate", 768, "float16"),
    ("mean", 768, "float16"),
    ("gate", 1024, "float32"),
)
ROWS = (1, 32, 129, 7, 48, 2, 97, 65)
HINTS = (1, 7, 32, 48)


def exported(m, ex):
    sym = torch.export.Dim("rows", min=1, max=129)
    return torch.export.export(
        m, ex, dynamic_shapes=tuple({0: sym} if x.ndim == 2 else None for x in ex), strict=True
    )


def check(y, ref, name):
    v, s = tree.tree_flatten(y)
    t, p = tree.tree_flatten(ref)
    shape = (
        s == p
        and len(v) == len(t)
        and all(
            (a.shape, a.stride(), a.device, a.dtype) == (b.shape, b.stride(), b.device, b.dtype)
            for a, b in zip(v, t)
        )
    )
    ds = [digest(a) for a in v]
    gs = [digest(a) for a in t]
    exact = name != "resnorm2" or ds[1] == gs[1]
    try:
        errors = compare(y, ref, (1,) if name == "resnorm2" else ())
        valid = True
    except AssertionError:
        errors = None
        valid = False
    return dict(
        structure=shape,
        outputs=ds,
        reference=gs,
        exact_bits=exact,
        within_tolerance=valid,
        errors=errors,
    )


def case(cfg, db, seed, smoke):
    name, w, dt = cfg
    dtype = getattr(torch, dt)
    m, ex = make(name, 13, w, dtype, seed)
    ep = exported(m, ex)
    graph = str(ep.graph)
    generic, gb = build(lambda: DynamicRows(ep).native_dispatch())
    mv, mb = build(lambda: compile_catalog_rows(ep, db, HINTS))
    members = {}
    routes = {v.rows: v for v in mv.variants if v.selected is not None}
    want = {
        ("decomp", 768): {1, 32},
        ("resnorm2", 768): {32},
        ("gate", 768): {1},
        ("mean", 768): {32},
        ("gate", 1024): set(),
    }
    assert set(routes) == want[(name, w)]
    for n, v in routes.items():
        _, a = make(name, n, w, dtype, seed + 11 * n)
        static = torch.export.export(m, a, strict=True)
        other = compile_tuned(static, db)
        assert (
            signature(lower(static), a[0].device) == v.signature
            and other.schedule == v.selected
            and sha256(other.source.encode()).hexdigest() == v.source_sha256
        )
        members[n] = native_dispatch(other)

    def split(*a):
        return members.get(a[0].shape[0], generic)(*a)

    paths = {"eager": m, "generic_native": generic, "split_python": split, "table_native": mv.run}
    torch._dynamo.reset()
    graphs = []
    if not smoke:
        paths["inductor_dynamic"], graphs = counted_inductor(m, True)
    rows = ROWS if not smoke else (1, 32, 97)
    xs = [make(name, n, w, dtype, seed + 13007 + 73 * j)[1] for j, n in enumerate(rows)]
    hashes = lambda: [[digest(t) for t in a] for a in xs]
    before = hashes()
    ref = [m(*a) for a in xs]
    gates = {}
    for n, fn in paths.items():
        ys = [fn(*a) for a in xs]
        tests = [check(y, g, name) for y, g in zip(ys, ref, strict=True)]
        flat = [t for y in ys for t in tree.tree_leaves(y)]
        live = len({t.data_ptr() for t in flat}) == len(flat) and not {
            t.data_ptr() for t in flat
        }.intersection(t.data_ptr() for a in xs for t in a)
        gates[n] = dict(
            rows=tests,
            ownership=live,
            admitted=live
            and all(q["structure"] and q["exact_bits"] and q["within_tolerance"] for q in tests),
        )
    for j, n in enumerate(rows):
        assert (
            gates["table_native"]["rows"][j]["outputs"]
            == gates["split_python"]["rows"][j]["outputs"]
        )
        if n not in routes:
            assert (
                gates["table_native"]["rows"][j]["outputs"]
                == gates["generic_native"]["rows"][j]["outputs"]
            )
    elig = {n: v for n, v in paths.items() if gates[n]["admitted"]}
    warm = len(graphs)
    hot = {}
    rot = None
    if not smoke:
        for n in (min(routes, default=48), 97):
            req = xs[rows.index(n)]
            hot[str(n)] = measure(
                {k: (lambda f=f: f(*req)) for k, f in elig.items()}, seed + 31 * n
            )
        rot = clocks(elig, xs, seed + 941)
    assert warm == len(graphs) and before == hashes() and graph == str(ep.graph)
    return dict(
        config=cfg,
        seed=seed,
        rows=rows,
        hints=HINTS,
        bounds=mv.bounds,
        variants=[asdict(v) for v in mv.variants],
        generic=asdict(mv.generic),
        generic_source_sha256=mv.fallback_source_sha256,
        setup=dict(generic=gb, table=mb),
        inputs=before,
        inductor_backend_calls=warm,
        gates=gates,
        hot=hot,
        rotating=rot,
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
        for cfg in CONFIGS[:1] if a.smoke else CONFIGS:
            for seed in (4712111,) if a.smoke else (4670017, 4687319):
                c = case(cfg, db, seed, a.smoke)
                r["cases"].append(c)
                print(cfg, seed, {n: v["admitted"] for n, v in c["gates"].items()}, flush=True)
        assert sha256(db.read_bytes()).hexdigest() == before
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(r, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
