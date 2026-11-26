import ctypes as C
from copy import copy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from examples.graphs import case
from sparkfuse.catalog_rows import compile_catalog_rows
from sparkfuse.dynamic import DynamicRows
from sparkfuse.measure import compare
from sparkfuse.native import _native_dispatch
from sparkfuse.tuning import backend_id
from tools.import_campaign import import_bundle


def export(name="resnorm2", width=768, dt=torch.bfloat16):
    m, a = case(name, 13, width, dt, 1979)
    dim = torch.export.Dim("rows", min=1, max=129)
    return m, torch.export.export(
        m, a, dynamic_shapes=tuple({0: dim} if x.ndim == 2 else None for x in a), strict=True
    )


@pytest.fixture
def db(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = tmp_path / "catalog.sqlite"
    import_bundle(root / "reports/campaign-dispatch-sample.json.gz", path, backend_id())
    return path


@torch.inference_mode()
def test_replay_hit_fallback_stream_ownership_and_no_request_io(db, monkeypatch):
    m, ep = export()
    f = compile_catalog_rows(ep, db, [7, 32])
    orig = str(ep.graph)
    assert [(v.rows, v.selected is not None) for v in f.variants] == [(7, False), (32, True)]
    db.unlink()

    def forbidden(*a, **kw):
        raise AssertionError("request attempted setup or specialization")

    for path in (
        "sparkfuse.catalog_rows.build",
        "sparkfuse.catalog_rows.signature",
        "sparkfuse.catalog_rows.sqlite3.connect",
        "torch.export.export",
    ):
        monkeypatch.setattr(path, forbidden)
    side = torch.cuda.Stream()
    main = torch.cuda.current_stream()
    saved = []
    side.wait_stream(main)
    with torch.cuda.stream(side):
        for j, n in enumerate((32, 97, 7, 1, 32, 129)):
            _, a = case("resnorm2", n, 768, torch.bfloat16, 21377 + j)
            a = tuple(torch.cat((t.new_zeros(1), t.flatten()))[1:].view_as(t) for t in a)
            ys = f(*a)
            gs = m(*a)
            saved.append((ys, gs))
        g = torch.cuda.CUDAGraph()
        _, x = case("resnorm2", 32, 768, torch.bfloat16, 17711)
        f(*x)
        with torch.cuda.graph(g, stream=side):
            buf = f(*x)
        x[0].neg_()
        g.replay()
        last = m(*x)
    main.wait_stream(side)
    for y, ref in saved + [(buf, last)]:
        compare(y, ref, (1,))
        assert torch.equal(y[1].view(torch.int16), ref[1].view(torch.int16))
    assert str(ep.graph) == orig and len({t.data_ptr() for ys, _ in saved for t in ys}) == 2 * len(
        saved
    )


@torch.inference_mode()
def test_bounded_table_routes_multiple_entries_at_native_abi():
    # Tiny host-only callbacks record the native choice and deliberately return a launch error;
    # no test kernel runs and no uninitialized output escapes. Other tests use actual CUDA.
    _, ep = export("decomp", 768, torch.float32)
    dyn = DynamicRows(ep)
    k = copy(dyn.kernel)
    seen = []
    cb = C.CFUNCTYPE(C.c_int, C.POINTER(C.c_void_p), C.c_int, C.c_void_p)

    def lib(tag):
        @cb
        def launch(ptr, n, stream):
            seen.append((tag, n))
            return 17

        return SimpleNamespace(launch=launch)

    k.lib = lib("generic")
    f = _native_dispatch(k, dyn.bounds, specials=[(1, lib("one")), (32, lib("thirty-two"))])
    for n in (129, 32, 1, 7, 32):
        _, a = case("decomp", n, 768, torch.float32, n)
        with pytest.raises(RuntimeError, match="CUDA launch returned 17"):
            f(*a)
    assert seen == [
        ("generic", 129),
        ("thirty-two", 32),
        ("one", 1),
        ("generic", 7),
        ("thirty-two", 32),
    ]


@torch.inference_mode()
def test_constructor_rejects_bad_catalog_and_bounds(db):
    import sqlite3, json

    _, ep = export()
    before = sha256(db.read_bytes()).digest()
    for rows in ([32, 32], [False], [-1], [130], list(range(33))):
        with pytest.raises(ValueError, match="row"):
            compile_catalog_rows(ep, db, rows)
    assert sha256(db.read_bytes()).digest() == before
    from contextlib import closing

    with closing(sqlite3.connect(db)) as conn, conn:
        q = json.loads(conn.execute("select info from provenance").fetchone()[0])
        q["compiler"] = "0" * 64
        conn.execute("update provenance set info=?", (json.dumps(q),))
    with pytest.raises(ValueError, match="incompatible"):
        compile_catalog_rows(ep, db, [32])


@torch.inference_mode()
def test_every_eligible_route_still_checks_all_inputs(db):
    _, ep = export()
    f = compile_catalog_rows(ep, db, [32])
    for n in (7, 32):
        _, a = case("resnorm2", n, 768, torch.bfloat16, n)
        x, skip, w = a
        for bad in (
            (x, skip[: n - 1], w),
            (x, skip.float(), w),
            (x, skip.cpu(), w),
            (x, skip, w[:17]),
            (x, skip),
        ):
            with pytest.raises((ValueError, RuntimeError)):
                f(*bad)
    for n in (0, 130):
        _, a = case("resnorm2", n, 768, torch.bfloat16, n)
        with pytest.raises(ValueError, match="bound"):
            f(*a)


@torch.inference_mode()
def test_lifted_singleton_weight_and_nested_request_tree(db):
    class Broadcast(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "weight",
                torch.linspace(-1, 1, 768, device="cuda", dtype=torch.float32).reshape(1, 768),
            )

        def forward(self, request):
            r = request[0] + request[1]
            return {"result": r * self.weight, "visible": r}

    m = Broadcast().eval()
    _, inputs = case("gate", 13, 768, torch.float32, 713)
    dim = torch.export.Dim("rows", min=1, max=129)
    ep = torch.export.export(
        m, (inputs,), dynamic_shapes={"request": ({0: dim}, {0: dim})}, strict=True
    )
    f = compile_catalog_rows(ep, db, [1, 32])
    assert not any(v.selected for v in f.variants)
    for j, n in enumerate((32, 1, 97, 32)):
        _, a = case("gate", n, 768, torch.float32, 6001 + j)
        m.weight.add_(0.25)
        out = f(a)
        ref = m(a)
        assert tuple(out) == tuple(ref) and all(
            torch.equal(out[k].view(torch.int32), ref[k].view(torch.int32)) for k in ref
        )
    with pytest.raises(ValueError, match="pytree"):
        f(*a)
