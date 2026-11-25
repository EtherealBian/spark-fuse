"""A list of exact request shapes may own one, and only one, frozen packed projection."""

from itertools import chain

import pytest
import torch
from examples.quant_fanout import QuantFanout
from sparkfuse.family import capture_family
from sparkfuse.ir import Unsupported
from sparkfuse.resident import ResidentExecutable

OPTS = dict(
    int8_rhs_layout="packed_fanout",
    partition_options=dict(host_backend="native", token_quant=True, group_epilogues=True),
)


def fixture(ns=(64, 32, 16), rows=(17, 33, 65)):
    g = torch.Generator(device="cuda").manual_seed(1535743)
    ws = [
        torch.randint(-128, 128, (n, 128), device="cuda", dtype=torch.int8, generator=g).t()
        for n in ns
    ]
    ws[1] = ws[1].contiguous()
    model = QuantFanout(ws, [torch.rand((1, n), generator=g, device="cuda") for n in ns]).eval()
    req = {
        m: [
            tuple(
                torch.randn(m, 128, dtype=torch.bfloat16, generator=g, device="cuda")
                for _ in range(2)
            )
            for _ in range(4)
        ]
        for m in rows
    }
    return model, req, [r[:2] for r in req.values()]


def exact(a, b):
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and a.stride() == b.stride()
        and a.storage_offset() == b.storage_offset()
        and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    )


@pytest.mark.parametrize(
    "ns,io", [((16, 32), "python"), ((64, 32, 16), "native"), ((8, 16, 24, 32), "native")]
)
@torch.inference_mode()
def test_single_final_layout_allocation_and_held_outputs(monkeypatch, ns, io):
    model, req, pairs = fixture(ns)
    ws = [getattr(model, f"weight_{i}") for i in range(len(ns))]
    inputs = [req[m][j] for j in (3, 2) for m in (65, 17, 33)]
    refs = [model(*x) for x in inputs]
    for ref in refs:
        codes = ref[1].cpu().int()
        assert all(
            torch.equal(a.cpu(), codes @ w.cpu().int()) for a, w in zip(ref[3::2], ws, strict=True)
        )
    state = sum(t.nbytes for t in model.buffers())
    stage = sum(t.nbytes for pair in pairs for t in pair[0])
    allocs = []
    original = torch.empty_strided

    def watch(size, stride, *args, **kw):
        v = original(size, stride, *args, **kw)
        if (
            type(v) is torch.Tensor
            and v.device.type == "cuda"
            and tuple(size) == (128, sum(ns))
            and v.dtype == torch.int8
            and tuple(stride) == (1, 128)
        ):
            allocs.append(v.data_ptr())
        return v

    monkeypatch.setattr(torch, "empty_strided", watch)
    with capture_family(
        model, pairs, request_backend=io, max_static_bytes=stage + state, **OPTS
    ) as f:
        assert (
            f.metadata["static_bytes"] == state + stage
            and f.metadata["weight_snapshot_bytes"] == state
        )
        handles = list(f._handles.values())
        packed = [h._fn._sparkfuse_packed_rhs for h in handles]
        assert allocs == [packed[0].data_ptr()] and all(p is packed[0] for p in packed)
        assert all(torch.equal(v, w) for v, w in zip(packed[0].split(ns, dim=1), ws, strict=True))
        for i, w in enumerate(ws):
            w.zero_()
            setattr(model, f"weight_{i}", torch.ones_like(w))
            getattr(model, f"channel_{i}").add_(100)
            assert len({getattr(h._fn, f"channel_{i}").data_ptr() for h in handles}) == 1
        out = [f(*x) for x in inputs]
        all_ptr = [t.data_ptr() for t in chain.from_iterable(out)]
        assert len(set(all_ptr)) == len(all_ptr)
        with pytest.raises(Unsupported, match="not admitted"):
            f(*(torch.zeros(18, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)))
    assert all(
        exact(x, y) for o, r in zip(out, refs, strict=True) for x, y in zip(o, r, strict=True)
    )


@torch.inference_mode()
def test_aggregate_cap_before_packing(monkeypatch):
    model, _, pairs = fixture()
    state = sum(t.nbytes for t in model.buffers())
    stage = sum(t.nbytes for p in pairs for t in p[0])
    allocs = []
    orig = torch.empty_strided

    def watch(*a, **kw):
        v = orig(*a, **kw)
        if (
            type(v) is torch.Tensor
            and v.is_cuda
            and v.dtype == torch.int8
            and v.shape == (128, 112)
        ):
            allocs.append(v.data_ptr())
        return v

    monkeypatch.setattr(torch, "empty_strided", watch)
    with pytest.raises(ValueError, match="byte limit"):
        capture_family(model, pairs, max_static_bytes=state + stage - 1, **OPTS)
    assert allocs == []


@pytest.mark.parametrize(
    "change,pattern",
    [
        ("bytes", "packed RHS contents"),
        ("stride", "descriptors"),
        ("shape", "descriptors"),
        ("channel", "state contents"),
        ("tie", "untied|distinct|GEMMs|requires signed|exactly one"),
        ("order", "descriptors"),
    ],
)
@torch.inference_mode()
def test_second_shape_cannot_refresh_private_state(monkeypatch, change, pattern):
    model, _, pairs = fixture(ns=(32, 32, 32), rows=(17, 33))
    closed = []
    first = []
    exporter = torch.export.export
    close = ResidentExecutable.close

    def on_close(h):
        buf = h._fn._sparkfuse_packed_rhs
        closed.append(buf.cpu().clone())
        close(h)
        assert h._graph is None

    def export(m, ex, *a, **kw):
        if ex[0].shape[0] == 17:
            first.append(torch.cat([model.weight_0, model.weight_1, model.weight_2], dim=1).cpu())
        else:
            if change == "bytes":
                m.weight_1[2, 3].bitwise_xor_(1)
            if change == "stride":
                m.weight_0 = m.weight_0.contiguous()
            if change == "channel":
                m.channel_1[0, 0].add_(1)
            if change == "tie":
                m.weight_1 = m.weight_0
            if change == "shape":
                m.weight_1 = m.weight_1[:, :16].contiguous()
                m.channel_1 = m.channel_1.view(-1)[:16].clone().view(1, 16)
        ep = exporter(m, ex, *a, **kw)
        if ex[0].shape[0] == 33 and change == "order":
            # Swap the logical sources in the exported graph, not just the data under each name.
            params = {n.name: n for n in ep.graph.nodes if n.op == "placeholder"}
            mms = [n for n in ep.graph.nodes if n.target == torch.ops.aten._int_mm.default]
            a, b = mms[:2]
            wa, wb = a.args[1], b.args[1]
            assert wa.name in params and wb.name in params
            a.args = (a.args[0], wb)
            b.args = (b.args[0], wa)
            ep.graph_module.recompile()
        return ep

    monkeypatch.setattr(ResidentExecutable, "close", on_close)
    monkeypatch.setattr(torch.export, "export", export)
    with pytest.raises(Unsupported, match=pattern):
        capture_family(model, pairs, **OPTS)
    assert len(closed) == len(first) == 1 and torch.equal(closed[0], first[0])
