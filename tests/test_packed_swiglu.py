"""Split view aliases, public activation rounding, offsets, guards and live packed weights."""

from math import prod
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from sparkfuse.packed_swiglu import PackedSwiGLUProjection
from sparkfuse.partition import partition_export


class Gate(nn.Module):
    def __init__(self, method="chunk", reverse=False, views=False):
        super().__init__()
        self.method = method
        self.reverse = reverse
        self.views = views

    def forward(self, x):
        d = x.shape[-1] // 2
        if self.method == "chunk":
            a, b = x.chunk(2, -1)
        elif self.method == "split":
            a, b = x.split(d, -1)
        else:
            a, b = x.split([d, d], -1)
        if self.reverse:
            a, b = b, a
        act = F.silu(a)
        early = act.flatten() if self.views else None
        y = b * act if self.reverse else act * b
        if self.views:
            return y, early, act, (a, b), x, a[..., 1:]
        return y


def bits(x, y):
    assert x.shape == y.shape and x.dtype == y.dtype and x.stride() == y.stride()
    assert torch.equal(x.isnan(), y.isnan())
    keep = ~x.isnan()
    assert torch.equal(x[keep].view(torch.int16), y[keep].view(torch.int16))


def ptr(x):
    return x.untyped_storage().data_ptr()


@pytest.mark.parametrize("host", ["native", "python"])
@pytest.mark.parametrize(
    "method,reverse,visible,dt,shape",
    [
        ("chunk", False, True, torch.bfloat16, (2, 1, 128)),
        ("split", True, True, torch.float16, (3, 14)),
        ("sizes", True, False, torch.bfloat16, (2, 5, 24)),
        ("chunk", False, False, torch.float16, (16,)),
    ],
)
@torch.inference_mode()
def test_packed_gate_stream_alignment_views(host, method, reverse, visible, dt, shape):
    m = Gate(method, reverse, visible)
    x = torch.randn(shape, device="cuda", dtype=dt)
    ep = torch.export.export(m, (x,))
    gm, rr = partition_export(ep, host_backend=host, packed_gates=True)
    (r,) = [r for r in rr if r.get("kind") == "packed_swiglu"]
    assert r["visible_activation"] == visible and r["gate_half"] == int(reverse)
    for off in (0, 1, 2, 4):
        xx = torch.randn(prod(shape) + off, dtype=dt, device="cuda")[off:].reshape(shape)
        before = xx.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            a, b = gm(xx), m(xx)
        torch.cuda.current_stream().wait_stream(stream)
        bits(xx, before)
        if visible:
            from torch.utils._pytree import tree_leaves

            x1, x2 = tree_leaves(a), tree_leaves(b)
            for i, (v, w) in enumerate(zip(x1, x2, strict=True)):
                bits(v, w)
                assert v.storage_offset() == w.storage_offset()
                assert [ptr(t) == ptr(v) for t in x1] == [ptr(t) == ptr(w) for t in x2]
        else:
            bits(a, b)
            assert ptr(a) != ptr(xx)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_all_gate_encodings_and_rounded_product(dtype):
    allbits = (
        torch.arange(-32768, 32768, device="cuda", dtype=torch.int32)
        .to(torch.int16)
        .view(dtype)
        .reshape(256, 256)
    )
    g = torch.Generator(device="cuda").manual_seed(1771)
    up = torch.randn(allbits.shape, dtype=dtype, device="cuda", generator=g)
    x = torch.cat([allbits, up], -1)
    m = Gate(views=True)
    gm, rr = partition_export(
        torch.export.export(m, (x,)), packed_gates=True, host_backend="native"
    )
    assert any(r.get("kind") == "packed_swiglu" for r in rr)
    a, b = gm(x), m(x)
    bits(a[0], b[0])
    bits(a[2], b[2])
    assert (
        int(
            (
                a[0].nan_to_num()
                != ((allbits.float() / (1 + (-allbits.float()).exp())) * up.float())
                .to(dtype)
                .nan_to_num()
            ).sum()
        )
        > 0
    )


@torch.inference_mode()
def test_reload_real_weight_replay_no_stale_cache():
    dt = torch.bfloat16
    g, u = (nn.Linear(32, 64, bias=False, device="cuda", dtype=dt) for _ in range(2))
    m = PackedSwiGLUProjection.from_linears(g, u)
    assert (
        list(m.state_dict()) == ["weight"]
        and len({ptr(m.weight), ptr(g.weight), ptr(u.weight)}) == 3
    )
    x = torch.randn(4, 32, dtype=dt, device="cuda")
    gm, rr = partition_export(
        torch.export.export(m, (x,)), packed_gates=True, host_backend="native"
    )
    assert any(r.get("kind") == "packed_swiglu" for r in rr)
    assert ptr(gm.weight) == ptr(m.weight)
    original = gm(x)
    bits(original, m(x))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = gm(x)
    graph.replay()
    bits(captured, original)
    g.weight.zero_()
    bits(gm(x), original)
    identity = ptr(m.weight)
    m.reload_weights_(g.weight, u.weight)
    assert identity == ptr(m.weight)
    graph.replay()
    bits(captured, m(x))
    assert torch.count_nonzero(captured) == 0 and torch.count_nonzero(original) > 0
    old = m.weight.clone()
    a, b = m.weight.chunk(2, 0)
    m.reload_weights_(b, a)
    assert torch.equal(m.weight, torch.cat(old.chunk(2, 0)[::-1]))
    with pytest.raises(ValueError):
        m.reload_weights_(u.weight.float(), g.weight)
    with pytest.raises(ValueError):
        PackedSwiGLUProjection.from_linears(nn.Linear(32, 64), u)


@pytest.mark.parametrize(
    "kind", ["wrong_axis", "unequal", "same_half", "noncontiguous", "fp32", "sigmoid"]
)
@torch.inference_mode()
def test_reject_other_graphs(kind):
    class M(nn.Module):
        def forward(self, x):
            if kind == "wrong_axis":
                a, b = x.chunk(2, 0)
            elif kind == "unequal":
                a, b, c = x.split([3, 3, 2], -1)
            else:
                a, b = x.chunk(2, -1)
            if kind == "same_half":
                b = a
            return (a.sigmoid() if kind == "sigmoid" else F.silu(a)) * b

    dt = torch.float32 if kind == "fp32" else torch.bfloat16
    x = torch.randn(8, 8, device="cuda", dtype=dt)
    x = x.T if kind == "noncontiguous" else x
    m = M()
    gm, rr = partition_export(torch.export.export(m, (x,)), packed_gates=True)
    assert not any(r.get("kind") == "packed_swiglu" for r in rr)
    torch.testing.assert_close(gm(x), m(x))


@pytest.mark.parametrize("enabled,schedule", [(True, "typo"), (False, "scalar")])
def test_reject_invalid_packed_gate_policy(enabled, schedule):
    with pytest.raises(ValueError, match="packed_gate_schedule"):
        partition_export(None, packed_gates=enabled, packed_gate_schedule=schedule)
