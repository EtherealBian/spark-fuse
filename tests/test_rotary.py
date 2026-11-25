"""The optional index-program pass must preserve narrow arithmetic, layouts, and graph liveness."""

import pytest
import torch
from torch import nn
from sparkfuse.partition import partition_export


def branch(x, c, s, mode="normal"):
    d = x.shape[-1] // 2
    hi = -x[..., d:]
    lo = x[..., :d]
    rot = torch.cat((lo, hi) if mode == "reverse" else (hi, lo), -1)
    return torch.add(x * c, rot * s, alpha=2 if mode == "scaled" else 1), hi


class Pair(nn.Module):
    def __init__(self, mode="normal"):
        super().__init__()
        self.mode = mode

    def forward(self, q, k, c, s):
        qy, hi = branch(q, c, s, self.mode)
        if self.mode == "barrier":
            k = k + torch.sin(qy)
        ky, _ = branch(k, c.clone() if self.mode == "tables" else c, s)
        if self.mode == "twice":
            qy, _ = branch(qy, c, s)
            ky, _ = branch(ky, c, s)
        return (qy, ky, hi) if self.mode == "expose" else (qy, ky)


def make(dt, orders=(True, False), same_heads=False, shared=True, seed=79, seq=7, width=14):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def inp(shape, offset):
        n = 1
        for z in shape:
            n *= z
        flat = torch.randn(n + offset, device="cuda", dtype=dt, generator=g)
        flat[::13] = -0.0
        flat[::17] = 0.0
        return flat[offset:].view(shape)

    args = []
    for h, sw, off in zip((5, 5 if same_heads else 2), orders, (1, 2), strict=True):
        a = inp((2, seq, h, width) if sw else (2, h, seq, width), off)
        args.append(a.transpose(1, 2) if sw else a)
    return (*args, *(inp((1 if shared else 2, 1, seq, width), off) for off in (3, 5)))


def assert_bits(a, b):
    typ = torch.int32 if b.dtype == torch.float32 else torch.int16
    assert a.stride() == b.stride() and a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.contiguous().view(typ), b.contiguous().view(typ))


@pytest.mark.parametrize("host", ["native", "python"])
@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "orders,shared", [((False, False), False), ((True, False), True), ((True, True), False)]
)
@torch.inference_mode()
def test_pair_layout_and_rounding(host, dt, orders, shared):
    a = make(dt, orders, shared=shared)
    m = Pair()
    gm, rs = partition_export(torch.export.export(m, a), rotary_pairs=True, host_backend=host)
    (hit,) = [r for r in rs if r.get("kind") == "rotary_pair"]
    assert hit["physical_order"] == ["BSHD" if sw else "BHSD" for sw in orders]
    assert len(hit["ops"]) == 14 and len(hit["live_outs"]) == 2
    fresh = make(dt, orders, shared=shared, seed=911)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        y = gm(*fresh)
        b = m(*fresh)
    torch.cuda.current_stream().wait_stream(stream)
    for u, v in zip(y, b, strict=True):
        assert_bits(u, v)
    assert len({t.untyped_storage().data_ptr() for t in (*fresh, *y)}) == 6


@pytest.mark.parametrize(
    "mode", ["reverse", "scaled", "expose", "tables", "barrier", "strided_phase", "odd"]
)
@torch.inference_mode()
def test_do_not_contract_unproved_graph(mode):
    m = Pair(mode)
    a = make(torch.bfloat16, same_heads=True, width=15 if mode == "odd" else 14)
    if mode == "strided_phase":
        a = (*a[:2], *(v.repeat_interleave(2, dim=-1)[..., ::2] for v in a[2:]))
    gm, rs = partition_export(torch.export.export(m, a), rotary_pairs=True)
    assert not any(r.get("kind") == "rotary_pair" for r in rs)
    for u, v in zip(gm(*a), m(*a), strict=True):
        assert_bits(u, v)


@torch.inference_mode()
def test_two_chained_pairs_use_rewired_liveins():
    m = Pair("twice")
    a = make(torch.bfloat16)
    gm, rs = partition_export(torch.export.export(m, a), rotary_pairs=True, host_backend="native")
    assert sum(r.get("kind") == "rotary_pair" for r in rs) == 2
    gm.graph.lint()
    for u, v in zip(gm(*a), m(*a), strict=True):
        assert_bits(u, v)


@pytest.mark.parametrize("host", ["native", "python"])
def test_launcher_guards_and_signs(host):
    m = Pair()
    a = make(torch.float16, orders=(False, False))
    gm, rs = partition_export(torch.export.export(m, a), rotary_pairs=True, host_backend=host)
    (hit,) = [r for r in rs if r.get("kind") == "rotary_pair"]
    run = gm.get_submodule(hit["module"])
    q, *rest = a
    for bad in (
        q.cpu(),
        q.float(),
        q[..., :-1],
        torch._neg_view(q),
        q.transpose(1, 2).contiguous().transpose(1, 2),
    ):
        with pytest.raises(ValueError):
            run(bad, *rest)
    with pytest.raises(ValueError, match="inference only"):
        run(q.detach().requires_grad_(), *rest)
    with torch.inference_mode():
        for u, v in zip(run(*a), m(*a), strict=True):
            assert_bits(u, v)
