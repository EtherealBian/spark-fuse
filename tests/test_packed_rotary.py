"""A packed RoPE epilogue must preserve output ownership, layout and narrow arithmetic."""

from math import prod

import pytest
import torch
from torch import nn
from sparkfuse.partition import partition_export


class Packed(nn.Module):
    def __init__(self, mode="normal", d=14):
        super().__init__()
        self.mode = mode
        self.d = d
        self.widths = (3 * d, 2 * d, d)

    def forward(self, x, c, s):
        a, b, e = x.split(self.widths, -1)
        q, k, v = (u.clone(memory_format=torch.contiguous_format) for u in (a, b, e))
        tail = v[..., 1:4]
        if self.mode == "late":
            c = c.sin()
        if self.mode == "cycle":
            c = v
        c, s = c.unsqueeze(1), s.unsqueeze(1)
        qt, kt = (z.view(*x.shape[:2], -1, self.d).transpose(1, 2) for z in (q, k))

        def rot(z, phase):
            lo, hi = z[..., : self.d // 2], -z[..., self.d // 2 :]
            half = torch.cat((lo, hi) if self.mode == "reverse" else (hi, lo), -1)
            p = z * phase
            return p + half * s, p

        rq, prod_q = rot(qt, c)
        rk, _ = rot(kt, c.clone() if self.mode == "tables" else c)
        outputs = (rq, rk, v)
        if self.mode == "visible":
            return (*outputs, tail, e, a, b, x)
        if self.mode in ("early", "late", "cycle"):
            return (*outputs, tail)
        if self.mode == "clone":
            return (*outputs, k)
        if self.mode == "transpose":
            return (*outputs, qt)
        if self.mode == "partial":
            return (*outputs, prod_q)
        return outputs


def inputs(dt=torch.bfloat16, seq=7, shared=False, seed=7927, d=14, payload=True):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def tensor(shape, off):
        vals = torch.randn(prod(shape) + off, device="cuda", dtype=dt, generator=gen)
        vals[::17] = 0.0
        vals[::11] = -0.0
        return vals[off:].reshape(shape)

    x = tensor((2, seq, 6 * d), 1)
    if payload:
        # Signalling/quiet NaNs and signed zero in V must be transported, not converted.
        it = bits_type(x)
        x[..., 5 * d : 5 * d + 4].view(it).copy_(
            torch.tensor([-1, torch.iinfo(it).max, torch.iinfo(it).min, 1], device="cuda", dtype=it)
        )
    shape = (1 if shared else 2, seq, d)
    return x, tensor(shape, 3), tensor(shape, 5)


def bits_type(t):
    return torch.int32 if t.dtype == torch.float32 else torch.int16


def same(got, want):
    for a, b in zip(got, want, strict=True):
        assert (
            a.shape == b.shape
            and a.stride() == b.stride()
            and a.storage_offset() == b.storage_offset()
        )
        assert a.dtype == b.dtype and torch.equal(
            a.contiguous().view(bits_type(a)), b.contiguous().view(bits_type(b))
        )


def ptr(t):
    return t.untyped_storage().data_ptr()


@pytest.mark.parametrize("host", ["native", "python"])
@pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("seq,shared", [(1, False), (7, True)])
@torch.inference_mode()
def test_live_aliases_and_rounding(host, dt, seq, shared):
    m = Packed("visible")
    a = inputs(dt, seq, shared)
    g, rs = partition_export(
        torch.export.export(m, a),
        packed_rotary=True,
        split_copies=True,
        rotary_pairs=True,
        host_backend=host,
    )
    (hit,) = [r for r in rs if r.get("kind") == "packed_rotary_copy"]
    assert hit["widths"] == list(m.widths) and hit["shared_phase"] == shared
    assert hit["rematerialized_phase_views"] == 2 and not any(
        r.get("kind") in ("rotary_pair", "split_copy3") for r in rs
    )
    g.graph.lint()
    args = inputs(dt, seq, shared, 7013)
    orig = [t.contiguous().view(bits_type(t)).clone() for t in args]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out = g(*args)
        ref = m(*args)
    torch.cuda.current_stream().wait_stream(stream)
    same(out, ref)
    rq, rk, v, tail, *views = out
    assert ptr(v) == ptr(tail) and all(ptr(z) == ptr(args[0]) for z in views)
    assert len({ptr(z) for z in (*args, rq, rk, v)}) == 6
    v.zero_()
    assert all(
        torch.equal(z.contiguous().view(bits_type(z)), saved)
        for z, saved in zip(args, orig, strict=True)
    )


@pytest.mark.parametrize(
    "reason",
    [
        "late",
        "cycle",
        "clone",
        "transpose",
        "partial",
        "reverse",
        "tables",
        "off",
        "phase_stride",
        "odd",
        "input_stride",
    ],
)
@torch.inference_mode()
def test_fallback(reason):
    d = 15 if reason == "odd" else 14
    args = inputs(d=d, payload=False)
    m = Packed(reason, d)
    if reason == "phase_stride":
        args = (args[0], *(z.repeat_interleave(2, -1)[..., ::2] for z in args[1:]))
    if reason == "input_stride":
        args = (args[0].repeat_interleave(2, -1)[..., ::2], *args[1:])
    ep = torch.export.export(m, args)
    g, rs = partition_export(
        ep,
        packed_rotary=reason != "off",
        split_copies=True,
        rotary_pairs=True,
        host_backend="native",
    )
    g.graph.lint()
    assert not any(r.get("kind") == "packed_rotary_copy" for r in rs)
    same(g(*args), m(*args))


@pytest.mark.parametrize("host", ["native", "python"])
@torch.inference_mode()
def test_multiple_groups_replay(host):
    class Double(nn.Module):
        def __init__(self):
            super().__init__()
            self.part = Packed()

        def forward(self, x, y, c, s):
            return (*self.part(x, c, s), *self.part(y, c, s))

    x, c, s = inputs()
    y, _, _ = inputs(seed=601)
    fn = Double()
    aa = (x, y, c, s)
    g, rs = partition_export(torch.export.export(fn, aa), packed_rotary=True, host_backend=host)
    g.graph.lint()
    assert sum(r.get("kind") == "packed_rotary_copy" for r in rs) == 2
    g(*aa)
    cg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cg):
        out = g(*aa)
    for seed in [521, 9049]:
        xx, cc, ss = inputs(seed=seed)
        yy, _, _ = inputs(seed=seed + 1)
        for dest, src in zip(aa, (xx, yy, cc, ss), strict=True):
            dest.copy_(src)
        cg.replay()
        same(out, fn(*aa))


@pytest.mark.parametrize("host", ["native", "python"])
def test_dispatch_guards(host):
    args = inputs(payload=False)
    ep = torch.export.export(Packed(), args)
    g, rs = partition_export(ep, packed_rotary=True, host_backend=host)
    (r,) = [r for r in rs if r.get("kind") == "packed_rotary_copy"]
    run = g.get_submodule(r["module"])
    x, c, s = args
    c = c.unsqueeze(1)
    s = s.unsqueeze(1)
    for badx in (
        x.cpu(),
        x.float(),
        x[..., :-1],
        torch._neg_view(x),
        x.repeat_interleave(2, -1)[..., ::2],
    ):
        with pytest.raises(ValueError):
            run(badx, c, s)
    with pytest.raises(ValueError, match="inference only"):
        run(x.detach().requires_grad_(), c, s)
    with pytest.raises(ValueError):
        run(x, c, s.to(torch.float32))
