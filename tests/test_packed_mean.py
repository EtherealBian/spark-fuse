import pytest
import torch
from torch import nn

from examples.graphs import DecomposedNorm, ResidualNorm, case
from sparkfuse import Schedule, Unsupported, compile_export, generate, lower
from sparkfuse.dynamic import DynamicRows
from sparkfuse.packed_mean import match
from test_mean_tree import FullTrace, same_bits, shifted


class Visible(nn.Module):
    def forward(self, weight, skip, x):
        r = x + skip
        f = r.float()
        inv = (2**-12 + f.square().mean(-1, keepdim=True)).rsqrt()
        return r, (inv * f).to(x.dtype) * weight


def inputs(n, h, dt, seed, offsets, visible):
    _, a = case("decomp", n, h, dt, seed)
    a = tuple(shifted(v, o) for v, o in zip(a, offsets, strict=True))
    return a[::-1] if visible else a


@pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "n,h,nt,vec,visible",
    [
        (16, 1024, 128, 2, False),
        (17, 4100, 256, 4, True),
        (31, 8160, 512, 8, False),
        (26, 1536, 128, 8, True),
    ],
)
@torch.inference_mode()
def test_tile_observable_and_alignment(n, h, nt, vec, visible, dt):
    m = Visible() if visible else DecomposedNorm()
    a = inputs(n, h, dt, 99491, (0, 0, 0), visible)
    k = compile_export(
        torch.export.export(m, a), schedule=Schedule(nt, "mean_packed_tile", True, vec)
    )

    def wrap(y):
        return y if visible else (y,)

    for i, offsets in enumerate([(0, 0, 0), (1, 0, 0), (0, 3, 0), (0, 0, 1), (1, 2, 3)]):
        a = inputs(n, h, dt, 110237 + i, offsets, visible)
        originals = tuple(v.clone() for v in a)
        actual = wrap(k(*a))
        again = wrap(k(*a))
        same_bits(actual, wrap(m(*a)))
        same_bits(actual, again)
        same_bits(a, originals)
        ptrs = [v.untyped_storage().data_ptr() for v in (*actual, *again, *a)]
        assert len(ptrs) == len(set(ptrs))


@pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_tile_nonfinite_signed_zero_subnormal_and_replay(dt):
    m = Visible()
    n, h = 19, 1024
    a = inputs(n, h, dt, 116167, (1, 0, 1), True)
    k = compile_export(
        torch.export.export(m, a), schedule=Schedule(128, "mean_packed_tile", True, 4)
    )
    w, s, x = a
    tiny = torch.finfo(dt).tiny
    sub = tiny * torch.finfo(dt).eps
    codes = torch.tensor([-0.0, 0.0, sub, -sub, tiny, -tiny, 1.0, -1.0], device="cuda", dtype=dt)
    w.copy_(codes.repeat(h // 8))
    x[:4].copy_(codes.repeat(4, h // 8))
    s[:4].copy_(x[:4])
    x[4, 13] = float("inf")
    x[5, 0] = float("nan")
    s[6, 17] = -float("inf")
    x[6, 17] = float("inf")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        prepared = k.prepare(*a)
        prepared()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            out = prepared()
        for swap in (False, True):
            if swap:
                x[:4].neg_()
                s[7].copy_(x[8])
                w.neg_()
            g.replay()
            ref = m(*a)
            for u, v in zip(out, ref, strict=True):
                mask = ~v.isnan()
                assert torch.equal(u.isnan(), v.isnan())
                same_bits((u[mask],), (v[mask],))
    torch.cuda.current_stream().wait_stream(stream)


def test_false_friends_and_explicit_schedule_only():
    dt = torch.bfloat16
    a = tuple(torch.empty(s, dtype=dt) for s in ((17, 1024), (17, 1024), (1024,)))
    sched = Schedule(128, "mean_packed_tile", True, 4)

    class LateCast(nn.Module):
        def forward(self, x, s, w):
            f = (x + s).float()
            return (w * (f * (f.square().mean(-1, True) + 1e-6).rsqrt())).to(x.dtype)

    class OtherOp(nn.Module):
        def forward(self, x, s, w):
            f = (x - s).float()
            return w * (f * (f.square().mean(-1, True) + 1e-6).rsqrt()).to(x.dtype)

    for m in (LateCast(), OtherOp(), FullTrace(), ResidualNorm()):
        p = lower(torch.export.export(m, a))
        assert match(p) is None
        with pytest.raises(Unsupported, match="mean_packed_tile|vector pack"):
            generate(p, sched)
    for args in (
        a[:1] + (a[1][:1], a[2]),
        tuple(u.float() for u in a),
        tuple(u[..., :1017] for u in a),
    ):
        args = tuple(v.contiguous() for v in args)
        p = lower(torch.export.export(DecomposedNorm(), args))
        assert match(p) is None
    p = lower(torch.export.export(DecomposedNorm(), a))
    assert match(p) is not None
    for t, c, v in [(32, True, 4), (256, False, 4), (256, True, 1)]:
        with pytest.raises(Unsupported, match="mean_packed_tile"):
            generate(p, Schedule(t, "mean_packed_tile", c, v))
    odd = tuple(torch.empty(s, dtype=dt) for s in ((17, 4100), (17, 4100), (4100,)))
    with pytest.raises(Unsupported, match="mean_packed_tile"):
        generate(
            lower(torch.export.export(DecomposedNorm(), odd)),
            Schedule(256, "mean_packed_tile", True, 8),
        )


@torch.inference_mode()
def test_dynamic_bound_and_parameter_order():
    m = Visible()
    a = inputs(22, 1024, torch.bfloat16, 91457, (0, 1, 0), True)

    def make(hi):
        d = torch.export.Dim("rows", min=16, max=hi)
        return torch.export.export(m, a, dynamic_shapes=(None, {0: d}, {0: d}))

    s = Schedule(128, "mean_packed_tile", True, 8)
    f = DynamicRows(make(49), schedule=s).native_dispatch()
    for n in (16, 37, 49):
        x = inputs(n, 1024, torch.bfloat16, 22027 + n, (0, 2, 0), True)
        same_bits(f(*x), m(*x))
    with pytest.raises(Unsupported, match="row bound"):
        DynamicRows(make(17000), schedule=s)
