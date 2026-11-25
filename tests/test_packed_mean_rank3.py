"""Rank-3 native dispatch must preserve logical shapes, output order, and storage guards."""

import pytest
import torch
from torch import nn

from sparkfuse import Schedule, Unsupported, compile_export, generate, lower
from sparkfuse.dynamic import DynamicRows
from sparkfuse.native import native_dispatch
from sparkfuse.packed_mean import match
from test_mean_tree import same_bits, shifted


class NamedStream(nn.Module):
    def __init__(self, order):
        super().__init__()
        self.order = order

    def forward(self, w, x, s):
        r = s + x
        f = r.float()
        y = w * (f * (2**-13 + f.square().mean(dim=-1, keepdim=True)).rsqrt()).to(x.dtype)
        if self.order == "only":
            return y
        return (r, y) if self.order == "raw-first" else (y, r)


def args_for(b, t, h, dt, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    return tuple(
        torch.randn(shape, generator=g, dtype=dt, device=device)
        for shape in ((h,), (b, t, h), (b, t, h))
    )


@pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape,vec,order",
    [((2, 17, 1536), 8, "raw-first"), ((1, 31, 1024), 4, "raw-last"), ((16, 1, 4100), 2, "only")],
)
@torch.inference_mode()
def test_rank3_native_offsets_shape_and_graph(shape, vec, order, dt):
    m = NamedStream(order)
    a = args_for(*shape, dt, "cuda", 498237)
    ep = torch.export.export(m, a)
    k = compile_export(ep, schedule=Schedule(256, "mean_packed_tile", True, vec))
    native = native_dispatch(k)
    assert k.plan.domain == shape and match(k.plan) is not None

    def wrap(o):
        return (o,) if order == "only" else o

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for i, offsets in enumerate([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 3), (1, 3, 2)]):
            a = tuple(
                shifted(v, n)
                for v, n in zip(args_for(*shape, dt, "cuda", 787981 + i), offsets, strict=True)
            )
            w, x, s = a
            # Distinguish adjacent batches and sequence positions; poison exactly one row.
            x.reshape(-1, shape[-1])[0, :4] = x.new_tensor(
                [-0.0, 0.0, torch.finfo(dt).smallest_normal * torch.finfo(dt).eps, float("inf")]
            )
            s.reshape(-1, shape[-1])[0, :4].copy_(x.reshape(-1, shape[-1])[0, :4])
            s.reshape(-1, shape[-1])[-1, 5] = float("nan")
            native(*a)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                out = wrap(native(*a))
            for replay in range(2):
                if replay:
                    x.neg_()
                    w.neg_()
                g.replay()
                ref = wrap(m(*a))
                for v, t in zip(out, ref, strict=True):
                    assert v.shape == t.shape == shape
                    assert torch.equal(v.isnan(), t.isnan())
                    good = ~t.isnan()
                    same_bits((v[good],), (t[good],))
            pointers = [v.untyped_storage().data_ptr() for v in (*out, *wrap(native(*a)), *a)]
            assert len(set(pointers)) == len(pointers)
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.parametrize(
    "xsh,ssh",
    [
        ((2, 17, 1024), (1, 17, 1024)),
        ((2, 17, 1024), (2, 1, 1024)),
        ((2, 17, 1024), (17, 1024)),
        ((2, 2, 9, 1024), (2, 2, 9, 1024)),
        ((1024,), (1024,)),
        ((2, 3, 1024), (2, 3, 1024)),
    ],
)
def test_not_all_rank_or_broadcast_is_flattenable(xsh, ssh):
    m = NamedStream("raw-last")
    a = tuple(torch.empty(sh, dtype=torch.bfloat16) for sh in [(1024,), xsh, ssh])
    ep = torch.export.export(m, a)
    try:
        p = lower(ep)
    except Unsupported:
        return
    assert match(p) is None
    with pytest.raises(Unsupported):
        generate(p, Schedule(256, "mean_packed_tile", True, 4))


def test_no_symbolic_or_transposed_rank3():
    m = NamedStream("only")
    a = args_for(2, 17, 1024, torch.float16, "cpu", 99419)
    dim = torch.export.Dim("sequence", min=16, max=64)
    ep = torch.export.export(m, a, dynamic_shapes=(None, {1: dim}, {1: dim}))
    with pytest.raises(Unsupported, match="rank-2"):
        DynamicRows(ep, schedule=Schedule(256, "mean_packed_tile", True, 4))
    with pytest.raises(Unsupported, match="contiguous"):
        lower(torch.export.export(m, (a[0], a[1].transpose(0, 1), a[2].transpose(0, 1))))
