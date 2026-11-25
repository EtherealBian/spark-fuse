"""The native descriptor is created once, not specialized or cached per observed shape."""

import gc
import pytest
import torch
from examples.graphs import case
from sparkfuse.dynamic import DynamicRows
from sparkfuse.measure import compare


def dynamic(name, dtype, lo=1):
    m, a = case(name, 7, 128, dtype)
    dim = torch.export.Dim("rows", min=lo, max=129)
    ep = torch.export.export(
        m, a, dynamic_shapes=tuple({0: dim} if x.ndim == 2 else None for x in a)
    )
    return m, DynamicRows(ep)


def offset(x):
    return torch.cat((x.new_zeros(1), x.flatten()))[1:].view_as(x)


@pytest.mark.parametrize("name", ["gate", "mean", "resnorm2", "scores2"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_native_row_polymorphic(name, dtype, monkeypatch):
    m, dyn = dynamic(name, dtype)
    fn = dyn.native_dispatch()

    def forbidden(*a, **kw):
        raise AssertionError("unexpected shape specialization")

    dyn._adapt = forbidden
    monkeypatch.setattr("sparkfuse.runtime.build", forbidden)
    del dyn
    gc.collect()
    for i, n in enumerate((1, 2, 33, 97, 129, 1)):
        _, a = case(name, n, 128, dtype, seed=414 + i)
        if i % 2:
            a = tuple(map(offset, a))
        old = fn(*a)
        new = fn(*a)
        outputs, _ = torch.utils._pytree.tree_flatten(old)
        fresh, _ = torch.utils._pytree.tree_flatten(new)
        assert all(x.data_ptr() != y.data_ptr() and x.shape[0] == n for x, y in zip(outputs, fresh))
        compare(old, m(*a), exact=(1,) if name in ("resnorm2", "scores2") else ())


def test_dynamic_native_guards():
    _, d = dynamic("gate", torch.float32, lo=3)
    f = d.native_dispatch()
    x = torch.randn(15, 128, device="cuda")
    for v in (x[:2], x[:0], x.new_empty(130, 128)):
        with pytest.raises(ValueError, match="bounds"):
            f(v, v)
    for v in (
        x.cpu(),
        x[:4],
        x[0],
        x.half(),
        x[:, ::2],
        x[:1].expand_as(x),
        torch._neg_view(x),
        x.detach().requires_grad_(),
        42,
    ):
        with pytest.raises(ValueError):
            f(x, v)
    for first in (42, x[0], x.cpu(), x.half()):
        with pytest.raises(ValueError):
            f(first, x)
    with pytest.raises(ValueError):
        f(x)


@torch.inference_mode()
def test_dynamic_native_bound_state_nested_and_static_bias():
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(128, device="cuda"))

        def forward(self, bias, d, *, skip):
            z = (d["x"] + skip[0]) + bias
            return {"sum": [z], "main": z * self.weight}

    m = M()
    dim = torch.export.Dim("r", min=1, max=91)
    x = torch.randn(7, 128, device="cuda")
    bias = torch.randn_like(x[:1])
    skip = offset(x)
    ep = torch.export.export(
        m, (bias, {"x": x}), {"skip": [skip]}, dynamic_shapes=(None, {"x": {0: dim}}, [{0: dim}])
    )
    d = DynamicRows(ep)
    f = d.native_dispatch()
    for n in (1, 13, 91):
        a = x.new_empty(n, 128).normal_()
        b = offset(a)
        m.weight.mul_(1.5)
        compare(f(bias, {"x": a}, skip=[b]), m(bias, {"x": a}, skip=[b]), exact=(0, 1))
    with pytest.raises(ValueError, match="shape"):
        f(bias.expand(91, -1).contiguous(), {"x": a}, skip=[b])
    with pytest.raises(ValueError, match="pytree"):
        f(bias, {"wrong": a}, skip=[b])
    assert d._adapt.cache_info().currsize == 0


@torch.inference_mode()
def test_dynamic_native_nondefault_stream_capture():
    m, d = dynamic("resnorm2", torch.bfloat16)
    f = d.native_dispatch()
    side = torch.cuda.Stream()
    main = torch.cuda.current_stream()
    _, a = case("resnorm2", 47, 128, torch.bfloat16)
    a = tuple(map(offset, a))
    side.wait_stream(main)
    with torch.cuda.stream(side):
        f(*a)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=side):
            out = f(*a)
        a[0].add_(2)
        g.replay()
    main.wait_stream(side)
    compare(out, m(*a), exact=(1,))
