import itertools
import pytest
import torch
from torch import nn
from sparkfuse import Schedule, Unsupported, compile_export
from sparkfuse.ir import broadcast_domain, lower
from sparkfuse.dynamic import DynamicRows
from sparkfuse.measure import compare
from examples.position_bias import CausalPositionPrior, inputs


class Outer(nn.Module):
    def forward(self, x, y):
        z = x * y
        return z, z.mean(-1, True), x.neg()


def test_shape_join_matches_torch():
    candidates = ((), (1,), (5,), (2, 1), (1, 5), (1, 3, 1), (4, 1, 5))
    for shapes in itertools.product(candidates, repeat=3):
        try:
            ref = tuple(torch.broadcast_shapes(*shapes))
        except RuntimeError:
            with pytest.raises(Unsupported, match="broadcast"):
                broadcast_domain(shapes)
        else:
            assert broadcast_domain(shapes) == ref


def test_join_overflow_before_build():
    class Mul(nn.Module):
        def forward(self, x, y):
            return x * y

    a, b = torch.empty(65536, 1, device="meta"), torch.empty(1, 65536, device="meta")
    ep = torch.export.export(Mul(), (a, b))
    with pytest.raises(Unsupported, match="32-bit"):
        lower(ep)


def test_reduction_not_replicated_over_new_domain():
    class M(nn.Module):
        def forward(self, x, y):
            return x.mean(-1, True) + y

    a, b = torch.randn(1, 7), torch.randn(9, 1)
    with pytest.raises(Unsupported, match="reduction must span"):
        lower(torch.export.export(M(), (a, b)))


@pytest.mark.parametrize("sched", [Schedule(128, "block", False, 1), Schedule(64, "warp", True, 4)])
def test_dynamic_outer_mixed_output(sched):
    args = (torch.randn(3, 1, device="cuda"), torch.randn(1, 65, device="cuda"))
    m = Outer()
    sym = torch.export.Dim("tokens", min=1, max=47)
    k = DynamicRows(torch.export.export(m, args, dynamic_shapes=({0: sym}, None)), schedule=sched)
    for count in (1, 2, 3, 17, 47):
        a = torch.randn(count, 1, device="cuda")
        compare(k(a, args[1]), m(a, args[1]), exact=(0, 2))
    with pytest.raises(ValueError, match="shape"):
        k(args[0], args[1].repeat(3, 1))


@pytest.mark.parametrize("sched", [Schedule(64, "warp", True, 4), Schedule(128, "block", False, 1)])
def test_three_axis_position_prior(sched):
    a = inputs(3, 7, 65, "cuda", 1, offset=1)
    m = CausalPositionPrior()
    k = compile_export(torch.export.export(m, a), schedule=sched)
    assert k.plan.domain == (3, 7, 65) and all(tuple(x.shape) != k.plan.domain for x in a)
    compare(k(*a), m(*a), exact=(1,))
    with pytest.raises(ValueError, match="layout"):
        k(*a[:3], torch.zeros(1, 65, 7, device="cuda").transpose(1, 2))


@pytest.mark.parametrize("target,base", [(torch.float16, 2048.0), (torch.bfloat16, 256.0)])
@pytest.mark.parametrize("sched", [Schedule(64, "warp", True, 4), Schedule(128, "block")])
def test_mean_dtype_rounds_before_accumulation(target, base, sched):
    class Mean(nn.Module):
        def forward(self, x):
            return x.mean(-1, True, dtype=target)

    m = Mean()
    x = torch.tensor([base + 0.75, -base], device="cuda").repeat(3, 16)
    eager = m(x)
    assert torch.all(eager == 0) and torch.all(x.mean(-1) == 0.375)
    k = compile_export(torch.export.export(m, (x,)), schedule=sched)
    compare(k(x), eager, exact=(0,))
