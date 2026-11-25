"""A one-collective component may span a fallback GEMM and still have a useful legal prefix."""

import operator
import pytest
import torch
from torch import nn
from sparkfuse.partition import ordered_collectives, partition_export


class AcrossProjection(nn.Module):
    def __init__(self, observable, alias):
        super().__init__()
        self.observable = observable
        self.alias = alias

    def forward(self, x, skip, w, matrix):
        r = x + skip
        f = r.float()
        mean = f.square().mean(-1, keepdim=True)
        normalized = w * (f * torch.rsqrt(mean + 1e-6)).to(x.dtype)
        y = r + torch.nn.functional.linear(normalized, matrix)
        if self.alias:
            return y, r, r.view_as(r)
        return (y, r) if self.observable else y


@pytest.mark.parametrize("observable,alias", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("host", ["python", "native"])
@torch.inference_mode()
def test_salvage_norm_without_crossing_projection(observable, alias, host):
    g = torch.Generator(device="cuda").manual_seed(4313)
    a = tuple(
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=g)
        for shape in [(17, 1024), (17, 1024), (1024,), (1024, 1024)]
    )
    m = AcrossProjection(observable, alias)
    ep = torch.export.export(m, a)
    kw = dict(host_backend=host, schedule_policy=ordered_collectives)
    old, rs = partition_export(ep, split_barriers=False, **kw)
    assert any("insertion point" in x.get("fallback", "") for x in rs)
    new, rs = partition_export(ep, **kw)
    (hit,) = [r for r in rs if "module" in r]
    assert (
        hit["split"]["reason"] == "fallback_barrier"
        and len(hit["live_outs"]) == 2
        and hit["schedule"]["layout"] == "mean_cohort"
    )
    graph = list(new.graph.nodes)
    call = next(n for n in graph if n.target == hit["module"])
    proj = next(n for n in graph if n.target == torch.ops.aten.linear.default)
    pairs = {n.args[1]: n for n in call.users if n.target is operator.getitem}
    assert len(pairs) == 2
    assert proj.args[0] is pairs[1] and graph.index(call) < graph.index(proj)
    assert any(
        n.target is torch.ops.aten.add.Tensor
        and proj in n.all_input_nodes
        and pairs[0] in n.all_input_nodes
        for n in graph
    )
    for code in (new, old):
        y = code(*a)
        ref = m(*a)
        v = y if isinstance(y, tuple) else (y,)
        expected = ref if isinstance(ref, tuple) else (ref,)
        assert all(
            torch.equal(x.view(torch.int16), r.view(torch.int16))
            for x, r in zip(v, expected, strict=True)
        )
        if alias:
            assert y[1].data_ptr() == y[2].data_ptr()
    new.graph.lint()


@torch.inference_mode()
def test_non_collective_diamond_not_indiscriminately_split():
    class NoReduce(nn.Module):
        def forward(self, x, w):
            a = (x + x).relu()
            return a + (a @ w)

    a = (torch.randn(16, 16, device="cuda"), torch.randn(16, 16, device="cuda"))
    m = NoReduce()
    gm, rs = partition_export(torch.export.export(m, a))
    assert not any("module" in r for r in rs)
    assert any("insertion point" in r.get("fallback", "") for r in rs)
    torch.testing.assert_close(gm(*a), m(*a))
