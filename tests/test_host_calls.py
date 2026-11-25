import pytest
import torch
from torch import nn
from sparkfuse.host_calls import specialize_host_calls
from sparkfuse.partition import partition_export


class Patch(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, x):
        h = self.linear(x)
        a = h.view(2, 2, 4).permute(1, 0, 2).transpose(0, 1)
        a = torch.nn.functional.dropout(a, 0.25, False)
        b = torch.nn.functional.gelu(a, approximate="tanh")
        return {"views": (h, a), "scores": b @ a.transpose(-1, -2)}


@torch.inference_mode()
def test_host_rewrites_exact_alias_guards_and_parameter_liveness():
    m = Patch().to("cuda").eval()
    x = torch.randn(2, 8, device="cuda")
    gm, _ = partition_export(torch.export.export(m, (x,)), min_ops=32)
    guards = [
        n.target for n in gm.graph.nodes if n.op == "call_module" and "guard" in str(n.target)
    ]
    audit = specialize_host_calls(gm)
    assert {r["from"] for r in audit} >= {
        "aten.dropout.default",
        "aten.linear.default",
        "aten.matmul.default",
        "aten.gelu.default",
        "aten.view.default",
        "aten.permute.default",
        "aten.transpose.int",
    }
    assert specialize_host_calls(gm) == []
    assert guards == [
        n.target for n in gm.graph.nodes if n.op == "call_module" and "guard" in str(n.target)
    ]
    for _ in range(2):
        result = gm(x)
        torch.testing.assert_close(result, m(x), rtol=0, atol=0)
        v, w = result["views"]
        assert v.untyped_storage().data_ptr() == w.untyped_storage().data_ptr()
        assert v.data_ptr() == w.data_ptr()
        assert w.stride() == m(x)["views"][1].stride()
        m.linear.weight.add_(0.125)
        x = x.roll(1, -1)
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        gm(x[:1])


@pytest.mark.parametrize(
    "p,train", [(0.1, True), (-0.1, False), (1.5, False), (float("nan"), False)]
)
def test_do_not_hide_dropout_probability_or_randomness(p, train):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    node = graph.call_function(torch.ops.aten.dropout.default, (x, p, train))
    graph.output(node)
    gm = torch.fx.GraphModule({}, graph)
    assert specialize_host_calls(gm) == []
    assert node.target is torch.ops.aten.dropout.default


def test_preserve_other_overloads_and_explicit_memory_formats():
    g = torch.fx.Graph()
    x = g.placeholder("x")
    y = g.call_function(
        torch.ops.aten.contiguous.default, (x,), dict(memory_format=torch.channels_last)
    )
    z = g.call_function(torch.ops.aten.view.dtype, (y, torch.int32))
    g.output(z)
    gm = torch.fx.GraphModule({}, g)
    assert specialize_host_calls(gm) == []


def test_contiguous_uses_same_copy_or_alias_semantics():
    g = torch.fx.Graph()
    x = g.placeholder("x")
    y = g.call_function(torch.ops.aten.contiguous.default, (x,))
    g.output(y)
    gm = torch.fx.GraphModule({}, g)
    assert len(specialize_host_calls(gm)) == 1
    base = torch.arange(32, device="cuda").view(8, 4)
    for v in (base, base.t()):
        out = gm(v)
        assert torch.equal(out, v) and out.is_contiguous()
        assert (out.data_ptr() == v.data_ptr()) == v.is_contiguous()
