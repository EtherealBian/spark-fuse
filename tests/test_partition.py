import pytest
import torch
from torch import nn
from sparkfuse.partition import partition_export
from sparkfuse.measure import compare
from sparkfuse import Unsupported


class Branch(nn.Module):
    def forward(self, x, y):
        a = x + y
        v = torch.nn.functional.silu(a)
        b = y @ y.t()
        return v * b, a


@torch.inference_mode()
def test_across_independent_fallback():
    m = Branch()
    a = tuple(torch.randn(16, 16, device="cuda") for _ in range(2))
    ep = torch.export.export(m, a)
    gm, r = partition_export(ep)
    assert sum("module" in d for d in r) == 1
    assert len(next(d["live_outs"] for d in r if "module" in d)) == 2
    compare(gm(*a), m(*a))


@torch.inference_mode()
def test_reject_contraction_with_outside_dependency():
    class Diamond(nn.Module):
        def forward(self, x):
            a = torch.nn.functional.silu(x)
            b = a @ x
            return a * b

    m = Diamond()
    a = (torch.randn(16, 16, device="cuda"),)
    gm, r = partition_export(torch.export.export(m, a))
    assert all("module" not in d for d in r)
    assert any("insertion point" in d["fallback"] for d in r)
    compare(gm(*a), m(*a))


@torch.inference_mode()
def test_reject_mutating_graph():
    class Mut(nn.Module):
        def forward(self, x):
            return x.add_(2).relu()

    a = (torch.zeros(4, 4, device="cuda"),)
    with pytest.raises(Unsupported):
        partition_export(torch.export.export(Mut(), a))
