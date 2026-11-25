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


class TwoCollectives(nn.Module):
    def forward(self, x, skip, w, b):
        r = x + skip
        h = torch.nn.functional.layer_norm(r, (x.shape[-1],), w, b)
        return (h * 0.125).softmax(-1), r, h


@torch.inference_mode()
@pytest.mark.parametrize("host", ["python", "native"])
def test_split_multi_collective_preserves_visible_values(host):
    m = TwoCollectives()
    a = tuple(torch.randn(shape, device="cuda") for shape in ((7, 97), (7, 97), (97,), (97,)))
    ep = torch.export.export(m, a)
    old, old_records = partition_export(ep, split_collectives=False)
    assert not any("module" in r for r in old_records)
    compare(old(*a), m(*a), exact=(1,))
    new, records = partition_export(ep, host_backend=host)
    assert len(records) == 2 and all(
        "module" in r and r["split"]["planned_regions"] == 2 for r in records
    )
    compare(new(*a), m(*a), exact=(1,))
    new.graph.lint()
    assert set(records[0]["live_outs"]) == {"add", "layer_norm"}


def test_split_with_fallback_between_residuals_and_visible_identity():
    class Block(nn.Module):
        def forward(self, x, skip):
            h = torch.nn.functional.layer_norm(x + skip, (16,))
            contracted = h @ x.T
            return torch.nn.functional.layer_norm(contracted + h, (16,)), h

    m = Block()
    args = tuple(torch.randn(16, 16, device="cuda") for _ in range(2))
    ep = torch.export.export(m, args)
    gm, d = partition_export(ep)
    assert len([r for r in d if "module" in r]) == 2
    assert "matmul" in str(gm.graph) and all(
        "insertion point" in r["split"]["original_rejection"] for r in d if "module" in r
    )
    compare(gm(*args), m(*args))
    gm.graph.lint()


def test_multi_collective_keeps_observable_alias_outside():
    class M(nn.Module):
        def forward(self, x, y):
            h = (x + y).softmax(-1)
            z = h.to(h.dtype)
            r = (z + x).softmax(-1)
            return h, z, r

    a = (torch.randn(5, 17, device="cuda"), torch.randn(5, 17, device="cuda"))
    m = M()
    gm, records = partition_export(torch.export.export(m, a))
    out = gm(*a)
    compare(out, m(*a))
    assert out[0].data_ptr() == out[1].data_ptr()
