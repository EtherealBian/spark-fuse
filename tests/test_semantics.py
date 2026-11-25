"""Semantic edge cases not exercised by random, normally distributed activations."""

import pytest
import torch
from torch import nn
from sparkfuse import compile_export, Unsupported, Schedule


@pytest.mark.parametrize("copy", [False, True])
def test_returned_noop_cast_alias(copy):
    class M(nn.Module):
        def forward(self, x):
            return torch.ops.aten.to.dtype(x, x.dtype, False, copy)

    x = torch.randn(9, 11, device="cuda")
    m = M()
    ep = torch.export.export(m, (x,))
    ref = m(x)
    assert (ref.data_ptr() == x.data_ptr()) == (not copy)
    if not copy:
        with pytest.raises(Unsupported, match="aliased outputs"):
            compile_export(ep)
    else:
        y = compile_export(ep)(x)
        assert y.data_ptr() != x.data_ptr()
        torch.testing.assert_close(y, ref, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("width", [1, 127])
def test_native_rms_default_epsilon_near_zero(dtype, width):
    class M(nn.Module):
        def forward(self, x):
            return torch.nn.functional.rms_norm(x, (width,))

    # Very small values differentiate FP32 epsilon from BF16/FP16 epsilon.
    x = torch.tensor([0, 1e-5, -1e-6, 0.5], device="cuda", dtype=dtype)[:, None].repeat(1, width)
    m = M()
    k = compile_export(torch.export.export(m, (x,)), schedule=Schedule(64, "warp", False, 4))
    torch.testing.assert_close(
        k(x), m(x), rtol=0.0015 if dtype != torch.float32 else 2e-5, atol=1e-7
    )


def test_nonfinite_pointwise_masks():
    class M(nn.Module):
        def forward(self, x, y):
            return torch.nn.functional.silu(x) * y, x.relu(), x.sigmoid()

    x = torch.tensor(
        [[float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 90.0, -90.0]], device="cuda"
    )
    y = torch.ones_like(x)
    m = M()
    ref = m(x, y)
    k = compile_export(torch.export.export(m, (x, y)), schedule=Schedule(64, "flat", False, 4))
    out = k(x, y)
    for a, b in zip(out, ref):
        assert torch.equal(a.isnan(), b.isnan())
        assert torch.equal(a.isposinf(), b.isposinf())
        assert torch.equal(a.isneginf(), b.isneginf())
        torch.testing.assert_close(a, b, equal_nan=True)


def test_nonfinite_report_is_strict_json():
    import json
    from sparkfuse.measure import compare

    ref = torch.tensor([torch.nan, torch.inf, -torch.inf, 1.0], device="cuda")
    item = compare(ref, ref)[0]
    assert (
        item["matched_nan"]
        == item["matched_posinf"]
        == item["matched_neginf"]
        == item["finite_pairs"]
        == 1
    )
    assert item["unequal"] == item["max_abs"] == 0
    all_bad = compare(ref[:3], ref[:3])[0]
    assert all_bad["max_abs"] is None and all_bad["mean_abs"] is None
    json.dumps([item, all_bad], allow_nan=False)
    with pytest.raises(AssertionError):
        compare(ref, ref.nan_to_num())
