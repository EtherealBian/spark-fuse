import pytest
import torch
from torch import nn
from sparkfuse import Schedule, compile_export, lower, Unsupported
from sparkfuse.dynamic import DynamicRows
from sparkfuse.measure import compare
from sparkfuse.partition import partition_export
from examples.graphs import ScoreSoftmax


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "layout,cache", [("block", False), ("warp", False), ("block", True), ("warp", True)]
)
@pytest.mark.parametrize("width", [1, 97, 769])
def test_scores(dtype, layout, cache, width):
    torch.manual_seed(405)
    m = ScoreSoftmax(True)
    x = torch.randn(13 * width + 1, device="cuda", dtype=dtype)[1:].view(13, width)
    b = torch.randn(width + 1, device="cuda", dtype=dtype)[1:]
    ep = torch.export.export(m, (x, b))
    k = compile_export(ep, schedule=Schedule(128, layout, cache))
    prob, logits = k(x, b)
    ref, middle = m(x, b)
    torch.testing.assert_close(logits, middle, rtol=0, atol=0)
    # Stronger than a generic absolute tolerance for near-zero probabilities.
    torch.testing.assert_close(
        prob,
        ref,
        rtol={torch.float32: 4e-6, torch.float16: 0.001, torch.bfloat16: 0.008}[dtype],
        atol=0,
    )
    torch.testing.assert_close(
        prob.float().sum(-1),
        torch.ones(13, device="cuda"),
        rtol={torch.float32: 2e-6, torch.float16: 0.0005, torch.bfloat16: 0.004}[dtype],
        atol=0,
    )


@pytest.mark.parametrize("cache", [False, True])
def test_math_edges_and_epilogue(cache):
    class M(nn.Module):
        def forward(self, x, bias):
            p = (x + bias).softmax(-1)
            return p * 2.0, p, -bias

    x = torch.tensor(
        [
            [1000.0, 999.0, 998.0, -1000.0],
            [0.0, -float("inf"), 2.0, -float("inf")],
            [-float("inf")] * 4,
            [1.0, 2.0, 3.0, float("inf")],
            [1.0, 2.0, float("nan"), 4.0],
        ],
        device="cuda",
    )
    b = torch.zeros(5, 1, device="cuda")
    ep = torch.export.export(M(), (x, b))
    k = compile_export(ep, schedule=Schedule(64, "block", cache))
    out = k(x, b)
    ref = M()(x, b)
    for u, v in zip(out, ref):
        torch.testing.assert_close(u, v, equal_nan=True, rtol=3e-6, atol=0)
    assert out[1][2:].isnan().all() and out[1][1, 1] == 0


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_explicit_dtype(out_dtype):
    class M(nn.Module):
        def forward(self, x):
            return torch.softmax(x, -1, dtype=out_dtype)

    # Casting before the collective is observable at this offset.
    x = torch.tensor([[3000.1, 3000.9, 2997.2]], device="cuda")
    m = M()
    k = compile_export(torch.export.export(m, (x,)))
    torch.testing.assert_close(k(x), m(x), atol=0, rtol=2e-6)


def test_aten_internal_half_to_float():
    class M(nn.Module):
        def forward(self, x):
            return torch.ops.aten._softmax.default(x, -1, True)

    x = torch.randn(3, 7, 49, device="cuda", dtype=torch.float16)
    m = M()
    k = compile_export(torch.export.export(m, (x,)))
    torch.testing.assert_close(k(x), m(x), atol=0, rtol=4e-6)


def test_guarded_rows_single_binary_and_native_partition():
    x = torch.randn(3, 97, device="cuda")
    b = torch.randn(97, device="cuda")
    m = ScoreSoftmax(True)
    dim = torch.export.Dim("tokens", min=1, max=97)
    ep = torch.export.export(m, (x, b), dynamic_shapes=({0: dim}, None))
    d = DynamicRows(ep, schedule=Schedule(128, "warp", True))
    for rows in (1, 3, 47, 97):
        t = x[0].repeat(rows, 1)
        compare(d(t, b), m(t, b), exact=(1,))
    static = torch.export.export(m, (x, b))
    gm, decisions = partition_export(static, host_backend="native")
    assert sum("module" in r for r in decisions) == 1
    compare(gm(x, b), m(x, b), exact=(1,))


def test_reject_nonfinal_axis_and_nested_collectives():
    class M(nn.Module):
        def __init__(self, axis):
            super().__init__()
            self.axis = axis

        def forward(self, x):
            return torch.softmax(x, self.axis)

    class N(nn.Module):
        def forward(self, x):
            return x.softmax(-1).mean(-1, keepdim=True)

    x = torch.randn(3, 7, device="cuda")
    for m in (M(0), N()):
        with pytest.raises(Unsupported):
            lower(torch.export.export(m, (x,)))


@pytest.mark.parametrize(
    "op,sch",
    [
        ("scores", Schedule(128, "warp", True)),
        ("gate", Schedule(128, "flat", False, 4)),
        ("mean", Schedule(128, "block", True, 4)),
    ],
)
def test_broadcast_coordinates(op, sch):
    class M(nn.Module):
        def forward(self, x, head_mask, head_scale, position):
            z = x * head_scale + head_mask + position
            if op == "mean":
                return z.square().mean(-1, keepdim=True), z
            if op == "scores":
                return z.softmax(-1), z
            return torch.nn.functional.silu(z) * z, z

    torch.manual_seed(19)
    shapes = [(2, 3, 5, 97), (2, 1, 5, 97), (1, 3, 1, 1), (5, 1)]
    args = tuple(torch.randn(sh, device="cuda") for sh in shapes)
    m = M()
    k = compile_export(torch.export.export(m, args), schedule=sch)
    compare(k(*args), m(*args), exact=(1,))


def test_mixed_cast_boundary():
    class M(nn.Module):
        def forward(self, x):
            return torch.softmax(x * 0.125, -1, dtype=torch.float32).to(x.dtype)

    x = torch.randn(3, 97, device="cuda", dtype=torch.bfloat16)
    m = M()
    gm, regions = partition_export(torch.export.export(m, (x,)), host_backend="native")
    assert sum("module" in d for d in regions) == 1
    compare(gm(x), m(x))


@pytest.mark.parametrize("width", [1, 3, 97, 128, 769])
@pytest.mark.parametrize("cache,vec,layout", [(True, 8, "warp"), (False, 4, "block")])
def test_packed_softmax_offset_and_nonfinite(width, cache, vec, layout):
    class M(nn.Module):
        def forward(self, x, bias):
            z = x + bias
            p = torch.softmax(z, dim=-1, dtype=torch.float32).to(x.dtype)
            return p * 0.5, z

    torch.manual_seed(782)
    args = tuple(
        torch.randn(torch.Size(d).numel() + 1, device="cuda", dtype=torch.bfloat16)[1:].view(d)
        for d in ((2, 3, 5, width), (2, 1, 5, width))
    )
    args[1][:, :, 1, :] = -torch.inf
    args[0][0, :, 2, 0] = torch.nan
    m = M()
    k = compile_export(torch.export.export(m, args), schedule=Schedule(128, layout, cache, vec))
    out = k(*args)
    ref = m(*args)
    for i, (a, b) in enumerate(zip(out, ref)):
        torch.testing.assert_close(a, b, rtol=0 if i == 1 else 0.008, atol=0, equal_nan=True)
