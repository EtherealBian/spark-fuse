import pytest
import torch
from torch import nn
from torch.nn import functional as F
from sparkfuse import Schedule, Unsupported, compile_export, lower
from sparkfuse.measure import compare
from sparkfuse.partition import partition_export
from sparkfuse.dynamic import DynamicRows


class ResidualLN(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.width = width
        self.eps = eps

    def forward(self, x, skip, w, b):
        z = x + skip
        return F.layer_norm(z, (self.width,), w, b, self.eps), z


@pytest.mark.parametrize("dt", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("s", [Schedule(128, "block", True, 4), Schedule(64, "warp", False, 1)])
@pytest.mark.parametrize("width", [1, 97, 768])
def test_residual_layernorm(dt, s, width):
    torch.manual_seed(273)
    a = tuple(
        torch.randn(sh, device="cuda", dtype=dt)
        for sh in ((2, 7, width), (2, 7, width), (width,), (width,))
    )
    a = tuple(torch.cat((x.new_empty(1), x.flatten()))[1:].view_as(x) for x in a)
    m = ResidualLN(width)
    k = compile_export(torch.export.export(m, a), schedule=s)
    compare(k(*a), m(*a), exact=(1,))


@pytest.mark.parametrize("weight,bias", [(True, False), (False, True), (False, False)])
def test_affine_optional_and_constant_rows(weight, bias):
    class M(nn.Module):
        def forward(self, x, w, b):
            return F.layer_norm(x, (17,), w if weight else None, b if bias else None)

    x = torch.full((3, 17), 4096.0, device="cuda")
    w = torch.linspace(0.1, 2, 17, device="cuda")
    b = w + 3
    a = (x, w, b)
    m = M()
    k = compile_export(torch.export.export(m, a))
    compare(k(*a), m(*a), exact=(0,))


def test_centered_variance_nonfinite_and_small_variance():
    class M(nn.Module):
        def forward(self, x):
            return F.layer_norm(x, (32,))

    x = torch.tensor(
        [[4096.25, 4095.75], [8.0, 8.0], [float("inf"), 1.0], [1.0, float("nan")]], device="cuda"
    ).repeat(1, 16)
    k = compile_export(torch.export.export(M(), (x,)))
    y = k(x)
    compare(y, M()(x))
    torch.testing.assert_close(
        y[:2].double(), F.layer_norm(x[:2].double(), (32,)), rtol=2e-6, atol=1e-7
    )


def test_dynamic_norm_and_native_partition():
    x, y = (torch.randn(3, 97, device="cuda") for _ in range(2))
    w, b = (torch.randn(97, device="cuda") for _ in range(2))
    a = (x, y, w, b)
    m = ResidualLN(97)
    sym = torch.export.Dim("tokens", min=1, max=47)
    dynamic = DynamicRows(
        torch.export.export(m, a, dynamic_shapes=({0: sym}, {0: sym}, None, None)),
        schedule=Schedule(64, "warp", True, 4),
    )
    for n in (1, 7, 47):
        a2 = (x[:1].repeat(n, 1), y[:1].repeat(n, 1), w, b)
        compare(dynamic(*a2), m(*a2), exact=(1,))
    mixed, r = partition_export(torch.export.export(m, a), host_backend="native")
    assert sum("module" in s for s in r) == 1
    compare(mixed(*a), m(*a), exact=(1,))


def test_multi_axis_rejected():
    class M(nn.Module):
        def forward(self, x):
            return F.layer_norm(x, (3, 7))

    with pytest.raises(Unsupported, match="last-axis"):
        lower(torch.export.export(M(), (torch.zeros(2, 3, 7),)))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_affine_pointer_offsets(dtype):
    m = ResidualLN(768)
    torch.manual_seed(617)

    def inp(offset):
        result = []
        for i, (shape, typ) in enumerate(
            [
                ((14, 768), dtype),
                ((14, 768), dtype),
                ((768,), dtype),
                ((768,), dtype),
            ]
        ):
            n = (offset + i) % 4
            x = torch.randn(torch.Size(shape).numel() + n, device="cuda", dtype=typ)
            result.append(x[n:].view(shape))
        return tuple(result)

    a = inp(0)
    k = compile_export(torch.export.export(m, a), schedule=Schedule(128, "block", True, 4))
    for i in (0, 1, 2, 3, 0):
        a = inp(i)
        compare(k(*a), m(*a), exact=(1,))


@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16])
def test_mixed_affine_guard(dt):
    m = ResidualLN(768)
    x = torch.zeros(14, 768, dtype=dt, device="cuda")
    w = torch.ones(768, device="cuda")
    with pytest.raises(Unsupported, match="mixed-dtype"):
        lower(torch.export.export(m, (x, x, w, w)))
