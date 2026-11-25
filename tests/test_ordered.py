import pytest
import torch
from torch import nn
from sparkfuse import Schedule, Unsupported, compile_export, lower


class Score(nn.Module):
    def forward(self, x):
        y = x / 8.0
        return y.softmax(-1), y


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width,cache", [(32, False), (33, True), (97, False), (257, True)])
def test_ordered_softmax_both_outputs(dtype, width, cache):
    generator = torch.Generator(device="cuda").manual_seed(12012)
    m = Score()
    x = torch.randn(17, width, device="cuda", dtype=dtype, generator=generator)
    k = compile_export(torch.export.export(m, (x,)), schedule=Schedule(128, "warp_xor", cache))
    for shift in (0, 1):
        y = torch.randn(17 * width + shift, device="cuda", dtype=dtype, generator=generator)[
            shift:
        ].view(17, width)
        for result, expected in zip(k(y), m(y)):
            torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("width,vec", [(31, 1), (1025, 1), (97, 2)])
def test_ordered_illegal(width, vec):
    ep = torch.export.export(Score(), (torch.randn(2, width),))
    p = lower(ep)
    with pytest.raises(Unsupported, match="warp_xor"):
        Schedule(128, "warp_xor", True, vec).validate(p)


class Center(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h

    def forward(self, x, y, w, b):
        residual = x + y
        return torch.nn.functional.layer_norm(residual, (self.h,), w, b, 1.0e-12), residual


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("width,cache", [(64, False), (768, True), (1536, False)])
def test_online_moments(dtype, width, cache):
    gen = torch.Generator(device="cuda").manual_seed(5142)
    m = Center(width)

    def make(off):
        return tuple(
            torch.randn(n + off, device="cuda", dtype=dtype, generator=gen)[off:].view(shape)
            for n, shape in (
                (11 * width, (11, width)),
                (11 * width, (11, width)),
                (width, (width,)),
                (width, (width,)),
            )
        )

    args = make(0)
    k = compile_export(
        torch.export.export(m, args), schedule=Schedule(128, "block_welford", cache, 4)
    )
    for a in [args, make(1)]:
        from sparkfuse.measure import compare

        compare(k(*a), m(*a), exact=(1,))


class SeparateAffine(nn.Module):
    def forward(self, x, w, b):
        prod = x * w
        return prod + b, prod


def test_explicit_mul_add_remains_separately_rounded():
    # Exact product is 1 - 2**-46. A contraction would return a nonzero result.
    a = tuple(
        torch.full((3, 64), v, device="cuda", dtype=torch.float32)
        for v in (1 + 2**-23, 1 - 2**-23, -1)
    )
    m = SeparateAffine()
    ep = torch.export.export(m, a)
    assert not any(v.op == "layer_affine" for v in lower(ep).vals)
    out, prod = compile_export(ep, schedule=Schedule(128, "flat"))(*a)
    assert torch.equal(out, torch.zeros_like(out)) and torch.equal(prod, torch.ones_like(prod))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_aligned_vectorized_layernorm_affine(dtype):
    g = torch.Generator(device="cuda").manual_seed(841813)
    h = 768
    a = tuple(
        torch.randn(shape, dtype=dtype, device="cuda", generator=g)
        for shape in [(33, h), (33, h), (h,), (h,)]
    )
    m = Center(h)
    ep = torch.export.export(m, a)
    assert sum(v.op == "layer_affine" for v in lower(ep).vals) == 1
    out, res = compile_export(ep, schedule=Schedule(128, "block_welford", False, 4))(*a)
    ref, _, _ = torch.ops.aten.native_layer_norm.default(a[0] + a[1], [h], a[2], a[3], 1e-12)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert torch.equal(res, a[0] + a[1])
