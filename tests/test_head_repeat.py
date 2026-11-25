"""Legality/liveness and raw-bit semantics for horizontal head repeat fusion."""

from math import prod
import pytest
import torch
from sparkfuse.partition import partition_export


class Repeats(torch.nn.Module):
    def __init__(self, g=3, mode="normal"):
        super().__init__()
        self.g = g
        self.mode = mode

    def forward(self, a, b):
        bs, h, s, d = a.shape
        g = self.g
        ua = a.unsqueeze(2)
        ea = ua.expand(bs, h, g, s, d)
        ra = ea.reshape(bs, h * g, s, d)
        if self.mode == "dependent":
            b = b + ra[:, ::g]
        if self.mode == "swapped":
            eb = b.unsqueeze(1).expand(bs, g, h, s, d)
        else:
            eb = b.unsqueeze(-3).expand(bs, h, g, s, d)
        rb = eb.reshape(bs, h * g, s, d)
        if self.mode == "visible":
            return ra, rb, ea, eb, ua, ra[..., 1:], rb[..., 1:]
        return ra, rb


def inputs(dt, shape, off):
    it = torch.int32 if dt == torch.float32 else torch.int16
    gen = torch.Generator(device="cuda").manual_seed(64991)

    def one(off):
        ii = torch.iinfo(it)
        a = torch.randint(
            ii.min, ii.max, (prod(shape) + off,), generator=gen, device="cuda", dtype=it
        )[off:]
        a[::17] = ii.min
        a[2::13] = 0x7FC00001 if it == torch.int32 else (0x7FC1 if dt == torch.bfloat16 else 0x7E01)
        return a.view(dt).reshape(shape)

    return tuple(one(n) for n in off)


def strict(x, y):
    it = torch.int32 if x.dtype == torch.float32 else torch.int16
    return (
        x.shape == y.shape
        and x.stride() == y.stride()
        and torch.equal(x.contiguous().view(it), y.contiguous().view(it))
    )


def addr(a):
    return a.untyped_storage().data_ptr()


@pytest.mark.parametrize("host", ["native", "python"])
@pytest.mark.parametrize(
    "dt,sh,g",
    [
        (torch.bfloat16, (2, 2, 32, 128), 6),
        (torch.float32, (1, 3, 5, 9), 2),
        (torch.float16, (2, 3, 1, 14), 5),
    ],
)
@torch.inference_mode()
def test_bits_views_and_expansion(host, dt, sh, g):
    m = Repeats(g, "visible")
    args = inputs(dt, sh, (0, 0))
    ep = torch.export.export(m, args)
    gm, rr = partition_export(ep, paired_head_repeats=True, host_backend=host)
    gm.graph.lint()
    (r,) = [r for r in rr if r.get("kind") == "paired_head_repeat"]
    assert r["logical_shape"] == [sh[0], sh[1], g, *sh[2:]]
    for off in [(0, 0), (1, 3), (2, 2), (4, 4)]:
        a, b = inputs(dt, sh, off)
        for inp in [(a, b), (a, a)]:
            before = tuple(x.clone() for x in inp)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                got = gm(*inp)
                ref = m(*inp)
            torch.cuda.current_stream().wait_stream(stream)
            assert all(strict(x, y) for x, y in zip(got, ref, strict=True)) and all(
                strict(x, y) for x, y in zip(inp, before, strict=True)
            )
            for i, x in enumerate((*inp, *got)):
                for j, y in enumerate((*inp, *got)):
                    ori = (*inp, *ref)
                    assert (addr(x) == addr(y)) == (addr(ori[i]) == addr(ori[j]))
    _, base = partition_export(ep)
    assert not any(r.get("kind") == "paired_head_repeat" for r in base)


@pytest.mark.parametrize(
    "sh,g,mode,layout",
    [
        ((2, 1, 5, 7), 3, "normal", "normal"),
        ((2, 2, 5, 7), 1, "normal", "normal"),
        ((2, 2, 5, 7), 3, "swapped", "normal"),
        ((2, 2, 5, 7), 3, "dependent", "normal"),
        ((2, 2, 5, 7), 3, "normal", "strided"),
        ((2, 2, 5, 7), 33, "normal", "normal"),
    ],
)
@torch.inference_mode()
def test_head_repeat_fallback(sh, g, mode, layout):
    a, b = inputs(torch.float32, sh, (0, 0))
    args = (a, b.transpose(2, 3).contiguous().transpose(2, 3)) if layout == "strided" else (a, b)
    m = Repeats(g, mode)
    gm, rr = partition_export(torch.export.export(m, args), paired_head_repeats=True)
    assert not any(r.get("kind") == "paired_head_repeat" for r in rr)


@pytest.mark.parametrize("host", ["native", "python"])
def test_head_repeat_launch_guard(host):
    a, b = inputs(torch.float16, (2, 2, 9, 12), (0, 0))
    gm, rr = partition_export(
        torch.export.export(Repeats(), (a, b)), paired_head_repeats=True, host_backend=host
    )
    pg = gm.get_submodule(next(x["module"] for x in rr if x.get("kind") == "paired_head_repeat"))
    for x in (
        a.float(),
        a.cpu(),
        a.clone().requires_grad_(True),
        a[:, :, :-1],
        a.transpose(1, 2).contiguous().transpose(1, 2),
    ):
        with pytest.raises((ValueError, RuntimeError)):
            pg(x, b)
