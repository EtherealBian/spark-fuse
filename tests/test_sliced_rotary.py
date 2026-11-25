"""The projected V and the Q/K input views stay aliases; a partial head's suffix is copied as bits."""

from math import prod
import pytest
import torch
from torch import nn
from sparkfuse.partition import partition_export


class SlicePair(nn.Module):
    def __init__(self, d=13, r=8, mode="aliases", off=3):
        super().__init__()
        self.d = d
        self.r = r
        self.off = off
        self.mode = mode

    def forward(self, x, c, s):
        d = self.d
        r = self.r
        i = self.off
        j = i + 3 * d + 1
        a = x[..., i : i + 3 * d]
        b = x[
            ...,
            (i + d if self.mode == "overlap" else j) : (
                i + 3 * d if self.mode == "overlap" else j + 2 * d
            ),
        ]
        v = x[..., j + 2 * d :]
        q, k = (z.view(*x.shape[:2], -1, d).transpose(1, 2) for z in (a, b))
        c, s = c.unsqueeze(1), s.unsqueeze(1)

        def one(z, c0):
            fr, tail = z[..., :r], z[..., r:]
            lo = fr[..., : r // 2]
            hi = -fr[..., r // 2 :]
            rot = torch.cat((lo, hi) if self.mode == "reverse" else (hi, lo), -1)
            y = fr * c0 + rot * s
            suffix = z[..., d - r :] if self.mode == "wrong_suffix" else tail
            return torch.cat((y, suffix), -1), fr, tail, y

        rq, front, tail, partial = one(q, c)
        rk, _, _, _ = one(k, c.clone() if self.mode == "tables" else c)
        if self.mode == "partial":
            return rq, rk, v, partial
        if self.mode == "aliases":
            return rq, rk, v, a, k, front, tail, x
        return rq, rk, v


def it(t):
    return torch.int32 if t.dtype == torch.float32 else torch.int16


def data(dt, seq, d, r, shared, seed, off=3, payload=True):
    sizes = (2, seq, off + 5 * d + 1 + 9)
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def make(shape, offset):
        z = torch.randn(prod(shape) + offset, device="cuda", dtype=dt, generator=gen)
        z[::7] = 0.0
        z[::13] = -0.0
        return z[offset:].view(shape)

    x = make(sizes, 1)
    c, s = (make((1 if shared else 2, seq, r), i) for i in (3, 5))
    if payload:
        for h, start in [(3, off), (2, off + 3 * d + 1)]:
            z = x[..., start : start + h * d].view(2, seq, h, d)[..., r:]
            raw = z.view(it(x))
            raw.copy_(
                torch.randint(
                    torch.iinfo(it(x)).min,
                    torch.iinfo(it(x)).max,
                    raw.shape,
                    dtype=it(x),
                    generator=gen,
                    device="cuda",
                )
            )
            raw[..., ::3] = -1
        x[..., -9:].view(it(x)).fill_(-1)
    return x, c, s


def check(out, ref):
    for a, b in zip(out, ref, strict=True):
        assert (
            a.shape == b.shape
            and a.dtype == b.dtype
            and a.stride() == b.stride()
            and a.storage_offset() == b.storage_offset()
        )
        assert torch.equal(a.contiguous().view(it(a)), b.contiguous().view(it(b)))


def ptr(t):
    return t.untyped_storage().data_ptr()


@pytest.mark.parametrize("host", ["python", "native"])
@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("seq,d,r,shared", [(1, 14, 14, False), (7, 13, 8, True), (3, 9, 2, False)])
@torch.inference_mode()
def test_aliases_rounding_and_suffix(host, dt, seq, d, r, shared):
    m = SlicePair(d, r)
    args = data(dt, seq, d, r, shared, 4637)
    g, rs = partition_export(torch.export.export(m, args), sliced_rotary=True, host_backend=host)
    (hit,) = [z for z in rs if z.get("kind") == "sliced_rotary_pair"]
    g.graph.lint()
    assert (
        hit["rotary_dim"] == r
        and hit["head_dim"] == d
        and hit["offsets"] == [3, 3 + 3 * d + 1]
        and hit["shared_phase"] == shared
    )
    aa = data(dt, seq, d, r, shared, 3119)
    before = aa[0].view(it(aa[0])).clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out = g(*aa)
        ref = m(*aa)
    torch.cuda.current_stream().wait_stream(stream)
    check(out, ref)
    assert len({ptr(z) for z in (*aa, *out[:2])}) == 5 and all(
        ptr(z) == ptr(aa[0]) for z in out[2:]
    )
    out[0].zero_()
    out[1].zero_()
    assert torch.equal(aa[0].view(it(aa[0])), before)


@pytest.mark.parametrize(
    "reason",
    [
        "overlap",
        "wrong_suffix",
        "tables",
        "partial",
        "reverse",
        "input_stride",
        "phase_stride",
        "odd",
        "off",
    ],
)
@torch.inference_mode()
def test_fallback(reason):
    d, r = 13, 7 if reason == "odd" else 8
    m = SlicePair(d, r, reason)
    aa = data(torch.bfloat16, 7, d, r, False, 7181, payload=False)
    if reason == "input_stride":
        aa = (aa[0].repeat_interleave(2, -1)[..., ::2], *aa[1:])
    if reason == "phase_stride":
        aa = (*aa[:1], *(z.repeat_interleave(2, -1)[..., ::2] for z in aa[1:]))
    g, rs = partition_export(
        torch.export.export(m, aa), sliced_rotary=reason != "off", host_backend="native"
    )
    assert not any(z.get("kind") == "sliced_rotary_pair" for z in rs)
    g.graph.lint()
    check(g(*aa), m(*aa))


@pytest.mark.parametrize("host", ["python", "native"])
@torch.inference_mode()
def test_two_pairs_and_graph_replay(host):
    class Twice(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = SlicePair(mode="normal")

        def forward(self, x, y, c, s):
            return (*self.f(x, c, s), *self.f(y, c, s))

    fn = Twice()

    def gen(seed):
        return data(torch.bfloat16, 5, 13, 8, False, seed)

    x, c, s = gen(7717)
    y, *_ = gen(9011)
    aa = (x, y, c, s)
    g, rs = partition_export(torch.export.export(fn, aa), sliced_rotary=True, host_backend=host)
    assert sum(z.get("kind") == "sliced_rotary_pair" for z in rs) == 2
    g.graph.lint()
    g(*aa)
    cg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cg):
        out = g(*aa)
    for seed in [41, 9367]:
        xx, cc, ss = gen(seed)
        yy, *_ = gen(seed + 1)
        for z, v in zip(aa, (xx, yy, cc, ss), strict=True):
            z.copy_(v)
        cg.replay()
        check(out, fn(*aa))
        assert ptr(out[2]) == ptr(x) and ptr(out[5]) == ptr(y)


@pytest.mark.parametrize("host", ["python", "native"])
def test_runtime_guards(host):
    aa = data(torch.bfloat16, 3, 13, 8, True, 8707)
    f = SlicePair()
    g, rs = partition_export(torch.export.export(f, aa), sliced_rotary=True, host_backend=host)
    (r,) = [z for z in rs if z.get("kind") == "sliced_rotary_pair"]
    run = g.get_submodule(r["module"])
    x, c, s = aa
    c, s = c.unsqueeze(1), s.unsqueeze(1)
    for bad in [
        x.cpu(),
        x.float(),
        x[..., :-1],
        x.repeat_interleave(2, -1)[..., ::2],
        torch._neg_view(x),
    ]:
        with pytest.raises(ValueError):
            run(bad, c, s)
    with pytest.raises(ValueError, match="inference only"):
        run(x.requires_grad_(), c, s)


@pytest.mark.parametrize("host", ["python", "native"])
@torch.inference_mode()
def test_equal_heads_late_phase_early_consumer(host):
    # Lower by data dependencies, even when FX records the upper slice first,
    # both branches have identical shapes, and an output has an early user.
    class Reordered(nn.Module):
        def forward(self, x, c, s):
            c = (c * s).unsqueeze(1)
            s = s.unsqueeze(1)

            def branch(start):
                h = x[..., start : start + 26].view(2, 5, 2, 13).transpose(1, 2)
                f = h[..., :8]
                rot = torch.cat((-f[..., 4:], f[..., :4]), -1)
                return torch.cat((f * c + rot * s, h[..., 8:]), -1)

            top = branch(44)
            early = top + top
            bottom = branch(3)
            return early, bottom, top, x[..., 20:50]

    fn = Reordered()
    aa = data(torch.bfloat16, 5, 13, 8, False, 4777, payload=False)
    g, rs = partition_export(torch.export.export(fn, aa), sliced_rotary=True, host_backend=host)
    (hit,) = [r for r in rs if r.get("kind") == "sliced_rotary_pair"]
    assert hit["heads"] == [2, 2] and hit["offsets"] == [44, 3]
    g.graph.lint()
    for seed in (9001, 3313):
        args = data(torch.bfloat16, 5, 13, 8, False, seed, payload=False)
        y = g(*args)
        check(y, fn(*args))
        assert ptr(y[-1]) == ptr(args[0]) and len({ptr(z) for z in y}) == 4
