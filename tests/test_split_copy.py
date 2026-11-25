from math import prod

import pytest
import torch
from torch import nn
from sparkfuse.partition import partition_export


class Split(nn.Module):
    def __init__(self, widths, visible=False):
        super().__init__()
        self.widths = widths
        self.visible = visible

    def forward(self, x):
        a, b, c = x.split(self.widths, -1)
        out = tuple(v.clone(memory_format=torch.contiguous_format) for v in (a, b, c))
        return (out, (b, c, a), x) if self.visible else out


@pytest.mark.parametrize("host", ["native", "python"])
@pytest.mark.parametrize(
    "shape,widths,dt",
    [
        ((2, 1, 64), (40, 16, 8), torch.bfloat16),
        ((5, 23), (5, 11, 7), torch.float16),
        ((7, 2, 24), (12, 8, 4), torch.float32),
        ((24,), (8, 8, 8), torch.bfloat16),
    ],
)
@torch.inference_mode()
def test_copies_and_visible_aliases(host, shape, widths, dt):
    m = Split(widths, True)
    x = torch.empty(shape, dtype=dt, device="cuda")
    ep = torch.export.export(m, (x,))
    gm, rr = partition_export(ep, split_copies=True, host_backend=host)
    (r,) = [r for r in rr if r.get("kind") == "split_copy3"]
    assert r["widths"] == list(widths) and r["schedule"]["max_packet_elements"] >= 1
    it = torch.int32 if dt == torch.float32 else torch.int16
    for off in (0, 1, 2, 4):
        # Raw bit patterns include signed zero, infinities and NaN payloads; no float arithmetic is allowed.
        raw = torch.randint(
            torch.iinfo(it).min, torch.iinfo(it).max, (prod(shape) + off,), device="cuda", dtype=it
        )
        raw[off : off + 6] = torch.tensor(
            [0, -1, torch.iinfo(it).min, torch.iinfo(it).max, 1, -2], device="cuda", dtype=it
        )
        x = raw[off:].view(dt).view(shape)
        before = x.view(it).clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            out, views, orig = gm(x)
            ref, rv, _ = m(x)
        torch.cuda.current_stream().wait_stream(stream)

        def ptr(a):
            return a.untyped_storage().data_ptr()

        assert (
            ptr(orig) == ptr(x)
            and all(ptr(v) == ptr(x) for v in views)
            and len({ptr(t) for t in (*out, x)}) == 4
        )
        for a, b in zip((*out, *views), (*ref, *rv), strict=True):
            assert (a.shape, a.stride(), a.storage_offset()) == (
                b.shape,
                b.stride(),
                b.storage_offset(),
            ) and torch.equal(a.view(it), b.view(it))
        out[0].zero_()
        assert torch.equal(x.view(it), before)


@pytest.mark.parametrize(
    "reason", ["wrong_axis", "four", "missing", "preserve", "duplicate", "strided", "off"]
)
@torch.inference_mode()
def test_decline_unproven_copies(reason):
    class M(nn.Module):
        def forward(self, x):
            z = x.split(
                [2, 2, 2, 2] if reason == "four" else [2, 4, 2], 0 if reason == "wrong_axis" else -1
            )
            ret = tuple(
                p.clone(
                    memory_format=torch.preserve_format
                    if reason == "preserve"
                    else torch.contiguous_format
                )
                for p in z[: 2 if reason == "missing" else len(z)]
            )
            return (
                (*ret, z[0].clone(memory_format=torch.contiguous_format))
                if reason == "duplicate"
                else ret
            )

    x = torch.randn(8, 8, device="cuda")
    x = x.t() if reason == "strided" else x
    m = M()
    g, rr = partition_export(torch.export.export(m, (x,)), split_copies=reason != "off")
    assert not any(r.get("kind") == "split_copy3" for r in rr)
    for a, b in zip(g(x), m(x), strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_all_halfword_payloads_early_consumer_and_replay(dt):
    class M(nn.Module):
        def forward(self, x):
            first, second, third = x.split((128, 96, 32), -1)
            c0 = first.clone(memory_format=torch.contiguous_format)
            # An observable use of the first clone precedes materialization of either sibling.
            tail = c0[..., 3:18]
            c1 = second.clone(memory_format=torch.contiguous_format)
            c2 = third.clone(memory_format=torch.contiguous_format)
            return tail, c2, c1, c0

    raw = (
        torch.arange(-32768, 32768, dtype=torch.int32, device="cuda")
        .to(torch.int16)
        .reshape(256, 256)
    )
    x = raw.view(dt)
    m = M()
    g, rr = partition_export(torch.export.export(m, (x,)), split_copies=True, host_backend="native")
    assert sum(r.get("kind") == "split_copy3" for r in rr) == 1
    warm = g(x)
    del warm
    cg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cg):
        saved = g(x)
    for v in (raw.clone(), raw.flip([0])):
        raw.copy_(v)
        cg.replay()
        ref = m(x)
        for a, b in zip(saved, ref, strict=True):
            assert (
                a.storage_offset() == b.storage_offset()
                and a.stride() == b.stride()
                and torch.equal(a.view(torch.int16), b.view(torch.int16))
            )
        pts = [t.untyped_storage().data_ptr() for t in saved]
        assert pts[0] == pts[-1] and len(set([*pts, x.untyped_storage().data_ptr()])) == 4
