"""Value visibility is a legality condition, not a calibration-time property."""

import pytest
import torch
from sparkfuse import Schedule, Unsupported, compile_export, lower
from sparkfuse.native import native_dispatch
from sparkfuse.partition import partition_export
from sparkfuse.score_rounding import only_bf16_probability


class Score(torch.nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs

    def forward(self, x, mask):
        scale = x * 0.08838834764831845
        z = scale + mask
        p = z.softmax(-1, dtype=torch.float32)
        if self.outputs == "cast":
            return p.bfloat16(), z
        if self.outputs == "after":
            return p.bfloat16().float() * 2, z
        if self.outputs == "raw":
            return p.bfloat16(), p, z
        if self.outputs == "half":
            return p.bfloat16(), p.half(), z
        if self.outputs == "before":
            return (p * 2).bfloat16(), z
        raise AssertionError(self.outputs)


def make(width, seed, off, nonfinite):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def arr(r, o):
        return torch.randn(r * width + o, device="cuda", dtype=torch.bfloat16, generator=g)[
            o:
        ].view(r, width)

    x = arr(25, off)
    m = arr(1, 3 * off)
    x.mul_(11)
    m[:, ::113] = -float("inf")
    x[6, ::13] = torch.finfo(torch.bfloat16).min
    if nonfinite:
        x[0].fill_(float("-inf"))
        x[1, 0] = float("inf")
        x[2, 3] = float("nan")
    return x, m


def assert_bits(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype and a.stride() == b.stride()
    assert torch.equal(a.isnan(), b.isnan())
    valid = ~b.isnan()
    it = torch.int32 if a.dtype == torch.float32 else torch.int16
    assert torch.equal(a.view(it)[valid], b.view(it)[valid])


@pytest.mark.parametrize("outputs", ["cast", "after", "raw", "half", "before"])
@torch.inference_mode()
def test_rounding_visibility(outputs):
    m = Score(outputs)
    a = make(1537, 85259, 0, False)
    ep = torch.export.export(m, a)
    plan = lower(ep)
    allow = outputs in ("cast", "after")
    assert only_bf16_probability(plan) == allow
    if not allow:
        with pytest.raises(Unsupported, match="exclusively immediate BF16"):
            Schedule(128, "warp_bf16_guard", True).validate(plan)
    gm, records = partition_export(ep, ordered_scores="guarded")
    (rec,) = [r for r in records if r.get("module")]
    assert rec["schedule"]["layout"] == ("warp_bf16_guard" if allow else "warp_slots")
    for u, v in zip(gm(*a), m(*a), strict=True):
        assert_bits(u, v)


@pytest.mark.parametrize("width,threads", [(1025, 32), (1536, 64), (1537, 128), (2048, 256)])
@torch.inference_mode()
def test_guarded_stream_offsets_and_nonfinite(width, threads):
    model = Score("cast")
    ep = torch.export.export(model, make(width, 85427, 0, False))
    k = compile_export(ep, schedule=Schedule(threads, "warp_bf16_guard", True))
    f = native_dispatch(k)
    for off, fn in [(0, k), (1, f), (3, f)]:
        a = make(width, 85457 + off, off, True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            actual = fn(*a)
            ref = model(*a)
        torch.cuda.current_stream().wait_stream(stream)
        for u, v in zip(actual, ref, strict=True):
            assert_bits(u, v)
