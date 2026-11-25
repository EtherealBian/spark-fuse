"""Wide ordered CUDA lanes: raw finite values, nonfinite rows, offsets and framework fallbacks."""

import pytest
import torch
from sparkfuse import Schedule, Unsupported, compile_export, lower
from sparkfuse.native import native_dispatch
from sparkfuse.partition import partition_export


class Score(torch.nn.Module):
    def __init__(self, explicit=False):
        super().__init__()
        self.explicit = explicit

    def forward(self, x, mask):
        z = x * 0.125 + mask
        z_for_prob = z.float() if self.explicit else z
        p = z_for_prob.softmax(-1, dtype=torch.float32)
        return (z, p) if x.dtype == torch.float32 else (z, p, p.to(x.dtype))


def args(rows, width, dt, offset, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def make(r, n):
        return torch.randn(r * width + n, device="cuda", dtype=dt, generator=g)[n:].reshape(
            r, width
        )

    x = make(rows, offset)
    mask = make(1, offset * 3)
    mask[:, ::37] = float("-inf")
    return x, mask


def exact(a, b):
    assert a.dtype == b.dtype and a.stride() == b.stride() and a.shape == b.shape
    assert torch.equal(a.isnan(), b.isnan())
    valid = ~b.isnan()
    it = torch.int32 if a.dtype == torch.float32 else torch.int16
    assert torch.equal(a.contiguous().view(it)[valid], b.contiguous().view(it)[valid])


@pytest.mark.parametrize(
    "width,threads", [(1025, 32), (1089, 64), (1536, 128), (2047, 256), (2048, 128)]
)
@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16, torch.float32])
@torch.inference_mode()
def test_slots_visible_raw_and_active_stream(width, threads, dt):
    m = Score(explicit=dt == torch.float16)
    initial = args(13, width, dt, 0, 75781)
    ep = torch.export.export(m, initial)
    k = compile_export(ep, schedule=Schedule(threads, "warp_slots", True))
    native = native_dispatch(k)
    assert "const int j=lane+slot*32" in k.source and "held[slot]" in k.source
    for off, fn in [(0, k), (1, native), (3, k)]:
        x, mask = args(13, width, dt, off, 75821 + off)
        x[0].fill_(float("-inf"))
        x[1, 1] = float("nan")
        x[2, 2] = float("inf")
        x[3, ::2] = torch.finfo(dt).max
        x[4, 1::2] = torch.finfo(dt).min
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            actual = fn(x, mask)
            ref = m(x, mask)
        torch.cuda.current_stream().wait_stream(stream)
        for a, b in zip(actual, ref, strict=True):
            exact(a, b)


@pytest.mark.parametrize(
    "w,layout,expected",
    [
        (31, "slots", "outside_ordered_score_domain"),
        (32, "slots", "warp_xor"),
        (1024, "slots", "warp_xor"),
        (1025, "loop", "warp_xor"),
        (1025, "slots", "warp_slots"),
        (2048, "slots", "warp_slots"),
        (2049, "slots", "outside_ordered_score_domain"),
    ],
)
@torch.inference_mode()
def test_ordered_score_dispatch(w, layout, expected):
    a = args(3, w, torch.bfloat16, 1, 76301)
    m = Score()
    gm, rr = partition_export(torch.export.export(m, a), ordered_scores=layout)
    gm.graph.lint()
    assert any(
        r.get("reason") == expected or r.get("schedule", {}).get("layout") == expected for r in rr
    )
    if expected == "outside_ordered_score_domain":
        assert any(str(n.target) == "aten.softmax.int" for n in gm.graph.nodes)
    for x, y in zip(gm(*a), m(*a), strict=True):
        exact(x, y)


@pytest.mark.parametrize(
    "w,threads,cache,vec",
    [
        (1024, 128, True, 1),
        (2049, 128, True, 1),
        (1089, 512, True, 1),
        (1089, 128, False, 1),
        (1089, 128, True, 2),
    ],
)
def test_illegal_slot_schedule(w, threads, cache, vec):
    ep = torch.export.export(Score(), (torch.randn(2, w), torch.randn(1, w)))
    with pytest.raises(Unsupported):
        Schedule(threads, "warp_slots", cache, vec).validate(lower(ep))


def test_conflicting_and_unknown_score_policy():
    ep = torch.export.export(Score(), (torch.randn(2, 34), torch.randn(1, 34)))
    for kw in (
        {"ordered_scores": "unknown"},
        {"ordered_scores": "slots", "framework_softmax": True},
    ):
        with pytest.raises(ValueError, match="softmax policy"):
            partition_export(ep, **kw)


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("width", [1024, 1025, 1537, 2048])
@torch.inference_mode()
def test_implicit_half_precision_boundary(width, raw):
    class Implicit(torch.nn.Module):
        def forward(self, x, mask):
            z = x * 0.08838834764831845 + mask
            p = (
                torch.ops.aten._softmax.default(z, -1, True)
                if raw
                else z.softmax(-1, dtype=torch.float32)
            )
            return p.half(), z, p

    a = args(17, width, torch.float16, 1, 76579)
    m = Implicit()
    ep = torch.export.export(m, a)
    plan = lower(ep)
    assert plan.vals[plan.red].imm == "half_to_float"
    for mode in ("slots", "loop"):
        gm, records = partition_export(ep, ordered_scores=mode, host_backend="native")
        boundaries = [r for r in records if r.get("kind") == "reference_boundary"]
        if width > 1024:
            assert len(boundaries) == 1 and boundaries[0]["ordered_width_limit"] == 1024
            assert boundaries[0]["half_to_float"] and boundaries[0]["target"] == (
                "aten._softmax.default" if raw else "aten.softmax.int"
            )
            with pytest.raises(Unsupported, match="keep ATen"):
                Schedule(128, "warp_slots" if mode == "slots" else "warp_xor", True).validate(plan)
        else:
            assert not boundaries and any(
                r.get("schedule", {}).get("layout") == "warp_xor" for r in records
            )
        for off in (0, 1):
            vals = args(17, width, torch.float16, off, 76757 + off)
            for v, ref in zip(gm(*vals), m(*vals), strict=True):
                exact(v, ref)
