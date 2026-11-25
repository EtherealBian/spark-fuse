"""Validate the four-output cache-copy pass, including externally visible alias views."""

import torch
import pytest
from sparkfuse.partition import partition_export
from test_cache_append import addr, check, inputs


class KV(torch.nn.Module):
    def __init__(self, groups=6, mode="visible"):
        super().__init__()
        self.groups = groups
        self.mode = mode

    def forward(self, k, kt, v, vt):
        ko = torch.cat((k, kt), 2)
        if self.mode == "dependent":
            vt = ko[:, :, -vt.size(2) :].sin()
        vo = torch.cat((v, vt), 2)
        b, h, s, d = ko.shape
        ku = ko.unsqueeze(2)
        ke = ku.expand(b, h, self.groups, s, d)
        kr = ke.reshape(b, h * self.groups, s, d)
        g = self.groups + (self.mode == "mismatch")
        vu = vo.unsqueeze(2)
        ve = vu.expand(b, h, g, s, d)
        vr = ve.reshape(b, h * g, s, d)
        if self.mode == "hidden":
            return kr, vr
        if self.mode == "views":
            return ku, ve, kr, vr
        return ko, vo, kr, vr, ku, ve, ko[:, :, 1:], vr[..., 1:]


@pytest.mark.parametrize("host", ["native", "python"])
@pytest.mark.parametrize(
    "dt,d,sw,mode",
    [
        (torch.bfloat16, 128, (True, False, True, False), "visible"),
        (torch.float32, 9, (False, True, False, True), "views"),
        (torch.float16, 14, (True, False, False, True), "visible"),
    ],
)
@torch.inference_mode()
def test_fanout_bits_aliases_streams(host, dt, d, sw, mode):
    m = KV(mode=mode)
    ep = torch.export.export(m, inputs(dt, d, sw, (0,) * 4))
    gm, rs = partition_export(
        ep, host_backend=host, cache_fanout=True, paired_appends=True, paired_head_repeats=True
    )
    assert len([r for r in rs if r.get("kind") == "cache_fanout"]) == 1 and not any(
        r.get("kind") in ("paired_append", "paired_head_repeat") for r in rs
    )
    for offsets in [(0,) * 4, (1, 3, 2, 5), (2,) * 4, (4,) * 4]:
        args = inputs(dt, d, sw, offsets)
        variants = [args]
        if sw[0:2] == sw[2:4]:
            variants.append(args[:2] * 2)
        for a in variants:
            before = [x.clone() for x in a]
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                got = gm(*a)
                ref = m(*a)
            torch.cuda.current_stream().wait_stream(stream)
            check(got, ref)
            check(a, before)
            actual = (*a, *got)
            expected = (*a, *ref)
            assert all(
                (addr(x) == addr(y)) == (addr(expected[i]) == addr(expected[j]))
                for i, x in enumerate(actual)
                for j, y in enumerate(actual)
            )


@pytest.mark.parametrize(
    "mode,g", [("dependent", 6), ("mismatch", 6), ("hidden", 6), ("visible", 1)]
)
@torch.inference_mode()
def test_fanout_fallback(mode, g):
    a = inputs(torch.float16, 8, (False,) * 4, (0,) * 4)
    m = KV(g, mode)
    gm, rs = partition_export(torch.export.export(m, a), cache_fanout=True)
    assert not any(r.get("kind") == "cache_fanout" for r in rs)
    if mode != "dependent":
        check(gm(*a), m(*a))


@torch.inference_mode()
def test_fanout_default_unchanged():
    a = inputs(torch.bfloat16, 16, (False,) * 4, (0,) * 4)
    ep = torch.export.export(KV(), a)
    gm, rs = partition_export(ep, paired_appends=True, paired_head_repeats=True)
    assert {r.get("kind") for r in rs} >= {"paired_append", "paired_head_repeat"} and not any(
        r.get("kind") == "cache_fanout" for r in rs
    )
    check(gm(*a), KV()(*a))


@pytest.mark.parametrize("host", ["python", "native"])
def test_fanout_runtime_rejects_incompatible_storage(host):
    a = inputs(torch.bfloat16, 16, (False,) * 4, (0,) * 4)
    ep = torch.export.export(KV(), a)
    gm, recs = partition_export(ep, cache_fanout=True, host_backend=host)
    mod = gm.get_submodule(next(r["module"] for r in recs if r.get("kind") == "cache_fanout"))
    k = a[0]
    for bad in (
        k.float(),
        k.cpu(),
        k[:, :, :-1],
        k.transpose(1, 2).contiguous().transpose(1, 2),
        k.clone().requires_grad_(True),
    ):
        with pytest.raises((ValueError, RuntimeError)):
            mod(bad, *a[1:])
