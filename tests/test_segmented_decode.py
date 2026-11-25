"""Segmented attention respects each independently observable cache and its physical layout."""

import pytest
import torch
import torch.nn.functional as F
from sparkfuse.decode_sites import partition_decode_sites
from sparkfuse.segmented_decode import partition_segmented_decode, _bits


class Boundary(torch.nn.Module):
    def __init__(self, live, early=False):
        super().__init__()
        self.live = live
        self.early = early

    def forward(self, q, k0, k1, v0, v1, mask):
        k = torch.cat((k0, k1), 2)
        v = torch.cat((v0, v1), 2)
        if self.early:
            q = q + k[:, :1, :1, :]
        y = F.scaled_dot_product_attention(q, k, v, mask, enable_gqa=True)
        return (y, *((k, k.view_as(k)) if self.live & 1 else ()), *((v,) if self.live & 2 else ()))


def args(past, dt, token_major):
    def buf(shape):
        return torch.randn(shape, dtype=dt, device="cuda")

    def old(sw):
        return buf((2, past, 2, 64)).transpose(1, 2) if sw else buf((2, 2, past, 64))

    q = buf((2, 8, 1, 64))
    m = buf((2, 1, 1, past + 1))
    m[0].fill_(-torch.inf)
    x = [q, old(token_major), buf((2, 2, 1, 64)), old(not token_major), buf((2, 2, 1, 64)), m]
    for t in x[1:5]:
        t[..., ::9] = -0.0
    return tuple(x)


@pytest.mark.parametrize(
    "live,past,token,dt",
    [
        (0, 15, True, torch.float16),
        (1, 32, False, torch.bfloat16),
        (2, 64, True, torch.float16),
        (3, 128, False, torch.bfloat16),
        (3, 256, True, torch.float16),
    ],
)
@torch.inference_mode()
def test_segmented_append_visibility(live, past, token, dt):
    m = Boundary(live)
    ex, au, hold = [args(past, dt, token) for _ in range(3)]
    ep = torch.export.export(m, ex)
    f = partition_segmented_decode(ep, ex, au)
    compact = partition_decode_sites(ep, ex, au, domain="bounded")
    r = f.sites[0]["segmented"]
    assert (
        r["published"] == [bool(live & 1), bool(live & 2)]
        and len(r["appends"]) == 2
        and r["private_output_bytes_saved"]
        == (2 - (live & 1) - bool(live & 2)) * 2 * 2 * (past + 1) * 64 * 2
    )
    assert not any(n.target == torch.ops.aten.cat.default for n in f.module.graph.nodes)
    orig = [t.clone(memory_format=torch.preserve_format) for t in hold]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out = f.module(*hold)
        second = f.module(*hold)
        base = compact.module(*hold)
        ref = m(*hold)
    torch.cuda.current_stream().wait_stream(stream)
    assert all(_bits(a, b) for a, b in zip(out, base, strict=True))
    assert all(_bits(a, b) for a, b in zip(out[1:], ref[1:], strict=True))
    assert out[0].data_ptr() != second[0].data_ptr() and not torch.count_nonzero(out[0][0])
    assert all(_bits(a, b) for a, b in zip(orig, hold, strict=True))
    assert not {x.untyped_storage().data_ptr() for x in out} & {
        x.untyped_storage().data_ptr() for x in hold + tuple(second)
    }
    if live & 1:
        assert out[1].data_ptr() == out[2].data_ptr()
    for j in range(5):
        bad = list(hold)
        bad[j] = torch.full_like(hold[j], 65, memory_format=torch.preserve_format)
        with pytest.raises(ValueError, match="numerical domain"):
            f.module(*bad)


@torch.inference_mode()
def test_refuse_cycle_when_query_reads_appended_cache():
    ex, au = args(32, torch.bfloat16, True), args(32, torch.bfloat16, True)
    m = Boundary(3, True)
    ep = torch.export.export(m, ex)
    r = partition_segmented_decode(ep, ex, au)
    assert (
        len(r.sites) == 1
        and "segmented" not in r.sites[0]
        and "dominate" in r.skipped[-1]["reason"]
    )
    assert any(n.target == torch.ops.aten.cat.default for n in r.module.graph.nodes)
