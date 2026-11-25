"""Reference softmax is a policy boundary even when its input and outputs have other users."""

import pytest
import torch
from sparkfuse.partition import partition_export


class Soft(torch.nn.Module):
    def forward(self, logits, mask):
        score = logits * 0.125 + mask
        p = torch.softmax(score, -1, dtype=torch.float32)
        q = torch.ops.aten._softmax.default(score, -1, False)
        return p.to(logits.dtype), q, score, q[..., 1:]


@pytest.mark.parametrize("w", [31, 1024, 1025, 2049])
@pytest.mark.parametrize("dt", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_softmax_graph_boundaries(w, dt):
    m = Soft()
    rng = torch.Generator(device="cuda").manual_seed(46763)
    a, b = [torch.randn(3, w, device="cuda", dtype=dt, generator=rng) for _ in range(2)]
    b[:, ::29] = float("-inf")
    ep = torch.export.export(m, (a, b))
    gm, records = partition_export(ep, framework_softmax=True)
    witness = [r["target"] for r in records if r.get("reason") == "framework_softmax"]
    assert set(witness) == {"aten.softmax.int", "aten._softmax.default"} and len(witness) == 2
    assert any(r.get("module") and "mul" in r["ops"] and "add" in r["ops"] for r in records)
    for x, y in zip(gm(a, b), m(a, b), strict=True):
        it = torch.int32 if x.dtype == torch.float32 else torch.int16
        assert x.stride() == y.stride() and torch.equal(
            x.contiguous().view(it), y.contiguous().view(it)
        )
    _, q, _, view = gm(a, b)
    assert q.untyped_storage().data_ptr() == view.untyped_storage().data_ptr()
