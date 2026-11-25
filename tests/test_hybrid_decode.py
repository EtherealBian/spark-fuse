import pytest
import torch
from torch.nn import functional as F
from test_symbolic_sites import Step, req, export
from sparkfuse.hybrid_decode import partition_hybrid_decode, _bits, _RegionModule


class Visible(Step):
    def forward(self, q, k, v, nk, nv, mask):
        # Intermediates using the symbolic KV axis must remain dynamic; never specialize them
        # from the two observed lengths. Public residual is read by both reduction and caller.
        dyn = F.relu(k + k)
        h, ck, cv = super().forward(q, dyn, v, nk, nv, mask)
        r = h + h
        f = r.float()
        norm = (f * torch.rsqrt(f.pow(2).mean(-1, True) + 1e-6)).to(r.dtype)
        return F.silu(norm) * norm, ck, cv, r


@pytest.mark.parametrize("host", [False, True])
@torch.inference_mode()
def test_dynamic_axes_and_visible_residual(host):
    torch.manual_seed(1019)
    m = Visible().to(device="cuda", dtype=torch.bfloat16).eval()
    a, b = req(7), req(51)
    ep = export(m, a)
    graph = str(ep.graph)
    h = partition_hybrid_decode(ep, a, b, domain="bounded", host_calls=host)
    r = [v for v in h.regions if "module" in v]
    assert r and h.attention and bool(h.host_calls) == host
    dynamic = next(n for n in ep.graph.nodes if n.target is torch.ops.aten.relu.default)
    assert all(dynamic.name not in d["ops"] and dynamic.args[0].name not in d["ops"] for d in r)
    assert any(len(d["live_outs"]) >= 2 for d in r)
    assert str(ep.graph) == graph
    live = []
    for length in (2, 31, 32, 63, 129):
        xs = req(length)
        ys = h.module(*xs)
        truth = m(*xs)
        assert all(torch.equal(x, y) for x, y in zip(ys[1:3], truth[1:3], strict=True))
        for x, y in zip(ys, truth, strict=True):
            torch.testing.assert_close(x, y, atol=0.035, rtol=0.04, check_stride=True)
        live.extend(ys)
    assert len({x.untyped_storage().data_ptr() for x in live}) == 20
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        h.module(*req(130))


@torch.inference_mode()
def test_second_trace_mismatch_keeps_aten(monkeypatch):
    m = Visible().to(device="cuda", dtype=torch.bfloat16).eval()
    a, b = req(7), req(51)
    ep = export(m, a)
    orig = _RegionModule.forward
    calls = {}

    def injected(self, *args):
        ys = orig(self, *args)
        calls[self] = calls.get(self, 0) + 1
        if calls[self] == 2:
            return tuple(y + 1 for y in ys)
        return ys

    monkeypatch.setattr(_RegionModule, "forward", injected)
    h = partition_hybrid_decode(ep, a, b)
    assert (
        len(calls) > 0
        and all(k == 2 for k in calls.values())
        and not any("module" in r for r in h.regions)
        and any("not byte/ownership" in r.get("fallback", "") for r in h.regions)
    )
    xs = req(33)
    assert all(torch.equal(y, z) for y, z in zip(h.module(*xs)[1:3], m(*xs)[1:3], strict=True))


def test_byte_gate_distinguishes_signed_zero_and_nan_payload():
    assert not _bits(torch.tensor([0.0]), torch.tensor([-0.0])) and _bits(
        torch.tensor([float("nan")]), torch.tensor([float("nan")])
    )
