import pytest
import torch
from test_symbolic_sites import Step, req
from sparkfuse.hybrid_decode import partition_hybrid_decode
from sparkfuse.index_program import IndexRegion


class RotStep(Step):
    def __init__(self, expose=False):
        super().__init__()
        self.exposed = expose

    def forward(self, q, oldk, oldv, nk, nv, m, c, s):
        def rot(x):
            hi = -x[..., 32:]
            lo = x[..., :32]
            return x * c + torch.cat((hi, lo), -1) * s, hi

        q, half = rot(q)
        nk, _ = rot(nk)
        out = super().forward(q, oldk, oldv, nk, nv, m)
        return (*out, q, nk, half) if self.exposed else (*out, q, nk)


def batch(s):
    return (
        *req(s),
        *(torch.randn(2, 1, 1, 64, device="cuda", dtype=torch.bfloat16) for _ in range(2)),
    )


def export(m, a):
    p = torch.export.Dim("past", min=2, max=129)
    return torch.export.export(
        m, a, dynamic_shapes=({}, {2: p}, {2: p}, {}, {}, {3: p + 1}, {}, {}), strict=True
    )


@pytest.mark.parametrize("exposed", [False, True])
@torch.inference_mode()
def test_visible_rotations_and_extra_half_consumer(exposed):
    torch.manual_seed(1221)
    a, b = batch(7), batch(45)
    m = RotStep(exposed).to(device="cuda", dtype=torch.bfloat16).eval()
    ep = export(m, a)
    g = str(ep.graph)
    h = partition_hybrid_decode(ep, a, b, paired_rotary=True)
    assert h.attention and sum("module" in r for r in h.rotary) == int(not exposed)
    live = []
    for s in (2, 31, 65, 129):
        xs = batch(s)
        y = h.module(*xs)
        ref = m(*xs)
        torch.testing.assert_close(y[0], ref[0], atol=0.025, rtol=0.02, check_stride=True)
        for x, z in zip(y[1:], ref[1:], strict=True):
            assert (
                torch.equal(x.reshape(-1).view(torch.uint8), z.reshape(-1).view(torch.uint8))
                and x.stride() == z.stride()
                and (x._base is None) == (z._base is None)
            )
        live.extend(y)
    assert len({x.untyped_storage().data_ptr() for x in live}) == len(live) and str(ep.graph) == g


@torch.inference_mode()
def test_reject_second_trace_corruption_and_do_not_delete_original(monkeypatch):
    m = RotStep().to(device="cuda", dtype=torch.bfloat16).eval()
    a, b = batch(7), batch(45)
    ep = export(m, a)
    orig = IndexRegion.forward
    calls = []

    def inject(self, *xs):
        ys = orig(self, *xs)
        calls.append(1)
        return ys if len(calls) != 2 else (ys[0], ys[1] + 1)

    monkeypatch.setattr(IndexRegion, "forward", inject)
    h = partition_hybrid_decode(ep, a, b, paired_rotary=True)
    assert (
        len(calls) == 2
        and len(h.rotary) == 1
        and "fallback" in h.rotary[0]
        and "module" not in h.rotary[0]
    )
    xs = batch(63)
    ys = h.module(*xs)
    ref = m(*xs)
    assert all(torch.equal(x, y) for x, y in zip(ys[1:], ref[1:], strict=True))
