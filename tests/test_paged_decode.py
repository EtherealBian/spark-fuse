import pytest
import torch
from torch import fx
from examples.paged_decode_example import PageGatherDecode, request, materialize
from sparkfuse.paged_decode import compile_paged_decode, recognize
from sparkfuse.ir import Unsupported


@pytest.mark.parametrize(
    "cfg", [(2, 6, 2, 128, 64, 63, 2, "bf16"), (1, 8, 4, 64, 32, 65, 0, "fp16")]
)
@torch.inference_mode()
def test_sharing_saturation_tail_stream_and_freshness(cfg):
    s = cfg[5]
    m = PageGatherDecode(s)
    ex = request(cfg, 7121, "shuffled")
    au = request(cfg, 8419, "endpoints")
    ep = torch.export.export(m, ex)
    f = compile_paged_decode(ep, ex, au)
    assert f.plan.page == cfg[4] and len(f.programs) == 1 + int(s > cfg[4])
    other = torch.cuda.Stream()
    live = []
    for j, p in enumerate(("shuffled", "shared", "repeated", "endpoints")):
        xs = request(cfg, 8813 + j, p)
        base = m(*xs)
        other.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(other):
            a = f(*xs)
            b = f.run_bounded(*xs)
        torch.cuda.current_stream().wait_stream(other)
        live.extend((a, b))
        torch.testing.assert_close(a, base, atol=0.004, rtol=0.016, check_stride=True)
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))
        assert all(a.untyped_storage().data_ptr() != v.untyped_storage().data_ptr() for v in xs)
    assert len({a.data_ptr() for a in live}) == len(live)


@torch.inference_mode()
def test_ignores_unused_pages_and_tail_but_reads_actual_map():
    cfg = (2, 8, 2, 64, 32, 37, 0, "bf16")
    m = PageGatherDecode(37)
    ex = request(cfg, 9323, "shuffled")
    f = compile_paged_decode(torch.export.export(m, ex), ex, request(cfg, 9283, "endpoints"))
    q, k, v, tab = ex
    tab.zero_()
    tab[:, 1] = 1
    before = f(q, k, v, tab)
    k[1, :, 5:] = 51
    v[1, :, 5:] = -51
    k[2:] = 13
    v[2:] = 19
    assert torch.equal(f(q, k, v, tab).view(torch.int16), before.view(torch.int16))
    tab.fill_(2)
    out = f(q, k, v, tab)
    assert torch.equal(out, torch.full_like(q, 19)) and not torch.equal(out, before)


@torch.inference_mode()
def test_rejects_unaligned_wrong_types_domain_policy_and_budget():
    cfg = (2, 8, 2, 64, 32, 33, 2, "bf16")
    m = PageGatherDecode(33)
    ex = request(cfg, 9323, "shuffled")
    au = request(cfg, 8371, "shared")
    ep = torch.export.export(m, ex)
    f = compile_paged_decode(ep, ex, au)
    q, k, v, table, mask = ex
    raw = torch.empty(q.numel() + 1, device=q.device, dtype=q.dtype)
    off = raw[1:].view_as(q)
    for xs in [(off, k, v, table, mask), (q, k, v, table.long(), mask), (q, k, v, table[:1], mask)]:
        with pytest.raises(ValueError):
            f.run_bounded(*xs)
    with pytest.raises(Unsupported):
        compile_paged_decode(ep, ex, au, max_workspace_bytes=f.plan.scratch - 1)
    with pytest.raises(Unsupported):
        compile_paged_decode(ep, ex, ex)
    q[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError):
        f(*ex)


@torch.inference_mode()
def test_matcher_refuses_visible_page_reads_and_changed_semantics():
    cfg = (2, 8, 2, 64, 32, 65, 0, "bf16")
    ex = request(cfg, 9733, "shuffled")
    m = PageGatherDecode(65)

    class Expose(torch.nn.Module):
        def forward(self, q, k, v, t):
            return m(q, k, v, t), t.clamp(0, k.shape[0] - 1)

    with pytest.raises(Unsupported):
        recognize(torch.export.export(Expose(), ex))
    ep = torch.export.export(m, ex)
    ns = list(ep.graph.nodes)
    (n,) = [n for n in ns if n.target == torch.ops.aten.clamp.default]
    a, lo, hi = n.args
    n.args = (a, lo, hi - 1)
    with pytest.raises(Unsupported):
        recognize(ep)
    n.args = (a, lo, hi)
    n = next(z for z in ns if z.target == torch.ops.aten.permute.default)
    n.args = (n.args[0], [0, 1, 2, 3, 4])
    with pytest.raises(Unsupported):
        recognize(ep)
