import pytest
import torch
from examples.ragged_pages import RaggedGatherDecode, ragged_request, explicit_mask, PROFILES
from examples.paged_decode_example import PageGatherDecode
from sparkfuse.paged_decode import compile_paged_decode, recognize
from sparkfuse.ir import Unsupported


@pytest.mark.parametrize(
    "cfg", [(4, 8, 2, 64, 32, 31, 1, "bf16"), (4, 6, 2, 128, 64, 129, 4, "fp16")]
)
@torch.inference_mode()
def test_reusable_export_matches_materialized_mask_and_empty_sign(cfg):
    m = RaggedGatherDecode(cfg[5])
    a, b = [
        ragged_request(cfg, 5971 + i, p, "shuffled") for i, p in enumerate(("full", "extremes"))
    ]
    z = compile_paged_decode(torch.export.export(m, a), a, b)
    aa, bb = explicit_mask(a), explicit_mask(b)
    base = PageGatherDecode(cfg[5])
    f = compile_paged_decode(torch.export.export(base, aa), aa, bb)
    s = torch.cuda.Stream()
    live = []
    for i, p in enumerate(PROFILES):
        xs = ragged_request(cfg, 6011 + i, p, "repeated" if i % 2 else "shared")
        ref = m(*xs)
        old = f.run_bounded(*explicit_mask(xs))
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            y = z(*xs)
        torch.cuda.current_stream().wait_stream(s)
        live.append(y)
        assert torch.equal(y.view(torch.int16), old.view(torch.int16))
        torch.testing.assert_close(y, ref, atol=0.004, rtol=0.016, check_stride=True)
        assert not y.view(torch.int16)[xs[4] <= 0].count_nonzero()
        assert all(y.untyped_storage().data_ptr() != x.untyped_storage().data_ptr() for x in xs)
    assert len({q.data_ptr() for q in live}) == len(live)


@torch.inference_mode()
def test_merge_never_reads_unwritten_tail_or_poisoned_inactive_pages():
    cfg = (4, 8, 2, 64, 32, 129, 4, "bf16")
    m = RaggedGatherDecode(129)
    a, b = [ragged_request(cfg, 6211 + i, p, "shuffled") for i, p in enumerate(("mixed", "full"))]
    f = compile_paged_decode(torch.export.export(m, a), a, b)
    xs = ragged_request(cfg, 6493, "tile_edges", "shuffled")
    xs[4].copy_(torch.tensor([0, 1, 32, 65], device="cuda", dtype=torch.int32))
    got = f(*xs)
    (scratch,) = f.first(*xs)
    for row, n in enumerate(xs[4].tolist()):
        scratch[row, :, (n + 31) // 32 :].fill_(float("nan"))
    (redo,) = f.merge(scratch, xs[4])
    assert torch.equal(redo.view(torch.int16), got.view(torch.int16))
    xs[4].zero_()
    scratch.fill_(float("nan"))
    assert not f.merge(scratch, xs[4])[0].view(torch.int16).count_nonzero()


@torch.inference_mode()
def test_nonfinite_excluded_bias_and_integer_length_validation():
    cfg = (4, 8, 2, 64, 32, 65, 4, "bf16")
    a = ragged_request(cfg, 7103, "mixed", "endpoints")
    b = ragged_request(cfg, 7121, "full", "shared")
    ep = torch.export.export(RaggedGatherDecode(65), a)
    f = compile_paged_decode(ep, a, b)
    q, k, v, ids, lens, bias = a
    with pytest.raises(ValueError):
        f.run_bounded(q, k, v, ids, lens.long(), bias)
    with pytest.raises(ValueError):
        f.run_bounded(q, k, v, ids, lens[1:], bias)
    bias.fill_(-torch.inf)
    assert not f(*a).view(torch.int16).count_nonzero()
    bias[1, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        f(*a)


@torch.inference_mode()
def test_mask_predicate_and_fill_value_must_match():
    cfg = (4, 8, 2, 64, 32, 65, 4, "bf16")
    a = ragged_request(cfg, 7699, "full", "shared")
    ep = torch.export.export(RaggedGatherDecode(65), a)
    nodes = list(ep.graph.nodes)
    (n,) = [v for v in nodes if v.target == torch.ops.aten.ge.Tensor]
    n.target = torch.ops.aten.gt.Tensor
    with pytest.raises(Unsupported):
        recognize(ep)
    n.target = torch.ops.aten.ge.Tensor
    (n,) = [v for v in nodes if v.target == torch.ops.aten.masked_fill.Scalar]
    n.args = (*n.args[:-1], -10000.0)
    with pytest.raises(Unsupported):
        recognize(ep)
