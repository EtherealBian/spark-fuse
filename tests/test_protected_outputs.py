import operator
import pytest
import torch
from torch import fx
from sparkfuse.ir import Unsupported
from sparkfuse.observable import OutputSlice
from sparkfuse.hybrid_decode import partition_hybrid_decode, _bits
from test_symbolic_sites import Step, req, export


@pytest.mark.parametrize("exact,count", [((0,), 0), ((1, 2), 1), ((0, 1, 2), 0)])
@torch.inference_mode()
def test_protect_selected_flat_outputs(exact, count):
    m = Step().to(device="cuda", dtype=torch.bfloat16).eval()
    ex, au = req(7), req(51)
    ep = export(m, ex)
    graph = str(ep.graph)
    base = ep.module()
    part = partition_hybrid_decode(
        ep,
        ex,
        au,
        host_calls=True,
        paired_rotary=True,
        paired_append=True,
        exact_outputs=exact,
        warp_heads=4,
    )
    assert (
        len(part.attention) == count
        and part.protection["flat_indices"] == list(exact)
        and part.protection["unchanged_dependencies"]
        and part.protection["protected_nodes"]
    )
    assert not part.appends and str(ep.graph) == graph
    for s in (2, 31, 32, 77, 129):
        xs = req(s)
        xs[-1][1].fill_(-torch.inf)
        ys = part.module(*xs)
        ref = base(*xs)
        for i in exact:
            assert _bits(ys[i], ref[i]) and ys[i].stride() == ref[i].stride()
        for y, t in zip(ys, ref, strict=True):
            torch.testing.assert_close(y, t, atol=0.012, rtol=0.02, check_stride=True)
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        part.module(*req(130))


@torch.inference_mode()
def test_visible_repeat_only_one_branch_and_effects_still_checked():
    m = Step(public_repeat=True).to(device="cuda", dtype=torch.bfloat16).eval()
    ex, au = req(7), req(51)
    ep = export(m, ex)
    h = partition_hybrid_decode(ep, ex, au, host_calls=True, paired_append=True, exact_outputs=(3,))
    assert len(h.attention) == 1 and not h.appends and len(h.attention[0]["erased_repeats"]) == 1
    xs = req(44)
    assert _bits(h.module(*xs)[3], m(*xs)[3])
    ep = export(Step(dropout=0.1).to(device="cuda", dtype=torch.bfloat16).eval(), ex)
    rng = torch.cuda.get_rng_state()
    with pytest.raises(Unsupported):
        partition_hybrid_decode(ep, ex, au, exact_outputs=(0,))
    assert torch.equal(torch.cuda.get_rng_state(), rng)


def test_output_slice_uses_identity_arguments_and_flat_order():
    g = fx.Graph()
    a = g.placeholder("a")
    b = g.call_function(operator.add, (a, 1))
    c = g.call_function(operator.neg, (a,))
    out = g.output((c, {"cache": b}))
    m = fx.GraphModule({}, g)
    s = OutputSlice.capture(m, (1,))
    assert b in s.nodes and c not in s.nodes and s.leaves == (b,)
    s.verify(m)
    for indices in ((-1,), (2,), (True,), (1, 1), [0]):
        with pytest.raises(Unsupported):
            OutputSlice.capture(m, indices)
    b.args = (a, 2)
    with pytest.raises(Unsupported):
        s.verify(m)
    b.args = (a, 1)
    s.verify(m)
    with g.inserting_before(out):
        same = g.call_function(operator.add, (a, 1))
    out.args = ((c, {"cache": same}),)
    with pytest.raises(Unsupported):
        s.verify(m)
