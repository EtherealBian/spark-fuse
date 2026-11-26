import pytest
import torch
from examples.cache_slot import CacheSlot, request
from sparkfuse.slot_forward import compile_slot_forward, recognize
from sparkfuse.ir import Unsupported


@pytest.mark.parametrize("vis", [(False, False), (True, False), (False, True), (True, True)])
@torch.inference_mode()
def test_all_bit_patterns_visibility_offsets_and_ownership(vis):
    cfg = (3, 2, 33, 64, "bf16")
    m = CacheSlot(vis)
    a, b = [request(cfg, 19773 + i, "endpoints") for i in (0, 1)]
    ep = torch.export.export(m, a)
    f = compile_slot_forward(ep, a, b)
    assert f.plan.visible == vis
    side = torch.cuda.Stream()
    orig = torch.cuda.current_stream()
    outs = []
    expected = []
    for off, p in enumerate(("equal", "different", "mixed", "endpoints")):

        def offset(x):
            return (
                torch.empty(x.numel() + off, dtype=x.dtype, device=x.device)[off:]
                .view_as(x)
                .copy_(x)
            )

        xs = tuple(map(offset, request(cfg, 19857 + 71 * off, p)))
        old = tuple(x.clone() for x in xs)
        ref = m(*xs)
        side.wait_stream(orig)
        with torch.cuda.stream(side):
            ys = f(*xs)
        orig.wait_stream(side)
        assert len(ys) == 2 + sum(vis)
        assert all(
            y._base is None and y.data_ptr() not in {x.untyped_storage().data_ptr() for x in xs}
            for y in ys
        )
        assert all(
            torch.equal(x.view(torch.uint8), z.view(torch.uint8))
            for x, z in zip(xs, old, strict=True)
        )
        outs.extend(ys)
        expected.extend(ref)
    assert len({x.data_ptr() for x in outs}) == len(outs)
    assert all(
        torch.equal(y.view(torch.int16), r.view(torch.int16)) and y.stride() == r.stride()
        for y, r in zip(outs, expected, strict=True)
    )


@torch.inference_mode()
def test_requires_saturating_paired_graph_and_rejects_escapes():
    cfg = (2, 2, 17, 32, "fp16")
    a = request(cfg, 20069, "mixed")
    m = CacheSlot((True, False))
    ep = torch.export.export(m, a)
    ns = list(ep.graph.nodes)
    wi = next(n for n in ns if n.target == torch.ops.aten.clamp.default)
    wi.args = (wi.args[0], -1, 16)
    with pytest.raises(Unsupported):
        recognize(ep)
    wi.args = (wi.args[0], 0, 16)
    gath = [n for n in ns if n.target == torch.ops.aten.gather.default]
    before = gath[1].args
    gath[1].args = (gath[0].args[0], *before[1:])
    with pytest.raises(Unsupported):
        recognize(ep)
    gath[1].args = before
    out = ns[-1]
    ys = out.args[0]
    out.args = ((ys[0], ys[1], wi),)
    with pytest.raises(Unsupported):
        recognize(ep)
    out.args = (ys,)
    assert recognize(ep).visible == (True, False)


@torch.inference_mode()
def test_guards_alias_inputs_and_checked_independent_storage():
    cfg = (2, 2, 1, 32, "fp16")
    a, b = [request(cfg, i, "endpoints") for i in (20183, 20323)]
    x = (a[0], a[0], a[2], a[2], a[4], a[4])
    m = CacheSlot((True, True))
    ep = torch.export.export(m, a)
    f = compile_slot_forward(ep, a, b)
    ys = f(*x)
    ref = m(*x)
    assert len({t.data_ptr() for t in ys}) == 4 and all(
        torch.equal(t.view(torch.int16), v.view(torch.int16)) for t, v in zip(ys, ref, strict=True)
    )
    with pytest.raises(Unsupported):
        compile_slot_forward(ep, a, a)
    with torch.inference_mode(False), pytest.raises(ValueError):
        f(*a)
    for wrong in ((a[0].cpu(), *a[1:]), (*a[:-1], a[-1].long()), a[:-1], (*a[:-1], a[-1][:1])):
        with pytest.raises(ValueError):
            f(*wrong)
