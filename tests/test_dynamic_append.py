import pytest
import torch
from torch import nn
from test_symbolic_sites import Step, req, export
from sparkfuse.dynamic_append import Append
from sparkfuse.hybrid_decode import _bits, partition_hybrid_decode


def plan(dt, d):
    return dict(batch=2, kv_heads=2, width=d, lower=3, upper=130, device="cuda:0", dtype=str(dt))


def storage(shape, dt, offset=0):
    size = 1
    for x in shape:
        size *= x
    buf = torch.randint(-32768, 32768, (size + offset,), dtype=torch.int16, device="cuda")
    return buf[offset:].view(dt).reshape(shape)


@pytest.mark.parametrize("dt,width", [(torch.bfloat16, 64), (torch.float16, 128)])
@torch.inference_mode()
def test_append_payload_alignment_layout_alias_and_two_streams(dt, width):
    op = Append(plan(dt, width))
    current = torch.cuda.current_stream()
    other = torch.cuda.Stream()
    live = []
    for s in (2, 31, 64, 129):
        for f in (0, 1, 2, 3):

            def old(sw, off):
                return (
                    storage((2, s, 2, width), dt, off).transpose(1, 2)
                    if sw
                    else storage((2, 2, s, width), dt, off)
                )

            k, v = old(f & 1, int(f == 1)), old(f & 2, 2 * int(f == 2))
            n = storage((2, 2, 1, width), dt)
            xs = (k, n, v, n)
            ref = torch.cat((k, n), 2), torch.cat((v, n), 2)
            other.wait_stream(current)
            with torch.cuda.stream(other):
                y = op(*xs)
                assert all(z._base is None for z in y)
            current.wait_stream(other)
            assert all(
                _bits(a, b) and a.stride() == b.stride() for a, b in zip(y, ref, strict=True)
            )
            live.extend(y)
    assert len({x.untyped_storage().data_ptr() for x in live}) == len(live)
    k = storage((2, 2, 31, width * 2), dt)[..., ::2]
    v = k
    n = storage((2, 2, 1, width), dt)
    ref = torch.cat((k, n), 2)
    a, b = op(k, n, v, n)
    assert _bits(a, ref) and _bits(b, ref) and a.data_ptr() != b.data_ptr()
    for wrong in (
        (k[:, :, :1], n, v, n),
        (k, n, v[:, :, :27], n),
        (k, n[:, :, :, :32], v, n),
        (k, n, v, n.cpu()),
    ):
        with pytest.raises(ValueError):
            op(*wrong)


@pytest.mark.parametrize("expose", [True, False])
@torch.inference_mode()
def test_append_keeps_public_cache_and_extra_readers(expose):
    a, b = req(9), req(51)
    m = Step(public_repeat=expose).to(device="cuda", dtype=torch.bfloat16).eval()
    ep = export(m, a)
    base = str(ep.graph)
    p = partition_hybrid_decode(ep, a, b, paired_append=True)
    (r,) = p.appends
    assert "module" in r and r["past"] == [2, 129]
    body = p.module.get_submodule(r["module"])
    ret = []
    for s in (2, 31, 64, 129):
        xs = req(s)
        y = p.module(*xs)
        ref = m(*xs)
        torch.testing.assert_close(y[0], ref[0], atol=0.012, rtol=0.02, check_stride=True)
        assert all(
            _bits(x, z) and x.stride() == z.stride() for x, z in zip(y[1:], ref[1:], strict=True)
        )
        ret.extend(y[1:3])
    assert (
        len({v.data_ptr() for v in ret}) == len(ret)
        and str(ep.graph) == base
        and body.path.is_file()
    )


@torch.inference_mode()
def test_append_second_trace_rejection_leaves_both_aten_nodes(monkeypatch):
    a, b = req(9), req(51)
    m = Step().to(device="cuda", dtype=torch.bfloat16).eval()
    ep = export(m, a)
    orig = Append.forward
    calls = []

    def corrupt(self, *args):
        ys = orig(self, *args)
        calls.append(1)
        return ys if len(calls) != 2 else (ys[0], ys[1] + 1)

    monkeypatch.setattr(Append, "forward", corrupt)
    p = partition_hybrid_decode(ep, a, b, paired_append=True)
    (r,) = p.appends
    assert (
        len(calls) == 2
        and "fallback" in r
        and sum(n.target is torch.ops.aten.cat.default for n in p.module.graph.nodes) == 2
    )
    xs = req(77)
    assert all(_bits(y, z) for y, z in zip(p.module(*xs)[1:], m(*xs)[1:], strict=True))
