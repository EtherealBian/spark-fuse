"""Static strided-cache reads must never turn a public view into an allocation or mutate history."""

import pytest
import torch
from torch import nn
from sparkfuse.partition import partition_export


class Update(nn.Module):
    def __init__(self, d, r, mode="public"):
        super().__init__()
        self.d = d
        self.r = r
        self.mode = mode

    def forward(self, x, c, s, pk, pv):
        b, t, _ = x.shape
        d, r = self.d, self.r
        off = 2

        def raw(st, heads):
            return x[..., st : st + heads * d].view(b, t, heads, d).transpose(1, 2)

        c, s = c.unsqueeze(1), s.unsqueeze(1)

        def rotate(h):
            u = h[..., :r]
            rot = torch.cat((-u[..., r // 2 :], u[..., : r // 2]), -1)
            return torch.cat((u * c + rot * s, h[..., r:]), -1)

        k0 = raw(off + 3 * d + 1, 2)
        q0 = raw(off, 3)
        v = raw(off + (3 * d + 1 if self.mode == "overlap" else 5 * d + 3), 2)
        q, k = rotate(q0), rotate(k0)
        kc = torch.cat((pk, k), 2)
        vc = torch.cat((pv, v), 2)
        if self.mode == "live_k":
            return q, kc, vc, k
        return q, kc, vc, v, k0, q0[..., self.r :]


def it(x):
    return torch.int32 if x.dtype == torch.float32 else torch.int16


def data(dt, t, p, d, r, physical, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def gen(*shape):
        return torch.randn(*shape, dtype=dt, device="cuda", generator=g)

    wide = gen(2, t, 7 * d + 11)
    for st, h in ((2, 3), (3 * d + 3, 2), (5 * d + 5, 2)):
        z = wide[..., st : st + d * h].view(2, t, h, d)
        z[..., r:].view(it(z)).fill_(-1)
    wide[..., 5 * d + 5 : 7 * d + 5].view(it(wide)).fill_(-1)
    c, s = gen(1, t, r), gen(1, t, r)
    if physical == "holes":
        pk = gen(2, p, 2, 2 * d + 1)[..., 1 : d + 1].transpose(1, 2)
        pv = gen(2, p, 4 * d + 5)[..., 2 : 2 + 2 * d].view(2, p, 2, d).transpose(1, 2)
    else:
        pk, pv = gen(2, 2, p, d), gen(2, 2, p, d)
    for z in (pk, pv):
        z.view(it(z))[..., ::3] = -1
        z.view(it(z))[..., 1::3] = torch.iinfo(it(z)).min
    return wide, c, s, pk, pv


def ptr(x):
    return x.untyped_storage().data_ptr()


def check(g, f, args):
    before = [x.contiguous().view(it(x)).clone() for x in args]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        y = g(*args)
        ref = f(*args)
    torch.cuda.current_stream().wait_stream(stream)
    for x, z in zip(y, ref, strict=True):
        assert (x.stride(), x.storage_offset(), x.untyped_storage().nbytes()) == (
            z.stride(),
            z.storage_offset(),
            z.untyped_storage().nbytes(),
        )
        assert torch.equal(x.contiguous().view(it(x)), z.contiguous().view(it(z)))
    assert all(ptr(x) == ptr(args[0]) for x in y[3:])
    assert len({ptr(x) for x in y[:3]}) == 3 and not {ptr(x) for x in args} & {
        ptr(x) for x in y[:3]
    }
    for x in y[:3]:
        x.zero_()
    assert all(
        torch.equal(x.contiguous().view(it(x)), z) for x, z in zip(args, before, strict=True)
    )


@pytest.mark.parametrize("host", ["python", "native"])
@pytest.mark.parametrize("dt", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "t,p,d,r,physical",
    [(1, 127, 16, 16, "holes"), (3, 17, 14, 8, "holes"), (2, 31, 13, 6, "dense")],
)
@torch.inference_mode()
def test_exact_strided_history(host, dt, t, p, d, r, physical):
    args = data(dt, t, p, d, r, physical, 7397)
    fn = Update(d, r)
    g, records = partition_export(
        torch.export.export(fn, args), sliced_cache=True, sliced_rotary=True, host_backend=host
    )
    (hit,) = [v for v in records if v.get("kind") == "sliced_cache_update"]
    assert not any(v.get("kind") == "sliced_rotary_pair" for v in records)
    assert hit["history_strides"] == [list(z.stride()) for z in args[-2:]] and hit["prefix"] == p
    assert hit["offsets"] == [2, 3 * d + 3, 5 * d + 5]
    g.graph.lint()
    check(g, fn, data(dt, t, p, d, r, physical, 7603))
    if physical == "holes":
        call = g.get_submodule(hit["module"])
        x, c, s, k, v = args
        for kk, vv in ((k.contiguous(), v), (k, v.contiguous()), (k, torch._neg_view(v))):
            with pytest.raises(ValueError, match="mismatch"):
                call(x, c.unsqueeze(1), s.unsqueeze(1), kk, vv)


@pytest.mark.parametrize("mode", ["overlap", "live_k", "off"])
@torch.inference_mode()
def test_nonmatch(mode):
    fn = Update(16, 8, mode)
    args = data(torch.bfloat16, 2, 7, 16, 8, "holes", 8339)
    g, rs = partition_export(
        torch.export.export(fn, args),
        sliced_cache=mode != "off",
        sliced_rotary=True,
        host_backend="native",
    )
    assert not any(z.get("kind") == "sliced_cache_update" for z in rs)
    g.graph.lint()
    for a, b in zip(g(*args), fn(*args), strict=True):
        assert a.stride() == b.stride() and torch.equal(
            a.contiguous().view(it(a)), b.contiguous().view(it(b))
        )


@pytest.mark.parametrize("host", ["native", "python"])
@torch.inference_mode()
def test_cache_graph_replay_replaces_history(host):
    fn = Update(16, 8)

    def make(seed):
        return data(torch.bfloat16, 2, 11, 16, 8, "holes", seed)

    args = make(7817)
    gm, records = partition_export(
        torch.export.export(fn, args), sliced_cache=True, host_backend=host
    )
    assert sum(z.get("kind") == "sliced_cache_update" for z in records) == 1
    gm(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = gm(*args)
    for seed in (997, 521):
        nxt = make(seed)
        for a, n in zip(args, nxt, strict=True):
            a.copy_(n)
        graph.replay()
        for y, ref in zip(out, fn(*args), strict=True):
            assert torch.equal(y.contiguous().view(it(y)), ref.contiguous().view(it(ref)))
        assert all(ptr(v) == ptr(args[0]) for v in out[3:])


@pytest.mark.parametrize("strides", [[], [()], [(3, -1)], [(2**29, 1)], [(1, 1, 1)]])
def test_fixed_descriptor_rejects_invalid_before_compile(strides):
    from sparkfuse.index_program import IndexProgram

    with pytest.raises(ValueError, match="stride"):
        IndexProgram([(2, 2)], [(2, 2)], "bf16", "cuda", "", input_strides=strides)
