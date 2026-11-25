"""Observable public repeats/aliases, multiple sites, narrow bias and real runtime guards."""

import pytest
import torch
from torch.nn import functional as F
from sparkfuse.decode_sites import partition_decode_sites
from sparkfuse.decode_attention import compile_decode_attention
from sparkfuse.ir import Unsupported
from examples.single_decode import SingleDecode


def req(s=65, dtype=torch.bfloat16):
    q, k, v = [
        torch.randn(shape, device="cuda", dtype=dtype)
        for shape in ((2, 8, 1, 64), (2, 2, s, 64), (2, 2, s, 64))
    ]
    bias = torch.randn(2, 1, 1, s, device="cuda", dtype=dtype)
    bias[0].fill_(-torch.inf)
    bias[1, ..., ::7] = -torch.inf
    return q, k, v, bias


class Graph(torch.nn.Module):
    def __init__(self, expose="none", second=False):
        super().__init__()
        self.expose = expose
        self.second = second

    def forward(self, q, k, v, bias):
        b, g, s, d = k.shape

        def rep(x):
            return x.unsqueeze(2).expand(b, g, 4, s, d).reshape(b, g * 4, s, d)

        rk, rv = rep(k), rep(v)
        y = F.scaled_dot_product_attention(q, rk, rv, bias)
        if self.second:
            y = F.scaled_dot_product_attention(y, rk, rv, bias)
        return {
            "y": (y, y.view_as(y)),
            "cache": (k, v),
            "visible": rk if self.expose == "k" else rv if self.expose == "v" else q,
        }


@pytest.mark.parametrize(
    "expose,second,elide",
    [
        ("none", False, False),
        ("none", False, True),
        ("k", False, True),
        ("v", True, True),
        ("none", True, True),
    ],
)
@torch.inference_mode()
def test_extract_sites_observable_tree(expose, second, elide):
    model = Graph(expose, second)
    ex, au, held = req(), req(), req()
    ep = torch.export.export(model, ex)
    before = str(ep.graph)
    res = partition_decode_sites(ep, ex, au, elide_kv_repeats=elide, head_group=4 if elide else 1)
    assert len(res.sites) == 1 + second and not res.skipped and before == str(ep.graph)
    assert sum(len(r["erased_repeats"]) for r in res.sites) == (
        2 - (expose != "none") if elide else 0
    )
    assert (
        sum(r["removed_materialized_bytes"] for r in res.sites)
        == (2 - (expose != "none")) * 2 * 8 * 65 * 64 * 2 * elide
    )
    orig = [x.clone() for x in held]
    out = res.module(*held)
    ref = model(*held)
    assert list(out) == list(ref)
    for x, z in zip(torch.utils._pytree.tree_leaves(out), torch.utils._pytree.tree_leaves(ref)):
        torch.testing.assert_close(x, z, atol=0.012, rtol=0.02, check_stride=True)
    assert torch.equal(out["visible"], ref["visible"])
    assert torch.count_nonzero(out["y"][0][0]) == 0
    assert (
        out["y"][0].data_ptr() == out["y"][1].data_ptr()
        and out["cache"][0] is held[1]
        and out["cache"][1] is held[2]
    )
    assert all(torch.equal(x, z) for x, z in zip(orig, held))
    held[0][0, 0, 0, 0] = 65
    with pytest.raises(ValueError, match="numerical domain"):
        res.module(*held)


@pytest.mark.parametrize(
    "dt,compact,r",
    [
        (torch.float16, True, 2),
        (torch.bfloat16, True, 4),
        (torch.float16, False, 1),
        (torch.bfloat16, False, 1),
    ],
)
@torch.inference_mode()
def test_direct_narrow_bias_independent_fp64(dt, compact, r):
    def args():
        q, k, v, m = req(129, dt)
        if not compact:
            k, v = [t.repeat_interleave(4, 1) for t in (k, v)]
        return q, k, v, m

    x, a, held = args(), args(), args()
    model = SingleDecode(compact)
    ep = torch.export.export(model, x)
    f = compile_decode_attention(ep, x, a, tile=128, head_group=r)
    y = f.run_bounded(*held)
    q, k, v, m = [t.cpu().double() for t in held]
    ys = []
    for h in range(q.shape[1]):
        j = h // (q.shape[1] // k.shape[1])
        w = (q[:, h] @ k[:, j].mT / 8 + m[:, 0]).softmax(-1).nan_to_num()
        ys.append(w @ v[:, j])
    ref = torch.stack(ys, 1)
    torch.testing.assert_close(
        y.cpu().double(),
        ref,
        rtol=0.012 if dt == torch.bfloat16 else 0.003,
        atol=0.006 if dt == torch.bfloat16 else 0.001,
    )
    assert not torch.count_nonzero(y[0])
    assert y.stride() == model(*held).stride()


@torch.inference_mode()
def test_nonmatching_admission_and_no_effects():
    ex, au = req(), req()
    model = Graph()
    ep = torch.export.export(model, ex)
    with pytest.raises(Unsupported, match="independent"):
        partition_decode_sites(ep, ex, ex)
    with pytest.raises(Unsupported):
        partition_decode_sites(ep, ex, au, domain="guess")
    res = partition_decode_sites(ep, ex, au, tile=17)
    assert (
        not res.sites
        and len(res.skipped) == 1
        and not any(
            n.op == "call_module" and n.target != "_guards_fn" for n in res.module.graph.nodes
        )
    )

    class Drop(torch.nn.Module):
        def forward(self, q, k, v, m):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.1, enable_gqa=True)

    ep = torch.export.export(Drop(), ex)
    rng = torch.cuda.get_rng_state().clone()
    with pytest.raises(Unsupported, match="dropout"):
        partition_decode_sites(ep, ex, au)
    assert torch.equal(rng, torch.cuda.get_rng_state())

    class Mutate(torch.nn.Module):
        def forward(self, q, k, v, m):
            q.add_(1)
            return q

    ep = torch.export.export(Mutate(), ex)
    before = ex[0].clone()
    with pytest.raises(Unsupported):
        partition_decode_sites(ep, ex, au)
    assert torch.equal(before, ex[0])
