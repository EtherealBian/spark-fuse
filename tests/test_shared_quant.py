"""Plain activation quantization preserves input identity; shared private codes must never escape."""

import pytest
import torch
from torch import nn
from examples.bert_quant import TokenLinear
from examples.token_quant import quantize
from sparkfuse.partition import partition_export
from sparkfuse.resident import capture_export
from sparkfuse.shared_quant import share


def args(m=33, k=128, dt=torch.bfloat16, seed=211):
    return (
        torch.randn(
            1,
            m,
            k,
            device="cuda",
            dtype=dt,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ),
    )


def exact(xs, ys):
    assert len(xs) == len(ys)
    for a, b in zip(xs, ys, strict=True):
        assert (a.dtype, a.shape, a.stride(), a.storage_offset()) == (
            b.dtype,
            b.shape,
            b.stride(),
            b.storage_offset(),
        ) and torch.equal(
            a.contiguous().reshape(-1).view(torch.uint8),
            b.contiguous().reshape(-1).view(torch.uint8),
        )


@pytest.mark.parametrize("host", ["native", "python"])
@pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_plain_activation_special_values_and_passthrough(host, dt):
    class Model(nn.Module):
        def forward(self, x):
            q, s = quantize(x)
            return x, x[..., 1:3], q, s, q

    m = Model()
    (x,) = args(k=257, dt=dt)
    x[0, 0, 0] = float("nan")
    x[0, 1, 0] = float("inf")
    x[0, 2, 0] = -float("inf")
    x[0, 3].zero_()
    x[0, 3, 0] = -0.0
    x[0, 4, -1] = 127.0
    x[0, 4, :5] = torch.tensor([2.5, 3.5, -2.5, -3.5, 1e-7], device="cuda", dtype=dt)
    ep = torch.export.export(m, (x,))
    run, recs = partition_export(ep, activation_quant=True, host_backend=host)
    (r,) = [r for r in recs if r.get("kind") == "activation_token_quant"]
    assert r["output_roles"] == ["codes", "scale"] and len(r["live_ins"]) == 1
    out = run(x)
    exact(out, m(x))
    assert out[0] is x and out[1].untyped_storage().data_ptr() == x.data_ptr() and out[2] is out[4]
    gm, recs = partition_export(ep, host_backend=host)
    assert not any(r.get("kind") == "activation_token_quant" for r in recs)


class Siblings(nn.Module):
    def __init__(self, n, distinct=False):
        super().__init__()
        self.layers = nn.ModuleList(
            TokenLinear(nn.Linear(128, 32 + i * 8, device="cuda", dtype=torch.bfloat16))
            for i in range(n)
        )
        self.distinct = distinct

    def forward(self, x):
        return tuple(
            linear(x + i if self.distinct else x).transpose(0, 1)
            for i, linear in enumerate(self.layers)
        )


OPTS = dict(
    activation_quant=True,
    share_activation_quant=True,
    token_quant=True,
    affine_epilogues=True,
    group_epilogues=True,
    host_backend="native",
    min_ops=100000,
)


@pytest.mark.parametrize("count", [2, 3, 4])
@torch.inference_mode()
def test_share_quant_modules_and_single_packed_gemm(count):
    class Flat(Siblings):
        def forward(self, x):
            return tuple(linear(x) for linear in self.layers)

    m = Flat(count).eval()
    a = args()
    b = args(seed=113)
    ep = torch.export.export(m, a)
    gm, rs = partition_export(ep, **OPTS)
    (r,) = (r for r in rs if r.get("kind") == "shared_hidden_token_quantizers")
    assert (
        r["eliminated"] == count - 1
        and sum(r.get("kind") == "activation_token_quant" for r in rs) == 1
    )
    with capture_export(
        ep, a, b, int8_rhs_layout="packed_fanout", partition_options=OPTS, request_backend="native"
    ) as exe:
        assert any(
            r.get("kind") == "packed_int32_projection" and len(r["branches"]) == count
            for r in exe.metadata["regions"]
        )
        for seed in (413, 509, 991):
            h = args(seed=seed)
            out = exe(*h)
            exact(out, m(*h))
            exact(gm(*h), out)
            assert len({v.data_ptr() for v in out}) == count


@pytest.mark.parametrize("count,distinct", [(1, False), (5, False), (3, True)])
@torch.inference_mode()
def test_share_quant_reject_different_roots_or_arity(count, distinct):
    m = Siblings(count, distinct)
    a = args()
    gm = torch.export.export(m, a).module()
    assert not share(gm, "cuda")
    exact(gm(*a), m(*a))


@pytest.mark.parametrize("visible", ["codes", "scale", "code_view", "row_view"])
@torch.inference_mode()
def test_share_quant_never_unifies_public_storage(visible):
    class Expose(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.ModuleList(
                TokenLinear(nn.Linear(128, 32, device="cuda", dtype=torch.bfloat16))
                for _ in range(2)
            )

        def forward(self, x):
            res = []
            for linear in self.linear:
                q, s = quantize(x)
                v = q.reshape(-1, 128)
                row = s.reshape(-1, 1)
                mm = torch._int_mm(v, linear.rhs)
                n = ((mm.float() * row) * linear.channel + linear.bias).to(x.dtype)
                res.extend((n, {"codes": q, "scale": s, "code_view": v, "row_view": row}[visible]))
            return tuple(res)

    m = Expose()
    a = args()
    gm, rs = partition_export(torch.export.export(m, a), **OPTS)
    assert not any(r.get("kind") == "shared_hidden_token_quantizers" for r in rs)
    ys = gm(*a)
    exact(ys, m(*a))
    assert ys[1].data_ptr() != ys[3].data_ptr()


@pytest.mark.parametrize("barrier", ["guard", "rng", "unknown"])
@torch.inference_mode()
def test_share_quant_does_not_cross_unknown_ordering(barrier):
    a = args()
    m = Siblings(2)
    gm = torch.export.export(m, a).module()
    nodes = list(gm.graph.nodes)
    first = next(n for n in nodes if n.target is torch.ops.aten._int_mm.default)
    before = str(gm.graph)
    with gm.graph.inserting_after(first):
        if barrier == "guard":
            gm.graph.call_function(
                torch.ops.aten._assert_tensor_metadata.default,
                (first,),
                {"size": list(first.meta["val"].shape)},
            )
        elif barrier == "rng":
            gm.graph.call_function(torch.ops.aten.rand_like.default, (first,))
        else:
            gm.graph.call_function(torch.ops.aten.detach.default, (first,))
    original = str(gm.graph)
    assert original != before and not share(gm, "cuda") and str(gm.graph) == original


@torch.inference_mode()
def test_share_quant_opt_in_and_residual_recognition():
    class Residual(Siblings):
        def forward(self, x, r):
            return super().forward(x + r)

    a = args() + args(seed=993)
    m = Residual(3)
    ep = torch.export.export(m, a)
    gm, rs = partition_export(ep, **OPTS)
    assert sum(r.get("kind") == "residual_token_quant" for r in rs) == 1 and not any(
        r.get("kind") == "activation_token_quant" for r in rs
    )
    exact(gm(*a), m(*a))
    with pytest.raises(ValueError, match="requires activation_quant"):
        partition_export(ep, share_activation_quant=True)
