"""Single-shape frozen signed RHS layout: ownership, admission, staging and replay."""

import pytest
import torch
from torch import nn
from examples.token_quant import QuantProjection, quantize
from sparkfuse.resident import capture_export
from sparkfuse.ir import Unsupported


def requests(shape=(2, 16, 256), n=64, storage="row"):
    gen = torch.Generator(device="cuda").manual_seed(963551)
    args = [
        tuple(
            torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=gen) for _ in range(2)
        )
        for i in range(4)
    ]
    w = torch.randint(-127, 128, (shape[-1], n), device="cuda", dtype=torch.int8, generator=gen)
    w = w if storage == "row" else w.t().contiguous().t()
    c = torch.rand(1, n, device="cuda", generator=gen) * 0.03125 + 0.001
    return QuantProjection(w, c).eval(), args


def bits(x):
    return x.contiguous().reshape(-1).view(torch.uint8)


def exact(ys, want):
    assert len(ys) == len(want)
    for a, b in zip(ys, want, strict=True):
        assert (a.shape, a.dtype, a.stride(), a.storage_offset()) == (
            b.shape,
            b.dtype,
            b.stride(),
            b.storage_offset(),
        ) and torch.equal(bits(a), bits(b))


@pytest.mark.parametrize("backend", ["mixed", "aten"])
@pytest.mark.parametrize("layout", ["column_major", "preserve"])
@pytest.mark.parametrize("storage", ["row", "column"])
@torch.inference_mode()
def test_frozen_weight_content_staging_and_ownership(storage, layout, backend):
    m, args = requests(storage=storage)
    refs = [m(*a) for a in args]
    ep = torch.export.export(m, args[0])
    cap = sum(t.nbytes for t in (*args[0], *m.buffers()))
    opts = (
        {"host_backend": "native", "token_quant": True, "projection_epilogue": True}
        if backend == "mixed"
        else None
    )
    st = m.weight.stride()
    new = (1, m.weight.shape[0]) if layout == "column_major" else st
    with capture_export(
        ep,
        args[0],
        args[1],
        backend=backend,
        partition_options=opts,
        int8_rhs_layout=layout,
        max_static_bytes=cap,
    ) as run:
        meta = run.metadata
        assert meta["staging_and_state_bytes"] == cap and meta["int8_rhs_layout"] == layout
        (w,) = meta["frozen_int8_weights"]
        priv = getattr(run._fn, w["name"])
        assert (
            tuple(w["source_stride"]) == st
            and tuple(w["snapshot_stride"]) == new
            and priv.stride() == new
            and torch.equal(priv, m.weight)
            and priv.data_ptr() != m.weight.data_ptr()
        )
        assert m.weight.stride() == st
        m.weight.zero_()
        m.channel_scale.fill_(5.0)
        m.weight = m.weight.clone()
        ys = [run(*a) for a in args]
        assert not torch.equal(ys[-1][-1], m(*args[-1])[-1])
        stores = [t.untyped_storage().data_ptr() for row in ys for t in row]
        assert len(set(stores)) == len(stores)
        for actual, expected in zip(ys, refs, strict=True):
            exact(actual, expected)
    with pytest.raises(RuntimeError):
        run(*args[0])


@pytest.mark.parametrize(
    "reason",
    [
        "cap",
        "extra",
        "slice",
        "offset",
        "request_rhs",
        "alias",
        "formula",
        "invalid_policy",
        "default",
    ],
)
@torch.inference_mode()
def test_decline_unqualified_state_or_operators(reason):
    m, args = requests()
    w = m.weight
    cols = w.shape[1]

    class Bad(nn.Module):
        def __init__(self):
            super().__init__()
            v = w
            if reason == "slice":
                v = torch.empty(w.shape[0], cols * 2, device="cuda", dtype=torch.int8)[:, ::2]
                v.copy_(w)
            if reason == "offset":
                v = torch.zeros(w.numel() + 1, device="cuda", dtype=torch.int8)[1:].view_as(w)
                v.copy_(w)
            self.register_buffer("w", v)
            self.register_buffer("s", m.channel_scale)
            if reason == "alias":
                self.register_buffer("bytes", v.view(torch.uint8))

        def forward(self, x, r, *other):
            z = x + r
            q, s = quantize(z)
            b = other[0] if reason == "request_rhs" else self.w
            if reason == "formula":
                q = (z.float() / s).trunc().clamp(-127, 127).to(torch.int8)
            acc = torch._int_mm(q.reshape(-1, q.shape[-1]), b)
            y = (acc.float() * s.reshape(-1, 1)) * self.s
            return (
                (y, self.w) if reason == "extra" else (y, self.bytes) if reason == "alias" else (y,)
            )

    if reason == "request_rhs":
        args = [(*a, w.clone()) for a in args]
    bad = Bad().eval()
    ep = torch.export.export(bad, args[0])
    cap = sum(t.nbytes for t in (*args[0], *bad.buffers()))
    kwargs = {"int8_rhs_layout": "column_major"}
    if reason == "cap":
        kwargs["max_static_bytes"] = cap - 1
    if reason == "invalid_policy":
        kwargs["int8_rhs_layout"] = "always"
    if reason == "default":
        kwargs = {}
    with pytest.raises((ValueError, Unsupported)):
        capture_export(ep, args[0], args[1], backend="aten", **kwargs)


@pytest.mark.parametrize("tied_geometry", ["identical", "incompatible"])
@torch.inference_mode()
def test_multi_projection_tied_storage(tied_geometry):
    m, args = requests()
    w = m.weight

    class Two(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("a", w)
            self.register_buffer(
                "b",
                w.view_as(w)
                if tied_geometry == "identical"
                else w.view(w.shape[1], w.shape[0]).t(),
            )

        def forward(self, x, r):
            q, _ = quantize(x + r)
            lhs = q.reshape(-1, q.shape[-1])
            return torch._int_mm(lhs, self.a), torch._int_mm(lhs, self.b)

    mod = Two().eval()
    ep = torch.export.export(mod, args[0])
    cap = w.nbytes + sum(t.nbytes for t in args[0])
    opts = dict(backend="aten", max_static_bytes=cap, int8_rhs_layout="column_major")
    if tied_geometry == "incompatible":
        with pytest.raises(Unsupported, match="tied int8"):
            capture_export(ep, args[0], args[1], **opts)
        return
    with capture_export(ep, args[0], args[1], **opts) as run:
        inventory = run.metadata["frozen_int8_weights"]
        assert {x["name"] for x in inventory} == {"a", "b"}
        assert {x["storage_group"] for x in inventory} == {0}
        assert run._fn.a.data_ptr() == run._fn.b.data_ptr() != w.data_ptr()
        assert run.metadata["staging_and_state_bytes"] == cap
        refs = [mod(*a) for a in args]
        owned = run._fn.a.data_ptr()
        w.zero_()
        for a, ref in zip(args, refs, strict=True):
            exact(run(*a), ref)
            assert run._fn.a.data_ptr() == owned


@torch.inference_mode()
def test_extra_reduction_does_not_expand_general_allowlist():
    model, args = requests()

    class Extra(QuantProjection):
        def forward(self, x, r):
            return (*super().forward(x, r), x.abs().amax(0))

    ep = torch.export.export(Extra(model.weight, model.channel_scale).eval(), args[0])
    with pytest.raises(Unsupported, match="allowlist"):
        capture_export(ep, args[0], args[1], backend="aten", int8_rhs_layout="column_major")


@pytest.mark.parametrize("m,k,n", [(16, 256, 64), (17, 32, 64), (17, 72, 64), (17, 256, 33)])
@torch.inference_mode()
def test_reject_outside_common_signed_gemm_domain(m, k, n):
    model, req = requests((m, k), n)
    ep = torch.export.export(model, req[0])
    with pytest.raises(Unsupported, match="int_mm shape/stride"):
        capture_export(ep, req[0], req[1], backend="aten", int8_rhs_layout="column_major")


@torch.inference_mode()
def test_smallest_admitted_geometry():
    model, req = requests((17, 64), 8)
    ep = torch.export.export(model, req[0])
    with capture_export(ep, req[0], req[1], backend="aten", int8_rhs_layout="column_major") as run:
        out = run(*req[2])
        exact(out, model(*req[2]))
        assert torch.equal(out[3].cpu(), out[1].cpu().int() @ model.weight.cpu().int())
