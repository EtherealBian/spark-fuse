"""A packed, frozen weight may replace independent signed GEMMs, never visible output storage."""

import pytest
import torch
from torch import nn
from examples.token_quant import quantize
from sparkfuse.ir import Unsupported
from sparkfuse.resident import capture_export


class Branches(nn.Module):
    def __init__(self, widths, mode, gen, k=256):
        super().__init__()
        self.mode = mode
        self.count = len(widths)
        for i, n in enumerate(widths):
            w = torch.randint(-128, 128, (n, k), device="cuda", generator=gen, dtype=torch.int8).t()
            if i % 2:
                w = w.contiguous()
            self.register_buffer("w" + str(i), w)
            self.register_buffer(
                "c" + str(i),
                torch.rand((1, n) if i % 2 else (n,), device="cuda", generator=gen) * 0.03,
            )

    def forward(self, x, r):
        z = x + r
        q, s = quantize(z)
        a = q.reshape(-1, q.shape[-1])
        row = s.reshape(-1, 1)
        ys = [z, q, s]
        old = None
        for i in range(self.count):
            acc = torch._int_mm(a, getattr(self, "w" + str(i)))
            f = acc.float()
            b = f * row
            y = b * getattr(self, "c" + str(i))
            if self.mode == "all":
                ys.extend([acc, f, b, y])
            elif self.mode == "mixed" and i % 2 == 0:
                ys.extend([acc, b, y])
            elif self.mode == "alias" and i == 0:
                old = acc
                ys.extend([acc, y])
            else:
                ys.append(y)
        if self.mode == "alias":
            ys.extend([old, old[:, 1:]])
        return tuple(ys)


def inputs(widths=(64, 32, 16), mode="all", seed=1260361):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    req = [
        tuple(
            torch.randn(1, 17, 256, device="cuda", dtype=torch.bfloat16, generator=gen)
            for _ in range(2)
        )
        for _ in range(5)
    ]
    return Branches(widths, mode, gen).eval(), req


def options():
    return dict(
        int8_rhs_layout="packed_fanout",
        partition_options=dict(token_quant=True, group_epilogues=True, host_backend="native"),
    )


def eq(x, y):
    return (
        x.dtype == y.dtype
        and x.shape == y.shape
        and x.stride() == y.stride()
        and x.storage_offset() == y.storage_offset()
        and torch.equal(
            x.contiguous().reshape(-1).view(torch.uint8),
            y.contiguous().reshape(-1).view(torch.uint8),
        )
    )


@pytest.mark.parametrize(
    "widths,mode",
    [
        ((8, 16), "all"),
        ((64, 32, 16), "mixed"),
        ((64, 32, 16), "alias"),
        ((8, 24, 16, 32), "outputs"),
        ((8, 24, 16, 32), "all"),
    ],
)
@torch.inference_mode()
def test_signed_mm_fanout_preserves_live_roles_state_and_ownership(widths, mode):
    model, req = inputs(widths, mode)
    refs = [model(*r) for r in req]
    ep = torch.export.export(model, req[0])
    before = str(ep.graph)
    weights = [getattr(model, "w" + str(i)) for i in range(len(widths))]
    cap = sum(t.nbytes for t in (*req[0], *model.buffers()))
    owner = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(owner)
    with (
        torch.cuda.stream(stream),
        capture_export(ep, req[0], req[1], max_static_bytes=cap, **options()) as fn,
    ):
        assert str(ep.graph) == before
        assert fn.metadata["staging_and_state_bytes"] == cap
        (record,) = [
            r for r in fn.metadata["regions"] if r.get("kind") == "packed_int32_projection"
        ]
        assert record["shape"] == [17, 256, sum(widths)] and record["snapshot_stride"] == [1, 256]
        assert [x["packed_column_offset"] for x in record["source_weights"]] == [
            sum(widths[:i]) for i in range(len(widths))
        ]
        (weight,) = [t for t in fn._fn.buffers() if t.dtype == torch.int8]
        assert (
            weight.shape == (256, sum(widths))
            and weight.stride() == (1, 256)
            and weight._base is None
        )
        assert all(
            torch.equal(p, w) and p.untyped_storage().data_ptr() != w.untyped_storage().data_ptr()
            for p, w in zip(weight.split(widths, dim=1), weights, strict=True)
        )
        assert (
            len([n for n in fn._fn.graph.nodes if n.target == torch.ops.aten._int_mm.default]) == 1
        )
        if mode == "all":
            for ref in refs:
                for acc, w in zip(ref[3::4], weights, strict=True):
                    assert torch.equal(
                        acc.cpu(), ref[1].reshape(-1, 256).int().cpu() @ w.int().cpu()
                    )
        for i, w in enumerate(weights):
            w.zero_()
            setattr(model, "w" + str(i), torch.ones_like(w))
            getattr(model, "c" + str(i)).fill_(10)
        ys = [fn(*r) for r in req]
        assert any(not eq(a, b) for a, b in zip(ys[-1], model(*req[-1]), strict=True))
        ptrsets = []
        for out, ref in zip(ys, refs, strict=True):
            assert all(eq(x, y) for x, y in zip(out, ref, strict=True))
            ptrs = [t.untyped_storage().data_ptr() for t in out]
            if mode == "alias":
                assert out[-2] is out[3] and ptrs[-1] == ptrs[3] and len(set(ptrs)) == len(ptrs) - 2
            else:
                assert len(set(ptrs)) == len(ptrs)
            assert all(set(ptrs).isdisjoint(prev) for prev in ptrsets)
            ptrsets.append(set(ptrs))
    owner.wait_stream(stream)


@pytest.mark.parametrize(
    "why", ["cap", "tie", "one", "five", "aten", "options", "reused", "padded"]
)
@torch.inference_mode()
def test_packed_projection_declines_outside_contract(why):
    widths = (64,) * ({"one": 1, "five": 5}.get(why, 3))
    model, req = inputs(widths, "outputs")
    opts = options()
    if why == "tie":
        model.w1 = model.w0.view_as(model.w0)
    if why == "reused":
        model.w1 = model.w0
    if why == "padded":
        model.w1 = torch.zeros((256, 128), device="cuda", dtype=torch.int8)[:, ::2]
    if why == "cap":
        opts["max_static_bytes"] = sum(t.nbytes for t in (*req[0], *model.buffers())) - 1
    if why == "aten":
        opts.update(backend="aten", partition_options=None)
    if why == "options":
        opts["partition_options"].pop("group_epilogues")
    ep = torch.export.export(model, req[0])
    original = str(ep.graph)
    with pytest.raises((Unsupported, ValueError)):
        capture_export(ep, req[0], req[1], **opts)
    assert str(ep.graph) == original


@pytest.mark.parametrize("which", ["different_codes", "early", "extra_gemm"])
@torch.inference_mode()
def test_reject_noncohort_or_early_accumulator_user(which):
    base, req = inputs((64, 32), "all")

    class Other(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base
            self.register_buffer("extra", base.w0.clone())

        def forward(self, x, r):
            q, s = quantize(x + r)
            lhs = q.reshape(-1, 256)
            row = s.reshape(-1, 1)
            a = torch._int_mm(lhs, self.base.w0)
            early = a.clone() if which == "early" else a
            y = (a.float() * row) * self.base.c0
            b_lhs = quantize(r + x)[0].reshape(-1, 256) if which == "different_codes" else lhs
            b = torch._int_mm(b_lhs, self.base.w1)
            z = (b.float() * row) * self.base.c1
            extra = torch._int_mm(lhs, self.extra) if which == "extra_gemm" else b
            return early, y, z, extra

    model = Other().eval()
    ep = torch.export.export(model, req[0])
    ref = model(*req[0])
    with pytest.raises(
        Unsupported,
        match={
            "different_codes": "same signed codes",
            "early": "early accumulator",
            "extra_gemm": "group must consume all",
        }[which],
    ):
        capture_export(ep, req[0], req[1], **options())
    assert all(eq(x, y) for x, y in zip(ref, ep.module()(*req[0]), strict=True))


@pytest.mark.parametrize("nbranches", [2, 3, 4])
@torch.inference_mode()
def test_segmented_addressing_and_float_exceptions(nbranches):
    from sparkfuse.index_program import IndexProgram, IndexRegion
    from sparkfuse.packed_projection import emit

    m = 11
    ns = [73, 1, 129, 8][:nbranches]
    total = sum(ns)
    gen = torch.Generator(device="cuda").manual_seed(1431097 + nbranches)
    a = torch.randint(
        -(2**31), 2**31 - 1, (m * total + 3,), device="cuda", generator=gen, dtype=torch.int32
    )[3:].view(m, total)
    row = torch.randn(m + 1, device="cuda", generator=gen)[1:, None]
    cs = [torch.randn(n + 1, device="cuda", generator=gen)[1:] for n in ns]
    rs = [
        ("acc", "cast", "row_scaled", "output"),
        ("output",),
        ("acc", "row_scaled", "output"),
        ("cast", "output"),
    ][:nbranches]
    row[:6, 0] = torch.tensor(
        [0, -(2**31), 1, 0x7F800123, 0x7F800000, -8388608], device="cuda", dtype=torch.int32
    ).view(torch.float32)
    cs[0][:3] = torch.tensor(
        [0x7FC01234, -(2**31), 0x7F800000], device="cuda", dtype=torch.int32
    ).view(torch.float32)
    p = IndexProgram(
        [(m, 1), (m, total), *((n,) for n in ns)],
        [(m, n) for n, roles in zip(ns, rs, strict=True) for _ in roles],
        "f32",
        "cuda",
        emit([(m, n, roles) for n, roles in zip(ns, rs, strict=True)]),
        input_dtypes=["f32", "i32"] + ["f32"] * nbranches,
        output_dtypes=["i32" if role == "acc" else "f32" for roles in rs for role in roles],
    )
    reg = IndexRegion(p, "native")

    def ref():
        out = []
        for band, roles, c in zip(a.split(ns, dim=1), rs, cs, strict=True):
            v = band.contiguous()
            f = v.float()
            b = f * row
            y = b * c
            buf = {"acc": v, "cast": f, "row_scaled": b, "output": y}
            out.extend(buf[key] for key in roles)
        return out

    def check(out):
        assert len({t.untyped_storage().data_ptr() for t in out}) == len(out)
        assert all(eq(x, y) for x, y in zip(out, ref(), strict=True))

    owner = torch.cuda.current_stream()
    s = torch.cuda.Stream()
    s.wait_stream(owner)
    with torch.cuda.stream(s):
        out = reg(row, a, *cs)
        check(out)
        old = [t.clone() for t in out]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            static = reg(row, a, *cs)
        for j in range(3):
            a.add_(127)
            row[-1, 0] = j
            graph.replay()
            check(static)
            assert all(eq(x, y) for x, y in zip(out, old, strict=True))
    owner.wait_stream(s)
