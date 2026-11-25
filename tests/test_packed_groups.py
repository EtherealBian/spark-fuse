"""Multi-cohort compaction is all-or-nothing; public intermediates never become band views."""

import pytest
import torch
from torch import nn
from examples.token_quant import quantize
from sparkfuse.ir import Unsupported
from sparkfuse.resident import capture_export


class Stage(nn.Module):
    def __init__(self, widths, g, mode):
        super().__init__()
        self.mode = mode
        self.widths = widths
        for j, n in enumerate(widths):
            w = torch.randint(-127, 128, (n, 128), generator=g, device="cuda", dtype=torch.int8).t()
            self.register_buffer(f"w{j}", w if j % 2 == 0 else w.contiguous())
            self.register_buffer(f"c{j}", torch.rand(n, generator=g, device="cuda") * 0.005)
            self.register_buffer(f"b{j}", torch.randn(n, generator=g, device="cuda"))
        if mode == "extra":
            self.register_buffer("extra", self.w0.clone())

    def forward(self, x, r):
        z = x + r
        q, s = quantize(z)
        lhs = q.reshape(-1, 128)
        row = s.reshape(-1, 1)
        out = []
        for j in range(len(self.widths)):
            a = torch._int_mm(lhs, getattr(self, f"w{j}"))
            early = a.clone() if self.mode == "early" and j == 0 else a
            t = a.float() * row
            v = t * getattr(self, f"c{j}")
            narrow = (v + getattr(self, f"b{j}")).to(x.dtype)
            out.extend((early, t, narrow))
        if self.mode == "extra":
            out.append(torch._int_mm(lhs, self.extra))
        return tuple(out)


class Stack(nn.Module):
    def __init__(self, widths, g, mode="ok"):
        super().__init__()
        self.stages = nn.ModuleList(
            [Stage(w, g, mode if i == len(widths) - 1 else "ok") for i, w in enumerate(widths)]
        )

    def forward(self, x, r):
        out = []
        h = x
        for s in self.stages:
            y = s(h, r)
            h = y[2]
            out.extend(y)
        return tuple(out)


def workload(widths, mode="ok"):
    g = torch.Generator(device="cuda").manual_seed(2023379 + len(widths))
    model = Stack(widths, g, mode).eval()
    req = [
        tuple(
            torch.randn(17, 128, dtype=torch.bfloat16, device="cuda", generator=g) for _ in range(2)
        )
        for _ in range(4)
    ]
    return model, req


def opts():
    return dict(
        request_backend="native",
        int8_rhs_layout="packed_groups",
        partition_options=dict(
            token_quant=True,
            group_epilogues=True,
            affine_epilogues=True,
            projection_epilogue=True,
            host_backend="native",
            min_ops=100000,
        ),
    )


def identical(a, b):
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and a.stride() == b.stride()
        and a.storage_offset() == b.storage_offset()
        and torch.equal(
            a.contiguous().reshape(-1).view(torch.uint8),
            b.contiguous().reshape(-1).view(torch.uint8),
        )
    )


@pytest.mark.parametrize(
    "widths",
    [
        ((128, 8), (128, 24)),
        ((128, 8), (128, 16, 32), (128, 24, 8, 40)),
        ((128, 8),) * 4,
        ((128, 8),) * 8,
    ],
)
@torch.inference_mode()
def test_multigroup_snapshot_preserves_intermediates_and_source_isolation(widths):
    model, req = workload(widths)
    cap = sum(t.nbytes for t in (*req[0], *model.buffers()))
    ep = torch.export.export(model, req[0])
    before = str(ep.graph)
    refs = [model(*args) for args in req]
    h = req[-1][0]
    offset = 0
    for layer in model.stages:
        codes = quantize(h + req[-1][1])[0].cpu().int()
        outs = refs[-1][offset:]
        h = outs[2]
        for i in range(len(layer.widths)):
            assert torch.equal(outs[i * 3].cpu(), codes @ getattr(layer, f"w{i}").cpu().int())
        offset += len(layer.widths) * 3
    owner = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(owner)
    with (
        torch.cuda.stream(stream),
        capture_export(ep, *req[:2], max_static_bytes=cap, **opts()) as fn,
    ):
        assert fn.metadata["staging_and_state_bytes"] == cap and str(ep.graph) == before
        records = [r for r in fn.metadata["regions"] if r.get("kind") == "packed_int32_projection"]
        assert len(records) == len(widths)
        assert sum(n.target == torch.ops.aten._int_mm.default for n in fn._fn.graph.nodes) == len(
            widths
        )
        weights = [t for t in fn._fn.buffers() if t.dtype == torch.int8]
        assert len(weights) == len(widths)
        for j, (r, ns, layer) in enumerate(zip(records, widths, model.stages, strict=True)):
            assert (
                r["module"] == f"_sparkfuse_packed_projection_{j}"
                and r["weight"] == f"_sparkfuse_packed_rhs_{j}"
                and r["shape"] == [17, 128, sum(ns)]
            )
            w = fn._fn.get_buffer(r["weight"])
            assert w.stride() == (1, 128) and w.untyped_storage().nbytes() == w.nbytes
            for i, (band, desc) in enumerate(
                zip(w.split(ns, dim=1), r["source_weights"], strict=True)
            ):
                src = getattr(layer, f"w{i}")
                assert (
                    torch.equal(band, src)
                    and band.untyped_storage().data_ptr() != src.untyped_storage().data_ptr()
                    and desc["packed_column_offset"] == sum(ns[:i])
                )
                src.zero_()
        ys = [fn(*args) for args in req]
        ptr = [t.untyped_storage().data_ptr() for y in ys for t in y]
        assert len(ptr) == len(set(ptr))
    owner.wait_stream(stream)
    for actual, ref in zip(ys, refs, strict=True):
        assert all(identical(a, b) for a, b in zip(actual, ref, strict=True))


@pytest.mark.parametrize(
    "why",
    [
        "single",
        "nine",
        "cap",
        "uncovered",
        "tied",
        "early_last",
        "extra_last",
        "legacy",
        "backend",
        "requires_options",
    ],
)
@torch.inference_mode()
def test_all_or_nothing_before_first_pack(why, monkeypatch):
    import sparkfuse.packed_projection as pack

    n = {"single": 1, "nine": 9}.get(why, 2)
    ws = [(128, 8, 16)] * n
    if why == "uncovered":
        ws[-1] = (128,)
    mode = {"early_last": "early", "extra_last": "extra"}.get(why, "ok")
    model, req = workload(ws, mode)
    if why == "tied":
        model.stages[-1].w0 = model.stages[0].w0.view_as(model.stages[0].w0)
    opt = opts()
    if why == "cap":
        opt["max_static_bytes"] = sum(t.nbytes for t in (*req[0], *model.buffers())) - 1
    if why == "legacy":
        opt["int8_rhs_layout"] = "packed_fanout"
    if why == "backend":
        opt.update(backend="aten", partition_options=None)
    if why == "requires_options":
        opt["partition_options"].pop("group_epilogues")
    ep = torch.export.export(model, req[0])
    original = str(ep.graph)
    ref = model(*req[1])
    monkeypatch.setattr(pack, "_apply_group", lambda *a: pytest.fail("materialized partial group"))
    with pytest.raises((Unsupported, ValueError)):
        capture_export(ep, *req[:2], **opt)
    assert original == str(ep.graph) and all(
        identical(a, b) for a, b in zip(ep.module()(*req[1]), ref, strict=True)
    )


@pytest.mark.parametrize(
    "fault,decline",
    [
        (
            RuntimeError(
                "CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling cublasLtMatmul with M=17"
            ),
            True,
        ),
        (
            RuntimeError("CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling cublasLtMatmul"),
            False,
        ),
        (torch.cuda.OutOfMemoryError("CUDA out of memory"), False),
        (RuntimeError("CUBLAS_STATUS_NOT_SUPPORTED in an unrelated operation"), False),
    ],
)
def test_only_specific_library_capability_miss_is_a_rejection(monkeypatch, fault, decline):
    import sparkfuse.resident as mod

    def fail(*args, **kwargs):
        raise fault

    monkeypatch.setattr(mod, "_capture_export", fail)
    with pytest.raises(Unsupported if decline else type(fault)) as exc:
        mod.capture_export(None, (), (), int8_rhs_layout="packed_groups")
    assert (exc.value.__cause__ is fault) if decline else (exc.value is fault)
    with pytest.raises(type(fault)) as original:
        mod.capture_export(None, (), (), int8_rhs_layout="packed_fanout")
    assert original.value is fault


@torch.inference_mode()
def test_backend_specific_packed_width_72_is_probed_not_assumed():
    # GB10 can accept source widths 64/8 but reject a column-major width of 72 at K=64.
    # Other library/device versions may implement it. Both outcomes have a defined contract.
    g = torch.Generator(device="cuda").manual_seed(2041139)

    class Small(nn.Module):
        def __init__(self):
            super().__init__()
            self.stages = nn.ModuleList([Stage((64, 8), g, "ok") for _ in range(2)])
            for layer in self.stages:
                for j, n in enumerate((64, 8)):
                    setattr(
                        layer,
                        f"w{j}",
                        torch.randint(
                            -127, 128, (n, 64), device="cuda", dtype=torch.int8, generator=g
                        ).t(),
                    )

        def forward(self, x, r):
            outs = []
            for layer in self.stages:
                q, s = quantize(x + r)
                codes = q.reshape(-1, 64)
                row = s.reshape(-1, 1)
                one = []
                for j in range(2):
                    acc = torch._int_mm(codes, getattr(layer, f"w{j}"))
                    v = (acc.float() * row) * getattr(layer, f"c{j}")
                    one.append((v + getattr(layer, f"b{j}")).to(x.dtype))
                x = one[0]
                outs.extend(one)
            return tuple(outs)

    model = Small().eval()
    req = [
        tuple(
            torch.randn(17, 64, device="cuda", dtype=torch.bfloat16, generator=g) for _ in range(2)
        )
        for _ in range(3)
    ]
    ep = torch.export.export(model, req[0])
    ref = model(*req[-1])
    a = torch.ones(17, 64, device="cuda", dtype=torch.int8)
    w = torch.ones(72, 64, device="cuda", dtype=torch.int8).t()
    try:
        torch._int_mm(a, w)
    except RuntimeError as exc:
        if "CUBLAS_STATUS_NOT_SUPPORTED when calling cublasLtMatmul" not in str(exc):
            raise
        with pytest.raises(Unsupported, match="cuBLASLt") as reason:
            capture_export(ep, *req[:2], **opts())
        assert isinstance(reason.value.__cause__, RuntimeError)
    else:
        with capture_export(ep, *req[:2], **opts()) as fn:
            assert all(identical(a, b) for a, b in zip(fn(*req[-1]), ref, strict=True))
