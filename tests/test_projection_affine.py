"""Bias and the explicit narrow conversion are part of the exported numerical contract."""

import pytest
import torch
from torch import nn
from examples.quant_affine import QuantAffineFanout
from sparkfuse.index_program import IndexProgram, IndexRegion
from sparkfuse.partition import partition_export
from sparkfuse.projection_affine import emit
from sparkfuse.resident import capture_export
from sparkfuse.ir import Unsupported

OPTS = dict(token_quant=True, group_epilogues=True, affine_epilogues=True, host_backend="native")


def bits(x, y):
    return (
        x.shape == y.shape
        and x.dtype == y.dtype
        and x.stride() == y.stride()
        and x.storage_offset() == y.storage_offset()
        and torch.equal(
            x.contiguous().reshape(-1).view(torch.uint8),
            y.contiguous().reshape(-1).view(torch.uint8),
        )
    )


def fixture(ns, debug):
    gen = torch.Generator(device="cuda").manual_seed(1624499 + len(ns))
    k = 128
    ws = [
        torch.randint(-128, 128, (n, k), dtype=torch.int8, generator=gen, device="cuda").t()
        for n in ns
    ]
    cs = [torch.rand(n, device="cuda", generator=gen) * 0.02 for n in ns]
    bs = [
        torch.randn((1, n) if i % 2 else (n,), device="cuda", generator=gen)
        for i, n in enumerate(ns)
    ]
    dts = [torch.float16 if i % 2 else torch.bfloat16 for i in range(len(ns))]
    model = QuantAffineFanout(ws, cs, bs, dts, debug_outputs=debug).eval()
    req = [
        tuple(
            torch.randn(17, k, dtype=torch.bfloat16, device="cuda", generator=gen) for _ in range(2)
        )
        for _ in range(5)
    ]
    return model, req


@pytest.mark.parametrize(
    "ns,debug",
    [((8, 16), False), ((64, 32, 16), True), ((32, 8, 16, 24), False), ((8, 16, 24, 32), True)],
)
@torch.inference_mode()
def test_affine_packed_snapshot_exact_live_roles_fresh_ownership(ns, debug):
    model, req = fixture(ns, debug)
    ep = torch.export.export(model, req[0])
    fx = str(ep.graph)
    ws = [getattr(model, f"weight_{i}") for i in range(len(ns))]
    refs = [model(*a) for a in req]
    cap = sum(t.nbytes for t in (*req[0], *model.buffers()))
    gm, records = partition_export(ep, **OPTS)
    (group,) = (r for r in records if r.get("kind") == "grouped_int32_affine")
    expected = ["cast", "row_scaled", "output", "biased", "narrow"] if debug else ["narrow"]
    assert all(b["output_roles"] == expected for b in group["branches"])
    assert all(bits(x, y) for x, y in zip(gm(*req[2]), refs[2], strict=True))
    with capture_export(
        ep,
        *req[:2],
        int8_rhs_layout="packed_fanout",
        partition_options=OPTS,
        request_backend="native",
        max_static_bytes=cap,
    ) as f:
        (r,) = (r for r in f.metadata["regions"] if r.get("kind") == "packed_int32_projection")
        assert all(
            b["output_roles"] == ((["acc"] if debug else []) + expected) for b in r["branches"]
        )
        assert [b["narrow_dtype"] for b in r["branches"]] == [
            "f16" if i % 2 else "bf16" for i in range(len(ns))
        ]
        assert f.metadata["staging_and_state_bytes"] == cap and str(ep.graph) == fx
        if debug:
            for ref in refs:
                for a, w in zip(ref[3::6], ws, strict=True):
                    assert torch.equal(a.cpu(), ref[1].cpu().int() @ w.cpu().int())
        for t in model.buffers():
            t.zero_()
        kept = [f(*a) for a in req]
        ptr = [t.data_ptr() for y in kept for t in y]
        assert len(set(ptr)) == len(ptr)
    assert all(
        bits(x, y)
        for row, ref in zip(kept, refs, strict=True)
        for x, y in zip(row, ref, strict=True)
    )


@pytest.mark.parametrize(
    "mode",
    ["normal", "alpha", "scalar", "row_bias", "wide", "strided", "wrong_order", "early_consumer"],
)
@torch.inference_mode()
def test_affine_frontend_eligibility_and_placement(mode):
    class One(nn.Module):
        def forward(self, a, r, c, b):
            f = a.float()
            s = f * r
            o = s * c
            if mode == "alpha":
                y = torch.add(o, b, alpha=0.25)
            elif mode == "scalar":
                y = o + 1.25
            elif mode == "wrong_order":
                y = (s + b) * c
            else:
                y = o + b
            q = y if mode == "wide" else y.half()
            return (s.clone(), q) if mode == "early_consumer" else q

    g = torch.Generator(device="cuda").manual_seed(1637741)
    m, n = 21, 33
    a = torch.randint(-100, 100, (m, n), dtype=torch.int32, device="cuda", generator=g)
    r = torch.rand(m, 1, device="cuda", generator=g)
    c = torch.rand(n, device="cuda", generator=g)
    b = torch.randn((m, 1) if mode == "row_bias" else (n,), device="cuda", generator=g)
    if mode == "strided":
        b = torch.randn(2 * n, device="cuda", generator=g)[::2]
    module = One().eval()
    args = (a, r, c, b)
    ep = torch.export.export(module, args)
    gm, rs = partition_export(ep, affine_epilogues=True, host_backend="native")
    aff = [r for r in rs if r.get("kind") == "grouped_int32_affine"]
    assert bool(aff) == (mode in ("normal", "early_consumer"))
    want = module(*args)
    have = gm(*args)
    assert (
        all(bits(x, y) for x, y in zip(have, want, strict=True))
        if isinstance(have, tuple)
        else bits(have, want)
    )


@pytest.mark.parametrize("packed", [False, True])
@torch.inference_mode()
def test_affine_rounding_exceptional_values_and_odd_bands(packed):
    m = 9
    ns = (263, 17)
    g = torch.Generator(device="cuda").manual_seed(1649447)
    roles = ["cast", "row_scaled", "output", "biased", "narrow"]
    dts = ["bf16", "f16"]
    branches = [(m, n, roles, dt) for n, dt in zip(ns, dts, strict=True)]
    row = torch.randn(m, 1, device="cuda", generator=g)
    row[:5, 0] = torch.tensor(
        [0, -2147483648, 2139095040, -8388608, 2143294004], dtype=torch.int32, device="cuda"
    ).view(torch.float32)
    accs = [
        torch.randint(-(2**31), 2**31 - 1, (m, n), device="cuda", dtype=torch.int32, generator=g)
        for n in ns
    ]
    cs = [torch.randn(n, device="cuda", generator=g) for n in ns]
    bs = [torch.randn(n, device="cuda", generator=g) for n in ns]
    args = [row]
    idt = ["f32"]
    osh = []
    odt = []
    want = []
    if packed:
        args += [torch.cat(accs, dim=1)]
        idt += ["i32"]
    for j, (n, a, c, b) in enumerate(zip(ns, accs, cs, bs, strict=True)):
        if not packed:
            args.append(a)
            idt.append("i32")
        args.extend((c, b))
        idt.extend(("f32", "f32"))
        osh.extend([(m, n)] * 5)
        odt.extend(["f32"] * 4 + [dts[j]])
        f = a.float()
        s = f * row
        o = s * c
        y = o + b
        want.extend((f, s, o, y, y.to(torch.bfloat16 if j == 0 else torch.float16)))
    p = IndexProgram(
        [tuple(a.shape) for a in args],
        osh,
        "f32",
        "cuda",
        emit(branches, packed=packed),
        input_dtypes=idt,
        output_dtypes=odt,
    )
    got = IndexRegion(p, "native")(*args)
    for i, (x, y) in enumerate(zip(got, want, strict=True)):
        assert bits(x, y), (i, int((x.view(torch.uint8) != y.view(torch.uint8)).sum()))


@torch.inference_mode()
def test_affine_early_use_forbids_packing_all_branches():
    model, req = fixture((64, 32), True)
    ep = torch.export.export(model, req[0])
    mms = [n for n in ep.graph.nodes if n.target == torch.ops.aten._int_mm.default]
    node = mms[0]
    # Extra live exact copy before the second GEMM cannot read an accumulator that has not been produced.
    with ep.graph.inserting_after(node):
        copy = ep.graph.call_function(torch.ops.aten.clone.default, (node,))
        copy.meta = dict(node.meta)
    out = next(n for n in ep.graph.nodes if n.op == "output")
    out.args = (tuple(copy if j == 3 else a for j, a in enumerate(out.args[0])),)
    ep.graph_module.recompile()
    with pytest.raises((Unsupported, RuntimeError), match="early|group"):
        capture_export(ep, *req[:2], int8_rhs_layout="packed_fanout", partition_options=OPTS)


@pytest.mark.parametrize(
    "dt,target,half_ulp,one_bits",
    [("bf16", torch.bfloat16, 2.0**-8, 0x3F80), ("f16", torch.float16, 2.0**-11, 0x3C00)],
)
@torch.inference_mode()
def test_separate_rounding_and_exact_narrow_ties(dt, target, half_ulp, one_bits):
    # The first column is a contraction witness: fused (1+e)*(1-e)-1 = -e**2,
    # whereas separate, correctly rounded multiply and add return *positive* zero.
    e = 2.0**-23
    acc = torch.tensor([[1, 0, 0]], dtype=torch.int32, device="cuda")
    row = torch.tensor([[1 + e]], device="cuda")
    channel = torch.tensor([1 - e, 1, 1], device="cuda")
    bias = torch.tensor([-1, 1 + half_ulp, 1 + 3 * half_ulp], device="cuda")
    p = IndexProgram(
        [(1, 1), (1, 3), (3,), (3,)],
        [(1, 3)],
        "f32",
        "cuda",
        emit([(1, 3, ["narrow"], dt)]),
        input_dtypes=("f32", "i32", "f32", "f32"),
        output_dtypes=(dt,),
    )
    (out,) = IndexRegion(p, "native")(row, acc, channel, bias)
    assert out.dtype == target and out.view(torch.int16).cpu().tolist() == [
        [0, one_bits, one_bits + 2]
    ]
    assert torch.tensor(-e * e, device="cuda").to(target).view(torch.int16).item() != 0


@torch.inference_mode()
def test_unknown_metadata_assertion_is_not_discarded():
    from sparkfuse.projection_affine import match

    model, req = fixture((32, 16), False)
    gm = torch.export.export(model, req[0]).module()
    pos = {n: i for i, n in enumerate(gm.graph.nodes)}
    end = next(
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.aten.to.dtype and n.args[-1] is torch.bfloat16
    )
    assert match(end, torch.device("cuda:0"), pos) is not None
    guard = next(
        n for n in end.args[0].users if n.target == torch.ops.aten._assert_tensor_metadata.default
    )
    guard.kwargs = {**guard.kwargs, "sizes": [17, 32]}
    assert match(end, torch.device("cuda:0"), pos) is None
