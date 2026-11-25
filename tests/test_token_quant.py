from math import prod

import pytest
import torch
from torch import nn
from torch.utils._pytree import tree_flatten

from examples.token_quant import ResidualQuant, quantize
from sparkfuse.partition import partition_export
from sparkfuse.token_quant import candidate


def compile_case(m, args, host="native", enabled=True):
    gm, records = partition_export(
        torch.export.export(m, args), host_backend=host, token_quant=enabled
    )
    return gm, [r for r in records if r.get("kind") == "residual_token_quant"]


def exact(out, ref):
    xs, a = tree_flatten(out)
    ys, b = tree_flatten(ref)
    assert a == b
    for x, y in zip(xs, ys, strict=True):
        assert (x.dtype, x.device, x.shape, x.stride(), x.storage_offset()) == (
            y.dtype,
            y.device,
            y.shape,
            y.stride(),
            y.storage_offset(),
        )
        assert torch.equal(
            x.contiguous().reshape(-1).view(torch.uint8),
            y.contiguous().reshape(-1).view(torch.uint8),
        )
    for i in range(len(xs)):
        for j in range(i):
            assert (xs[i] is xs[j]) == (ys[i] is ys[j])
            assert (xs[i].untyped_storage().data_ptr() == xs[j].untyped_storage().data_ptr()) == (
                ys[i].untyped_storage().data_ptr() == ys[j].untyped_storage().data_ptr()
            )


@pytest.mark.parametrize("host", ["native", "python"])
@pytest.mark.parametrize("shape,dt", [((9, 257), torch.float16), ((2, 3, 129), torch.bfloat16)])
@torch.inference_mode()
def test_current_stream_and_visible_outputs(shape, dt, host):
    class M(nn.Module):
        def forward(self, x, r):
            z = x + r
            early = z[..., 1:5]
            q, s = quantize(z)
            return early, z, q, s, q, s, r, x

    g = torch.Generator(device="cuda").manual_seed(19307)

    def tensor():
        return torch.randn(prod(shape) + 1, device="cuda", dtype=dt, generator=g)[1:].view(shape)

    m = M()
    x, r = tensor(), tensor()
    gm, (site,) = compile_case(m, (x, r), host)
    assert site["output_roles"] == ["residual", "codes", "scale"] and len(site["cuda_sha256"]) == 64
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        held = gm(x, r)
        ref = m(x, r)
        same = gm(x, x)
        same_ref = m(x, x)
        cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cg, stream=stream):
            captured = gm(x, r)
        for value in (1.5, -3.25):
            x.fill_(value)
            cg.replay()
            exact(captured, m(x, r))
        exact(held, ref)
        exact(same, same_ref)
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.parametrize("zout,sout", [(False, False), (True, False), (False, True), (True, True)])
@torch.inference_mode()
def test_allocate_only_live_outputs_and_two_sites(zout, sout):
    class M(nn.Module):
        def forward(self, x, r):
            z1 = x + r
            q1, s1 = quantize(z1)
            z2 = r + x
            q2, s2 = quantize(z2)
            return (q1, q2, *((z1, z2) if zout else ()), *((s1, s2) if sout else ()))

    m = M()
    x = torch.randn(3, 515, device="cuda", dtype=torch.float16)
    r = x.neg()
    gm, sites = compile_case(m, (x, r))
    assert len(sites) == 2
    for site in sites:
        assert site["output_roles"] == (["residual"] if zout else []) + ["codes"] + (
            ["scale"] if sout else []
        )
        assert len(getattr(gm, site["module"]).program.plan.outputs) == len(site["output_roles"])
    exact(gm(x, r), m(x, r))


@pytest.mark.parametrize(
    "reason",
    [
        "eps",
        "range",
        "truncate",
        "escape",
        "dim",
        "broadcast",
        "strided",
        "f32",
        "wide",
        "rank",
        "off",
        "source",
        "minimum",
        "no_keepdim",
        "alpha",
        "unrelated_scale",
    ],
)
@torch.inference_mode()
def test_decline_other_quantizers(reason):
    class M(nn.Module):
        def forward(self, x, r):
            z = torch.add(x, r, alpha=2) if reason == "alpha" else x + r
            f = z.float()
            a = (r.float() if reason == "source" else f).abs()
            a = (a.amin if reason == "minimum" else a.amax)(
                0 if reason == "dim" else -1, keepdim=reason != "no_keepdim"
            )
            s = a.clamp_min(1.0e-4 if reason == "eps" else 1.0e-5) / 127.0
            quotient = f / (s + 0.0 if reason == "unrelated_scale" else s)
            rounded = quotient.trunc() if reason == "truncate" else quotient.round()
            q = rounded.clamp(-128 if reason == "range" else -127, 127).to(torch.int8)
            return (q, a) if reason == "escape" else q

    shape = (2, 8193) if reason == "wide" else (1, 1, 2, 9) if reason == "rank" else (17, 17)
    x = torch.randn(
        shape, device="cuda", dtype=torch.float32 if reason == "f32" else torch.bfloat16
    )
    r = torch.randn_like(x)
    if reason == "strided":
        x = x.t()
    if reason == "broadcast":
        r = r[:1]
    m = M()
    ep = torch.export.export(m, (x, r))
    gm, records = partition_export(
        ep, host_backend="native", token_quant=reason != "off", min_ops=100
    )
    assert not any(k.get("kind") == "residual_token_quant" for k in records)
    exact(gm(x, r), m(x, r))


@torch.inference_mode()
def test_unproven_assertion_is_not_erased():
    x = torch.ones(5, 33, device="cuda", dtype=torch.bfloat16)
    m = ResidualQuant()
    ep = torch.export.export(m, (x, x.clone()))
    guard = next(
        n for n in ep.graph.nodes if n.target is torch.ops.aten._assert_tensor_metadata.default
    )
    guard.kwargs = {**guard.kwargs, "dtype": torch.float64}
    ep.graph_module.recompile()
    q = next(
        n for n in ep.graph.nodes if n.target is torch.ops.aten.to.dtype and n.args[1] is torch.int8
    )
    assert candidate(q, x.device) is None
    gm, rs = partition_export(ep, token_quant=True)
    assert not any(k.get("kind") == "residual_token_quant" for k in rs)
    with pytest.raises(RuntimeError):
        gm(x, x.clone())


@pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_codebook_even_ties_and_nonfinite(dt):
    # Fix peak=127 -> scale=1. Include all 254 half-way points and adjacent narrow values.
    half = torch.arange(-127, 127, device="cuda", dtype=torch.float32).add(0.5).to(dt)
    hi = torch.nextafter(half, torch.full_like(half, float("inf")))
    lo = torch.nextafter(half, torch.full_like(half, -float("inf")))
    x = torch.cat([torch.full_like(half[:1], 127), lo, half, hi])[None, :].repeat(7, 1)
    r = torch.zeros_like(x)
    x[1].zero_()
    x[2, 0] = float("inf")
    x[3, 0] = -float("inf")
    x[4, 0] = torch.finfo(dt).max
    r[4, 0] = torch.finfo(dt).max
    x[5, 0] = torch.tensor(
        0x7F83 if dt is torch.bfloat16 else 0x7C23, device="cuda", dtype=torch.int16
    ).view(dt)
    x[6].zero_()
    x[6, -1] = -0.0
    r[6, -1] = -0.0
    gm, (site,) = compile_case(ResidualQuant(), (x, r))
    out = gm(x, r)
    exact(out, ResidualQuant()(x, r))
    assert out[2][0, 0].item() == 1.0
    assert torch.equal(
        out[1][0, 1 + len(half) : 1 + 2 * len(half)], half.float().round().to(torch.int8)
    )


@pytest.mark.parametrize("host", ["native", "python"])
def test_input_guards_and_int8_not_a_scalar_ir_dtype(host):
    from sparkfuse.ir import DT

    assert torch.int8 not in DT
    x = torch.ones(4, 31, device="cuda", dtype=torch.float16)
    gm, (site,) = compile_case(ResidualQuant(), (x, x.clone()), host)
    region = getattr(gm, site["module"])
    for bad in (
        x.float(),
        x.to(torch.int8),
        x.cpu(),
        x[:2],
        torch.empty(31, 4, device="cuda", dtype=x.dtype).t(),
        x.detach().requires_grad_(True),
    ):
        with pytest.raises(ValueError):
            region(x, bad)


@pytest.mark.parametrize("shape", [(1, 1), (2, 1, 8192), (1, 1, 31)])
@torch.inference_mode()
def test_admission_limits_and_scale_floor(shape):
    x = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
    r = torch.zeros_like(x)
    m = ResidualQuant()
    gm, (site,) = compile_case(m, (x, r))
    for val in (0.0, 2.0**-22, 2.0**-16):
        x.fill_(val)
        exact(gm(x, r), m(x, r))
