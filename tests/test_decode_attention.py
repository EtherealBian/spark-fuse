"""A standalone SDPA export, not an unrestricted attention substitute."""

import pytest
import torch
from examples.single_decode import SingleDecode
from sparkfuse.decode_attention import compile_decode_attention
from sparkfuse.decode_codegen import emit
from sparkfuse.ir import Unsupported


def make(s=33, *, b=2, h=4, g=2, d=64, dt=torch.bfloat16, mask=0, offset=0):
    result = []
    for shape in [(b, h, 1, d), (b, g, s, d), (b, g, s, d)] + ([(mask, 1, 1, s)] if mask else []):
        width = torch.float32 if len(result) == 3 else dt
        x = torch.empty(torch.Size(shape).numel() + offset, dtype=width, device="cuda")[
            offset:
        ].view(shape)
        x.normal_()
        result.append(x)
    return tuple(result)


def build(ex=None, au=None, model=None, **opts):
    ex = make() if ex is None else ex
    au = make() if au is None else au
    model = SingleDecode() if model is None else model
    ep = torch.export.export(model, ex)
    return compile_decode_attention(ep, ex, au, **opts)


@pytest.mark.parametrize(
    "s,tile,dt,mask",
    [
        (1, 32, torch.float16, 0),
        (31, 32, torch.bfloat16, 1),
        (32, 32, torch.float16, 2),
        (33, 32, torch.bfloat16, 2),
        (127, 64, torch.float16, 0),
        (129, 128, torch.bfloat16, 0),
        (257, 256, torch.float16, 1),
    ],
)
@torch.inference_mode()
def test_exported_tail_ownership_mask_and_stream(s, tile, dt, mask):
    def req():
        return make(s, dt=dt, mask=mask, offset=8)

    model = SingleDecode(scale=0.1875)
    ex, au, held = req(), req(), req()
    f = build(ex, au, model, tile=tile)
    if mask:
        held[3][0].fill_(-torch.inf)
        if mask > 1:
            held[3][1].fill_(-10000.0)
            held[3][1, ..., -1] = 10000.0
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    before = [v.clone() for v in held]
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        y = f.run_bounded(*held)
        second = f(*held)
        ref = model(*held)
    torch.cuda.current_stream().wait_stream(stream)
    assert f.plan.workspace_bytes == (0 if s <= tile else 2 * 4 * ((s + tile - 1) // tile) * 66 * 4)
    for z in (y, second):
        assert (
            z._base is None
            and z.storage_offset() == 0
            and z.data_ptr() not in [t.data_ptr() for t in held]
        )
        torch.testing.assert_close(
            z,
            ref,
            atol=0.008 if dt == torch.bfloat16 else 0.001,
            rtol=0.02 if dt == torch.bfloat16 else 0.003,
            check_stride=True,
        )
        if mask:
            assert torch.count_nonzero(z[0]) == 0
        if mask > 1:
            assert torch.equal(z[1, :, 0], held[2][1, :, -1].repeat_interleave(2, dim=0))
    assert y.data_ptr() != second.data_ptr() and all(
        torch.equal(x, u) for x, u in zip(held, before)
    )


@pytest.mark.parametrize(
    "bad",
    [
        "budget",
        "tile",
        "same_audit",
        "shape",
        "mask_dtype",
        "mask_heads",
        "qwidth",
        "prefill",
        "causal",
        "dropout",
        "scale",
        "compound",
        "tuple",
        "unbounded",
        "misaligned",
    ],
)
@torch.inference_mode()
def test_compile_fails_closed(bad):
    ex, au = make(), make()
    model = SingleDecode()
    opts = {}
    if bad == "budget":
        opts = dict(max_workspace_bytes=0)
    elif bad == "tile":
        opts = dict(tile=True)
    elif bad == "same_audit":
        au = ex
    elif bad == "shape":
        au = make(34)
    elif bad == "mask_dtype":
        ex, au = [(*a, torch.zeros(1, 1, 1, 33, device="cuda", dtype=torch.bool)) for a in (ex, au)]
    elif bad == "mask_heads":
        ex, au = [(*a, torch.zeros(2, 4, 1, 33, device="cuda")) for a in (ex, au)]
    elif bad == "qwidth":
        ex, au = make(d=32), make(d=32)
    elif bad == "prefill":
        ex, au = [(a[0].repeat(1, 1, 2, 1), *a[1:]) for a in (ex, au)]
    elif bad == "causal":
        model = SingleDecode(causal=True)
    elif bad == "dropout":
        model = SingleDecode(dropout=0.1)
    elif bad == "scale":
        model = SingleDecode(scale=-0.125)
    elif bad == "misaligned":
        au = make(offset=1)
    elif bad == "unbounded":
        au[0][0, 0, 0, 0] = torch.nan
    elif bad in ("compound", "tuple"):

        class Wrapped(SingleDecode):
            def forward(self, q, k, v):
                x = super().forward(q, k, v)
                return (x,) if bad == "tuple" else x + q

        model = Wrapped()
    with pytest.raises((Unsupported, ValueError)):
        build(ex, au, model, **opts)


@torch.inference_mode()
def test_runtime_guard_contract():
    ex = make(mask=1)
    f = build(ex, make(mask=1))
    q, k, v, m = ex
    for args in (
        (q, k),
        (q.cpu(), k, v, m),
        (q.float(), k, v, m),
        (q, k[..., :-1, :], v, m),
        make(mask=1, offset=1),
        (q.as_strided(q.shape, (q.stride(0), q.stride(1), 502, q.stride(3))), k, v, m),
    ):
        with pytest.raises(ValueError):
            f.run_bounded(*args)
    with torch.autocast("cuda"):
        with pytest.raises(ValueError):
            f.run_bounded(*ex)
    from torch.nn.attention import sdpa_kernel, SDPBackend

    with sdpa_kernel(SDPBackend.MATH):
        with pytest.raises(ValueError):
            f.run_bounded(*ex)
    with torch.inference_mode(False):
        with pytest.raises(ValueError):
            f.run_bounded(*ex)
    for x, val in ((m, float("inf")), (q, 65), (k, float("nan")), (v, -float("inf"))):
        saved = x.clone()
        x.flatten()[0] = val
        with pytest.raises(ValueError):
            f(*ex)
        x.copy_(saved)
    assert torch.equal(f(*ex), f.run_bounded(*ex))


def test_emitter_integer_abi_and_range():
    for args in [
        (32, 256, 256, 32768, 128, "bf16", 32, 1.0),
        (True, 4, 2, 32, 64, "f16", 32, 0.125),
        (1, 4, 2, 32, 64, "f16", 32, float("nan")),
    ]:
        with pytest.raises(ValueError):
            emit(*args)
