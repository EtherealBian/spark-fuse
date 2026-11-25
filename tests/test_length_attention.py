import math
import pytest
import torch
from examples.length_decode import CanonicalDecode, export_length_decode
from examples.single_decode import SingleDecode
from sparkfuse.length_attention import compile_length_decode
from sparkfuse.decode_attention import compile_decode_attention
from sparkfuse.ir import Unsupported


def batch(s, mask=False, dt=torch.bfloat16, offset=0, d=64):
    shapes = [(2, 8, 1, d), (2, 2, s, d), (2, 2, s, d)] + ([(2, 1, 1, s)] if mask else [])
    return tuple(
        torch.randn(math.prod(sh) + offset, device="cuda", dtype=dt)[offset:].view(sh)
        for sh in shapes
    )


@pytest.mark.parametrize(
    "mask,dt,d,tile", [(False, torch.float16, 64, 32), (True, torch.bfloat16, 128, 64)]
)
@torch.inference_mode()
def test_one_program_many_lengths(mask, dt, d, tile):
    make = lambda s: batch(s, mask, dt, 8, d)
    ex, au = make(7), make(41)
    model = CanonicalDecode()
    ep = export_length_decode(model, ex, upper=131)
    f = compile_length_decode(ep, ex, au, tile=tile)
    assert len(f.paths) == 2
    live = []
    ids = set()
    stream = torch.cuda.Stream()
    for s in (2, tile - 1, tile, tile + 1, 2 * tile, 131):
        xs = make(s)
        if mask:
            xs[-1][0].fill_(-torch.inf)
        before = [x.clone() for x in xs]
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            out = f(*xs)
            out2 = f.run_bounded(*xs)
        torch.cuda.current_stream().wait_stream(stream)
        assert f.plan.workspace_bytes(s) == (
            0 if s <= tile else 2 * 8 * (d + 2) * 4 * ((s + tile - 1) // tile)
        )
        static = compile_decode_attention(
            torch.export.export(SingleDecode(), xs), xs, make(s), tile=tile
        )
        ref = model(*xs)
        for z in (out, out2):
            torch.testing.assert_close(
                z,
                ref,
                atol=0.012 if dt == torch.bfloat16 else 0.002,
                rtol=0.02 if dt == torch.bfloat16 else 0.004,
                check_stride=True,
            )
            assert (
                torch.equal(z, static.run_bounded(*xs))
                and z._base is None
                and z.storage_offset() == 0
                and z.untyped_storage().nbytes() == z.numel() * z.element_size()
                and z.data_ptr() not in ids
            )
            ids.add(z.data_ptr())
            live.append(z)
        assert all(torch.equal(x, y) for x, y in zip(xs, before))


@torch.inference_mode()
def test_guard_and_symbol_bounds():
    ex, au = batch(15, True), batch(32, True)
    ep = export_length_decode(CanonicalDecode(), ex, lower=3, upper=65)
    f = compile_length_decode(ep, ex, au)
    for bad in (
        batch(2, True),
        batch(66, True),
        batch(8, True, offset=1),
        batch(10),
        batch(11, True, torch.float16),
        (ex[0], ex[1], au[2], ex[3]),
        (ex[0].transpose(1, 2), *ex[1:]),
        (*ex[:3], ex[3].float()),
    ):
        with pytest.raises(ValueError):
            f.run_bounded(*bad)
    ex[1][0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError):
        f(*ex)
    with torch.inference_mode(False), pytest.raises(ValueError):
        f.run_bounded(*au)


@torch.inference_mode()
def test_graph_and_compile_guards():
    ex, au = batch(11), batch(17)
    model = CanonicalDecode()
    ep = export_length_decode(model, ex, upper=65)
    for graph, other, opts in [
        (torch.export.export(model, ex), au, {}),
        (export_length_decode(SingleDecode(), ex, upper=65), au, {}),
        (export_length_decode(CanonicalDecode(causal=True), ex, upper=65), au, {}),
        (ep, batch(11), {}),
        (ep, (ex[0], *au[1:]), {}),
        (ep, au, {"max_workspace_bytes": 1}),
        (ep, au, {"tile": True}),
    ]:
        with pytest.raises((Unsupported, ValueError)):
            compile_length_decode(graph, ex, other, **opts)
    f = compile_length_decode(export_length_decode(model, ex, upper=29), ex, au)
    assert len(f.paths) == 1 and f.plan.workspace_bytes(29) == 0
