import pytest
import torch
from examples.length_decode import CanonicalDecode, export_length_decode
from sparkfuse.length_attention import compile_length_decode
from sparkfuse.ir import Unsupported


def inputs(h, g, d, s, dt, mask):
    xs = [
        torch.randn(shape, device="cuda", dtype=dt)
        for shape in ((2, h, 1, d), (2, g, s, d), (2, g, s, d))
    ]
    if mask:
        bias = torch.zeros(2, 1, 1, s, device="cuda", dtype=dt)
        bias[0].fill_(-torch.inf)
        bias[1, ..., ::7] = -torch.inf
        xs.append(bias)
    return tuple(xs)


@pytest.mark.parametrize(
    "h,g,d,dt,mask", [(6, 2, 128, torch.bfloat16, True), (8, 2, 64, torch.float16, False)]
)
@pytest.mark.parametrize("tile,warps", [(32, 4), (64, 1)])
@torch.inference_mode()
def test_register_warp_short_tail_full_mask_and_feedback(h, g, d, dt, mask, tile, warps):
    mk = lambda s: inputs(h, g, d, s, dt, mask)
    ex, au = mk(17), mk(65)
    model = CanonicalDecode()
    ep = export_length_decode(model, ex, upper=513)
    obj = compile_length_decode(ep, ex, au, host_backend="native", tile=tile, warp_heads=warps)
    paths = obj.paths
    live = []
    current = torch.cuda.current_stream()
    other = torch.cuda.Stream()
    for s in (2, 7, 8, 9, 31, 32, 33, 63, 64, 65, 129, 513):
        xs = mk(s)
        ref = model(*xs)
        other.wait_stream(current)
        with torch.cuda.stream(other):
            y = obj.run_bounded(*xs)
        current.wait_stream(other)
        torch.testing.assert_close(
            y,
            ref,
            atol=0.012 if dt == torch.bfloat16 else 0.002,
            rtol=0.02 if dt == torch.bfloat16 else 0.004,
            check_stride=True,
        )
        assert (
            y._base is None
            and y.storage_offset() == 0
            and all(y.data_ptr() != a.data_ptr() for a in xs)
        )
        live.append(y)
        if mask:
            assert torch.count_nonzero(y[0]) == 0
    assert obj.paths == paths and len(paths) == 2 and len({x.data_ptr() for x in live}) == len(live)
    xs = list(mk(9))
    xs[0].fill_(float("nan"))
    with pytest.raises((ValueError, Unsupported)):
        obj(*xs)
    with pytest.raises((ValueError, Unsupported)):
        obj.run_bounded(*mk(514))


@torch.inference_mode()
def test_warp_descriptor_rejects_invalid_block_or_tile():
    from sparkfuse.decode_codegen import emit

    for warp, tile in ((-1, 32), (2, 32), (True, 32), (4, 128)):
        with pytest.raises(ValueError):
            emit(2, 6, 2, 8193, 128, "bf16", tile, 0.125, warp_heads=warp, runtime_length=True)
    with pytest.raises(ValueError):
        emit(2, 6, 2, 33, 128, "bf16", 32, 0.125, warp_heads=4)
