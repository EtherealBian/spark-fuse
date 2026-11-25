"""Compare grouped query CTAs against the original FP32 partials and logical head ordering."""

import pytest
import torch
from examples.single_decode import SingleDecode
from sparkfuse.decode_attention import compile_decode_attention
from sparkfuse.ir import Unsupported


def batch(b, h, g, s, d, mask, dtype):
    seq = [
        torch.randn(shape, device="cuda", dtype=dtype)
        for shape in ((b, h, 1, d), (b, g, s, d), (b, g, s, d))
    ]
    if mask:
        bias = torch.randn(mask, 1, 1, s, device="cuda")
        bias[..., ::5] = -torch.inf
        if mask > 1:
            bias[0].fill_(-torch.inf)
        seq.append(bias)
    return tuple(seq)


@pytest.mark.parametrize(
    "s,tile,dtype,mask,group,width",
    [
        (7, 32, torch.float16, 0, 2, 64),
        (31, 32, torch.bfloat16, 2, 4, 128),
        (33, 32, torch.float16, 1, 2, 128),
        (65, 64, torch.bfloat16, 2, 4, 64),
        (129, 128, torch.float16, 0, 4, 128),
        (259, 256, torch.bfloat16, 1, 2, 64),
    ],
)
@torch.inference_mode()
def test_partials_equal_and_group_mapping(s, tile, dtype, mask, group, width):
    def req():
        return batch(2, 16, 2, s, width, mask, dtype)

    ex, au = req(), req()
    m = SingleDecode(scale=0.09375)
    ep = torch.export.export(m, ex)
    one = compile_decode_attention(ep, ex, au, tile=tile)
    many = compile_decode_attention(ep, ex, au, tile=tile, head_group=group)
    assert (
        one.plan.workspace_bytes == many.plan.workspace_bytes
        and many.plan.head_group == group
        and (len(one.programs) == len(many.programs))
    )
    assert "decode_group" in many.programs[0].source and "decode_partial" in one.programs[0].source
    if len(one.programs) == 2:
        assert one.programs[1].source == many.programs[1].source
    a = req()
    a[2][0, 0].fill_(7)
    a[2][0, 1].fill_(-9)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        y = many.run_bounded(*a)
        z = one.run_bounded(*a)
        p = many.programs[0](*a)[0]
        q = one.programs[0](*a)[0]
        f = m(*a)
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(p.contiguous().view(torch.uint8), q.contiguous().view(torch.uint8))
    assert (
        torch.equal(y.contiguous().view(torch.uint8), z.contiguous().view(torch.uint8))
        and y.stride() == f.stride()
        and y._base is None
        and z.data_ptr() != y.data_ptr()
    )
    if mask > 1:
        assert not torch.count_nonzero(y[0])
    else:
        assert torch.all(y[0, :8] == 7) and torch.all(y[0, 8:] == -9)


@pytest.mark.parametrize("h,g,group", [(8, 4, 4), (12, 4, 2), (8, 8, 2), (16, 2, True), (16, 2, 8)])
@torch.inference_mode()
def test_reject_nonintegral_head_group(h, g, group):
    ex, au = [batch(1, h, g, 63, 64, 0, torch.bfloat16) for _ in range(2)]
    ep = torch.export.export(SingleDecode(h != g), ex)
    with pytest.raises(Unsupported):
        compile_decode_attention(ep, ex, au, head_group=group)
