import pytest
import torch
from examples.paged_decode_example import PageGatherDecode, request
from examples.ragged_pages import RaggedGatherDecode, ragged_request
from sparkfuse.paged_decode import compile_paged_decode
from sparkfuse.ir import Unsupported

CASES = [
    (False, (2, 8, 2, 64, 32, 31, 0, "fp16")),
    (False, (2, 6, 2, 128, 64, 129, 1, "bf16")),
    (True, (4, 8, 2, 64, 32, 31, 4, "bf16")),
    (True, (4, 6, 2, 128, 64, 129, 1, "fp16")),
]


def make(mode, cfg, seed, j=0):
    p = ("shuffled", "endpoints", "repeated", "shared")[j % 4]
    return (
        ragged_request(cfg, seed, ("full", "extremes", "empty", "mixed")[j % 4], p)
        if mode
        else request(cfg, seed, p)
    )


def setup(mode, cfg, seed):
    m = (RaggedGatherDecode if mode else PageGatherDecode)(cfg[5])
    a, b = (make(mode, cfg, seed + i, i) for i in (0, 1))
    ep = torch.export.export(m, a, strict=True)
    return ep, a, b


@pytest.mark.parametrize("mode,cfg", CASES)
@torch.inference_mode()
def test_stream_ownership_and_identical_device_programs(mode, cfg):
    ep, a, b = setup(mode, cfg, 31309)
    z, f = (compile_paged_decode(ep, a, b, host_backend=k) for k in ("split", "native"))
    assert all(
        x.path == y.path and x.source == y.source
        for x, y in zip(z.programs, f.programs, strict=True)
    )
    origin = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    out = []
    gold = []
    for i in range(6):
        x = make(mode, cfg, 31517 + i, i)
        gold.append(z.run_bounded(*x))
        side.wait_stream(origin)
        with torch.cuda.stream(side):
            out.append(f.run_bounded(*x))
            assert all(out[-1].data_ptr() != v.data_ptr() for v in x)
        del x
    origin.wait_stream(side)
    assert len({y.data_ptr() for y in out}) == len(out)
    assert all(
        torch.equal(x.view(torch.int16), y.view(torch.int16)) and x.stride() == y.stride()
        for x, y in zip(gold, out, strict=True)
    )


@torch.inference_mode()
def test_native_descriptor_policy_and_checked_domain():
    mode, cfg = CASES[-1]
    ep, a, b = setup(mode, cfg, 32909)
    f = compile_paged_decode(ep, a, b, host_backend="native")
    q, k, v, table, length, bias = a

    def odd(x, offset):
        return (
            torch.empty(x.numel() + offset, device=x.device, dtype=x.dtype)[offset:]
            .view_as(x)
            .copy_(x)
        )

    for xs in (
        (odd(q, 1), *a[1:]),
        (q, odd(k, 1), *a[2:]),
        (*a[:-1], odd(bias, 1)),
        (q, k, v, table.long(), length, bias),
        (q, k, v, table, length.cpu(), bias),
        (q, k, v, table, length[:2], bias),
        a[:-1],
        (q.transpose(1, 3), *a[1:]),
    ):
        with pytest.raises(ValueError):
            f.run_bounded(*xs)

    class Special(torch.Tensor):
        pass

    with pytest.raises(ValueError):
        f.run_bounded(q.as_subclass(Special), *a[1:])
    with torch.inference_mode(False), pytest.raises(ValueError):
        f.run_bounded(*a)
    with torch.autocast("cuda"), pytest.raises(ValueError):
        f.run_bounded(*a)
    with pytest.raises(Unsupported):
        compile_paged_decode(ep, a, b, host_backend="invalid")
    # Sliced pools are fine if aligned; typed 4-byte offset tables/lengths need not be 16-byte aligned.
    x = (odd(q, 8), k, v, odd(table, 1), odd(length, 1), bias)
    assert torch.equal(f(*a).view(torch.int16), f(*x).view(torch.int16))
    bias.fill_(-torch.inf)
    assert not f(*a).view(torch.int16).count_nonzero()
    q.fill_(float("nan"))
    with pytest.raises(ValueError):
        f(*a)
