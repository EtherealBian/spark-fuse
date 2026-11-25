"""Observe host-equivalence, stream lifetime, policy and descriptor guards on real CUDA."""

import pytest
import torch
from examples.length_decode import CanonicalDecode, export_length_decode
from sparkfuse.length_attention import compile_length_decode


def inputs(s, mask, off=0, dt=torch.bfloat16):
    sh = [(2, 8, 1, 64), (2, 2, s, 64), (2, 2, s, 64)] + ([(2, 1, 1, s)] if mask else [])
    return tuple(
        torch.randn(torch.Size(a).numel() + off, device="cuda", dtype=dt)[off:].view(a) for a in sh
    )


def launchers(mask, upper):
    ex, au = inputs(7, mask), inputs(19, mask)
    m = CanonicalDecode()
    ep = export_length_decode(m, ex, upper=upper)
    return m, *[compile_length_decode(ep, ex, au, host_backend=k) for k in ("ctypes", "native")]


@pytest.mark.parametrize("upper,mask", [(29, False), (257, True)])
@torch.inference_mode()
def test_identical_code_bytes_layout_and_independent_outputs(upper, mask):
    m, py, n = launchers(mask, upper)
    assert n.paths == py.paths
    all_rows = []
    ptrs = set()
    for s in sorted({2, upper, 17, min(upper, 32), min(upper, 33)}):
        xs = inputs(s, mask, 8)
        if mask:
            xs[-1][0].fill_(-torch.inf)
        expected = m(*xs)
        a = py.run_bounded(*xs)
        b = n(*xs)
        c = n.run_bounded(*xs)
        for y in (a, b, c):
            assert (
                torch.equal(y.view(torch.int16), a.view(torch.int16))
                and y._base is None
                and y.storage_offset() == 0
                and y.stride() == expected.stride()
                and y.untyped_storage().nbytes() == y.numel() * 2
            )
            torch.testing.assert_close(y, expected, atol=0.012, rtol=0.02, check_stride=True)
            assert y.data_ptr() not in ptrs
            ptrs.add(y.data_ptr())
            all_rows.append(y)


@torch.inference_mode()
def test_guards_match_and_precede_dispatch():
    _, py, n = launchers(True, 129)
    xs = inputs(33, True)
    q, k, v, m = xs

    class Subclass(torch.Tensor):
        pass

    bad = [
        xs[:-1],
        xs + (q,),
        inputs(1, True),
        inputs(130, True),
        inputs(65, True, 1),
        (q, k, v[..., 1:, :], m),
        (q, k, v.float(), m),
        (q, k, v, m.expand(2, 8, 1, 33)),
        (q.as_strided(q.shape, (512, 64, 1, 1)), k, v, m),
        (q.as_subclass(Subclass), k, v, m),
        (torch.nn.Parameter(q, requires_grad=False), k, v, m),
        (q.cpu(), k, v, m),
        (q, k, None, m),
    ]
    for fn in (py.run_bounded, n.run_bounded):
        for args in bad:
            with pytest.raises(ValueError):
                fn(*args)
        with torch.inference_mode(False), pytest.raises(ValueError):
            fn(*xs)
        with torch.autocast("cuda"), pytest.raises(ValueError):
            fn(*xs)
    xs[3][0, 0, 0, 0] = torch.inf
    for fn in (py, n):
        with pytest.raises(ValueError):
            fn(*xs)


@torch.inference_mode()
def test_consumer_stream_survives_inputs_dropped_on_producer():
    _, py, n = launchers(False, 129)
    producer = torch.cuda.current_stream()
    consumer = torch.cuda.Stream()
    observed = []
    expected = []
    for idx in range(8):
        s = 32 if idx % 2 == 0 else 129
        xs = inputs(s, False)
        expected.append(py.run_bounded(*xs))
        consumer.wait_stream(producer)
        with torch.cuda.stream(consumer):
            torch.cuda._sleep(1_000_000)
            observed.append(n.run_bounded(*xs))
        del xs
        # Reuse/allocation pressure on the originating stream while the other stream is queued.
        recycled = [
            torch.full((2, 2, s, 64), -31, device="cuda", dtype=torch.bfloat16) for _ in range(4)
        ]
        del recycled
    producer.wait_stream(consumer)
    assert (
        all(
            torch.equal(x.view(torch.int16), y.view(torch.int16))
            for x, y in zip(observed, expected)
        )
        and len({x.data_ptr() for x in observed}) == 8
    )
