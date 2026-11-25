import ctypes
import pytest
import torch
from sparkfuse.contextual import _exact
from sparkfuse.ir import Unsupported
from sparkfuse.resident import capture_export
from sparkfuse import slot_copy


class Outputs(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        if self.n == 1:
            return x + 1
        return tuple((x + i).to(torch.bfloat16) if i % 2 else x + i for i in range(self.n))


def tensors(n=257):
    return [(torch.rand(n, device="cuda"),) for _ in range(11)]


def flat(y):
    return y if isinstance(y, tuple) else (y,)


@pytest.mark.parametrize("count", [1, 3, 21, 32])
@pytest.mark.parametrize("side", [False, True])
@torch.inference_mode()
def test_fresh_storage_survives_queue_close_and_mutation(count, side):
    owner = torch.cuda.Stream() if side else torch.cuda.current_stream()
    other = torch.cuda.Stream()
    with torch.cuda.stream(owner):
        model = Outputs(count)
        args = tensors(4097)
        ep = torch.export.export(model, args[0])
        ref = [model(*r) for r in args[2:]]
        with capture_export(
            ep, *args[:2], backend="aten", request_backend="native", fresh_copy_backend="cuda"
        ) as f:
            assert "fresh_copy" in f.metadata and "slot_copy" not in f.metadata
            ys = [f(*r) for r in args[2:]]
            out = [v for y in ys for v in flat(y)]
            keys = {v.untyped_storage().data_ptr() for v in out}
            assert len(keys) == len(out) and all(v[0].data_ptr() not in keys for v in args)
            with pytest.raises(ValueError):
                f(args[0][0][:-1].clone())
            for y, r in zip(ys, ref, strict=True):
                _exact(y, r, "fresh batch")
            with torch.cuda.stream(other):
                with pytest.raises(RuntimeError, match="stream"):
                    f(*args[-1])
        with pytest.raises(RuntimeError, match="closed"):
            f(*args[-1])
        other.wait_stream(owner)
        with torch.cuda.stream(other):
            for y, r in zip(ys, ref, strict=True):
                _exact(y, r, "after close")
            flat(ys[0])[0].fill_(777)
            _exact(ys[-1], ref[-1], "independent")
            if count > 1:
                _exact(ys[0][1:], ref[0][1:], "sibling")
        owner.wait_stream(other)


@pytest.mark.parametrize(
    "fresh,slots", [("aten", "aten"), ("cuda", "aten"), ("aten", "cuda"), ("cuda", "cuda")]
)
@torch.inference_mode()
def test_two_transports_are_independent(monkeypatch, fresh, slots):
    real = slot_copy.prepare
    calls = []
    fail = [False]
    builds = []

    def Spy(outputs, device):
        lib, entry, meta = real(outputs, device)
        callback = ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_void_p
        )
        forward = callback(entry)

        @callback
        def launch(ptr, n, stream):
            calls.append(n)
            return 1 if fail[0] else forward(ptr, n, stream)

        builds.append(1)
        return (lib, launch, forward), ctypes.cast(launch, ctypes.c_void_p).value, meta

    monkeypatch.setattr(slot_copy, "prepare", Spy)
    model = Outputs(3)
    r = tensors()
    ep = torch.export.export(model, r[0])
    with capture_export(
        ep,
        *r[:2],
        backend="aten",
        request_backend="native",
        fresh_copy_backend=fresh,
        slot_copy_backend=slots,
    ) as f:
        assert len(builds) == int("cuda" in (fresh, slots))
        calls.clear()
        ref = model(*r[2])
        _exact(f(*r[2]), ref, "fresh")
        assert len(calls) == int(fresh == "cuda")
        calls.clear()
        into = tuple(torch.empty_like(t) for t in ref)
        f.run_into(r[2], into)
        _exact(into, ref, "slots")
        assert len(calls) == int(slots == "cuda")
        if fresh == "cuda":
            fail[0] = True
            with pytest.raises(RuntimeError):
                f(*r[2])
            with pytest.raises(RuntimeError, match="failed"):
                f(*r[2])


@pytest.mark.parametrize("singleton_stride", [1, 19, 257])
@torch.inference_mode()
def test_exact_strides_of_fresh_outputs(singleton_stride):
    class Model(torch.nn.Module):
        def forward(self, x):
            return (x + 1).as_strided((1, 257, 1), (singleton_stride, 1, 19)), x.to(torch.bfloat16)

    m = Model()
    r = tensors()
    ep = torch.export.export(m, r[0])
    with capture_export(
        ep, *r[:2], backend="aten", request_backend="native", fresh_copy_backend="cuda"
    ) as f:
        y = f(*r[2])
        ref = m(*r[2])
        _exact(y, ref, "unusual singleton strides")
        assert all(y0.storage_offset() == 0 for y0 in y)


@torch.inference_mode()
def test_fresh_limits_and_fallback():
    r = tensors(67)
    m = Outputs(33)
    ep = torch.export.export(m, r[0])
    with pytest.raises(Unsupported, match="1..32"):
        capture_export(
            ep, *r[:2], request_backend="native", backend="aten", fresh_copy_backend="cuda"
        )
    for req, copy in [("python", "cuda"), ("native", "auto")]:
        with pytest.raises(ValueError, match="fresh copies"):
            capture_export(ep, *r[:2], request_backend=req, backend="aten", fresh_copy_backend=copy)
    with capture_export(ep, *r[:2], request_backend="native", backend="aten") as f:
        _exact(f(*r[-1]), m(*r[-1]), "default >32")
