"""Native resident staging must preserve the Python request and fresh-output contracts."""

import gc
import threading
import weakref

import pytest
import torch
from torch import nn
from sparkfuse.ir import Unsupported
from sparkfuse.resident import capture_export


class Visible(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, x, y):
        u = x + y
        v = x - y
        if self.mode == "tensor":
            return v
        if self.mode == "tuple1":
            return (v,)
        if self.mode == "alias":
            return (u, u[:, 1:])
        if self.mode == "input":
            return (x, u)
        if self.mode == "repeat":
            return (v, v)
        if self.mode == "nested":
            return {"first": [u], "last": v}
        return u, v, u == v


def req(i, dtype=torch.float32, shape=(17, 32)):
    g = torch.Generator(device="cuda").manual_seed(i)
    return tuple(torch.randn(shape, device="cuda", generator=g).to(dtype) for _ in range(2))


def same(x, y):
    if isinstance(x, tuple):
        return (
            type(x) is type(y)
            and len(x) == len(y)
            and all(same(a, b) for a, b in zip(x, y, strict=True))
        )
    return (
        x.dtype == y.dtype
        and x.device == y.device
        and x.shape == y.shape
        and x.stride() == y.stride()
        and x.storage_offset() == y.storage_offset()
        and torch.equal(
            x.contiguous().reshape(-1).view(torch.uint8),
            y.contiguous().reshape(-1).view(torch.uint8),
        )
    )


@pytest.mark.parametrize("mode", ["tensor", "tuple1", "multiple"])
@pytest.mark.parametrize("backend", ["aten", "mixed"])
@torch.inference_mode()
def test_native_io_qualified_fresh_results_after_close(mode, backend):
    # eq operator is outside the resident allowlist; use three different rounded expressions instead.
    class Model(Visible):
        def forward(self, x, y):
            return (x + y, x - y, x * y) if self.mode == "multiple" else super().forward(x, y)

    model = Model(mode)
    reqs = [req(1471259 + i) for i in range(6)]
    ep = torch.export.export(model, reqs[0])
    owner = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    side.wait_stream(owner)
    with torch.cuda.stream(side):
        run = capture_export(ep, *reqs[:2], backend=backend, request_backend="native")
        assert run.metadata["request_backend"] == "native"
        ys = [run(*a) for a in reqs]
        refs = [model(*a) for a in reqs]
        assert all(same(x, y) for x, y in zip(ys, refs, strict=True))
        ts = [t for y in ys for t in (y if type(y) is tuple else (y,))]
        assert len({t.untyped_storage().data_ptr() for t in ts}) == len(ts)
        run.close()
        assert run._static == () and run._native_io is None and run._outputs == []
        assert all(same(x, y) for x, y in zip(ys, refs, strict=True))
    owner.wait_stream(side)


@pytest.mark.parametrize("mode", ["repeat", "input", "alias", "nested"])
@torch.inference_mode()
def test_native_request_rejects_nonflat_or_alias_but_python_preserves(mode):
    m = Visible(mode)
    a, b = req(15), req(16)
    ep = torch.export.export(m, a)
    with pytest.raises(Unsupported, match="flat tuple of fresh"):
        capture_export(ep, a, b, backend="aten", request_backend="native")
    with capture_export(ep, a, b, backend="aten") as run:
        from torch.utils._pytree import tree_flatten

        x, s = tree_flatten(run(*a))
        y, v = tree_flatten(m(*a))
        assert s == v and all(same(u, w) for u, w in zip(x, y, strict=True))


@torch.inference_mode()
def test_native_request_preflight_rejections_recover_and_do_not_partially_stage():
    m = Visible("tensor")
    a, b = req(21), req(22)
    ep = torch.export.export(m, a)
    with capture_export(ep, a, b, backend="aten", request_backend="native") as run:

        class Sub(torch.Tensor):
            pass

        bad = [
            (a[0], a[0]),
            (a[0], a[1].cpu()),
            (a[0], a[1].t()),
            (a[0], torch.ones(17 * 32 + 1, device="cuda")[1:].view(17, 32)),
            (a[0], torch.ones(17 * 32 + 1, device="cuda")[:-1].view(17, 32)),
            (a[0], a[1].double()),
            (a[0], a[1].as_subclass(Sub)),
            (a[0],),
            (),
            (1, a[0]),
            (a[0], a[1]._neg_view()),
        ]
        for args in bad:
            old = [v.clone() for v in run._static]
            with pytest.raises((ValueError, Unsupported)):
                run(*args)
            assert (
                all(same(x, y) for x, y in zip(run._static, old, strict=True)) and not run._failed
            )
            assert same(run(*b), m(*b))
        p = nn.Parameter(a[0], requires_grad=True)
        assert same(run(p, a[1]), m(p, a[1]))
        with torch.autocast("cuda"), pytest.raises(ValueError, match="precision"):
            run(*a)
        with torch.cuda.stream(torch.cuda.Stream()), pytest.raises(RuntimeError, match="stream"):
            run(*a)
        errors = []

        def worker():
            try:
                run(*a)
            except Exception as e:
                errors.append(str(e))

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert len(errors) == 1 and "thread" in errors[0]
        assert same(run(*b), m(*b))


@torch.inference_mode()
def test_native_io_storage_failure_seals_handle_only_after_backend_execution():
    a, b = req(35), req(36)
    ep = torch.export.export(Visible("tensor"), a)
    with capture_export(ep, a, b, backend="aten", request_backend="native") as run:

        def broken(*args):
            raise RuntimeError("mocked backend copy failure")

        run._native_io = broken
        with pytest.raises(RuntimeError, match="mocked backend"):
            run(*a)
        with pytest.raises(RuntimeError, match="failed resident"):
            run(*b)


@torch.inference_mode()
def test_native_io_replays_changed_requests_on_static_singleton_stride():
    def shaped(seed):
        return tuple(t.as_strided((17, 1, 32), (32, 3072, 1)) for t in req(seed))

    a, b, c = shaped(51), shaped(52), shaped(53)
    m = Visible("tensor")
    ep = torch.export.export(m, a)
    with capture_export(ep, a, b, backend="aten", request_backend="native") as run:
        assert same(run(*c), m(*c))
        other = tuple(t.as_strided((17, 1, 32), (32, 32, 1)) for t in c)
        with pytest.raises(ValueError, match="layout"):
            run(*other)
        assert same(run(*a), m(*a))


@pytest.mark.parametrize("request_backend", ["python", "native"])
@torch.inference_mode()
def test_plain_process_releases_staging_wrappers_on_close(request_backend):
    # Instrumented CUDA tooling can retain operator argument tuples even on the Python control.
    # Test immediate Python-wrapper reclamation separately from device-access checks.
    a, b = req(79), req(80)
    model = Visible("tensor")
    ep = torch.export.export(model, a)
    run = capture_export(ep, a, b, backend="aten", request_backend=request_backend)
    sources = [weakref.ref(t) for t in run._static]
    last = run(*a)
    run.close()
    gc.collect()
    assert all(r() is None for r in sources) and same(last, model(*a))
