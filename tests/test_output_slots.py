"""Caller-owned buffers are an explicit mutable serving ABI, never ordinary fresh results."""

import threading
import pytest
import torch
from torch import nn
from sparkfuse.ir import Unsupported
from sparkfuse.resident import capture_export


class Outputs(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.register_buffer("bias", torch.randn(7, 32, device="cuda"))

    def forward(self, x, r):
        u = x + self.bias
        if self.mode == "one":
            return u
        if self.mode == "tuple1":
            return (u,)
        return u, (x - r).to(torch.bfloat16), u * r


def req(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return tuple(torch.randn(7, 32, generator=g, device="cuda") for _ in range(2))


def setup(mode="many", native=True):
    a, b = req(1), req(2)
    model = Outputs(mode)
    ep = torch.export.export(model, a)
    return (
        model,
        a,
        b,
        capture_export(ep, a, b, backend="aten", request_backend="native" if native else "python"),
    )


def leaves(x):
    return x if type(x) is tuple else (x,)


def same(x, y):
    return type(x) is type(y) and all(
        a.shape == b.shape
        and a.stride() == b.stride()
        and a.dtype == b.dtype
        and torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))
        for a, b in zip(leaves(x), leaves(y), strict=True)
    )


@pytest.mark.parametrize("mode", ["one", "tuple1", "many"])
@torch.inference_mode()
def test_output_slots_ring_survives_close_and_explicit_reuse(mode):
    m, a, b, exe = setup(mode)
    rs = [req(7 + i) for i in range(8)]
    with exe:
        refs = [m(*r) for r in rs]
        slots = [
            tuple(torch.empty_like(t) for t in v) if type(v) is tuple else torch.empty_like(v)
            for v in refs
        ]
        fresh = exe(*rs[0])
        future = [exe.run_into(r, s) for r, s in zip(rs, slots, strict=True)]
        assert all(x is y for x, y in zip(slots, future, strict=True))
        assert all(same(x, y) for x, y in zip(slots, refs, strict=True))
        assert (
            exe.run_into(rs[-1], slots[0]) is slots[0]
            and same(slots[0], refs[-1])
            and same(fresh, refs[0])
        )
        assert len({t.data_ptr() for s in slots for t in leaves(s)}) == 8 * len(leaves(refs[0]))
    assert same(slots[-1], refs[-1]) and same(fresh, refs[0])
    with pytest.raises(RuntimeError, match="closed"):
        exe.run_into(rs[0], slots[0])


@torch.inference_mode()
def test_output_slots_atomic_preflight_all_late_errors():
    m, a, b, exe = setup()
    inp = req(43)
    with exe:
        refs = m(*inp)
        good = tuple(torch.full_like(t, 91) for t in refs)

        class Sub(torch.Tensor):
            pass

        wide = torch.ones(225, device="cuda")
        with torch.inference_mode(False):
            grad = torch.empty(7, 32, device="cuda", requires_grad=True)
        last = good[-1]
        bad = [
            good[0],
            inp[0],
            a[1],
            b[1],
            m.bias,
            m.bias.detach(),
            next(exe._fn.buffers()),
            exe._static[0],
            exe._outputs[-1],
            last.double(),
            last.cpu(),
            last.t(),
            last.as_subclass(Sub),
            last._neg_view(),
            wide[1:].view(7, 32),
            wide[:-1].view(7, 32),
            grad,
            nn.Parameter(last, requires_grad=False),
            torch.from_dlpack(good[0]),
            None,
        ]
        for v in bad:
            before = tuple(t.clone() for t in exe._static)
            old = tuple(t.clone() for t in good)
            with pytest.raises((ValueError, Unsupported)):
                exe.run_into(inp, (*good[:-1], v))
            assert same(before, tuple(exe._static)) and same(old, good) and not exe._failed
        for wrong in ((), good[:-1], good + good, list(good)):
            with pytest.raises((ValueError, Unsupported)):
                exe.run_into(inp, wrong)
        for wrong in ((inp[0], inp[0]), (inp[0], inp[1].double()), list(inp)):
            with pytest.raises((ValueError, Unsupported)):
                exe.run_into(wrong, good)
        assert exe.run_into(inp, good) is good and same(refs, good)


@torch.inference_mode()
def test_output_slots_partial_overlap_distinct_external_storages():
    m, a, b, exe = setup()
    inp = req(50)
    with exe:
        refs = m(*inp)
        n = inp[0].numel()
        pool = torch.empty(n + 1, device="cuda")
        pool.fill_(91)
        # DLPack gives each slice separate storage with offset zero and a shifted pointer.
        u = torch.from_dlpack(pool[:-1]).view_as(inp[0])
        v = torch.from_dlpack(pool[1:]).view_as(inp[0])
        assert (
            u.data_ptr() != v.data_ptr()
            and u.untyped_storage().nbytes() == v.untyped_storage().nbytes() == 4 * n
        )
        with pytest.raises(Unsupported, match="overlap"):
            exe.run_into(inp, (u, torch.empty_like(refs[1]), v))
        with pytest.raises(Unsupported, match="overlap"):
            exe.run_into((u, inp[1]), (torch.empty_like(refs[0]), torch.empty_like(refs[1]), v))
        assert (pool == 91).all().item()


@torch.inference_mode()
def test_output_slots_guard_stream_thread_autocast_backend_and_fail_closed():
    m, a, b, exe = setup("one")
    out = torch.empty_like(a[0])
    errors = []
    with exe:
        with torch.cuda.stream(torch.cuda.Stream()), pytest.raises(RuntimeError, match="stream"):
            exe.run_into(a, out)
        with torch.autocast("cuda"), pytest.raises(ValueError, match="precision"):
            exe.run_into(a, out)
        with pytest.raises(Unsupported):
            exe.run_into(a, (out,))

        def worker():
            try:
                exe.run_into(a, out)
            except RuntimeError as err:
                errors.append(str(err))

        th = threading.Thread(target=worker)
        th.start()
        th.join()
        assert len(errors) == 1 and "thread" in errors[0]
        assert same(exe.run_into(a, out), m(*a))

        def fail(*a):
            raise RuntimeError("replay fault")

        exe._native_into = fail
        with pytest.raises(RuntimeError, match="replay fault"):
            exe.run_into(a, out)
        with pytest.raises(RuntimeError, match="failed"):
            exe(*b)
    _, _, _, alt = setup("one", native=False)
    with alt, pytest.raises(Unsupported, match="native"):
        alt.run_into(a, out)


@torch.inference_mode()
def test_output_slots_check_singleton_stride_before_write():
    class Mod(nn.Module):
        def forward(self, x, r):
            return (x + r).unsqueeze(1)

    a, b = req(61), req(62)
    m = Mod()
    ep = torch.export.export(m, a)
    with capture_export(ep, a, b, backend="aten", request_backend="native") as exe:
        ref = m(*a)
        dst = torch.empty_like(ref)
        weird = dst.as_strided(ref.shape, (32, 1999, 1))
        assert weird.is_contiguous() and weird.stride() != ref.stride()
        with pytest.raises(ValueError, match="layout"):
            exe.run_into(a, weird)
        assert same(exe.run_into(a, dst), ref)


@torch.inference_mode()
def test_output_slots_external_lifetime_and_regular_destination_version():
    m, a, b, exe = setup("one")
    with exe:
        with torch.inference_mode(False):
            dst = torch.empty_like(a[0])
        ver = dst._version
        other = req(77)
        side = torch.cuda.Stream()
        owner = torch.cuda.current_stream()
        side.wait_stream(owner)
        with torch.cuda.stream(side):
            alien = tuple(t.clone() for t in other)
        owner.wait_stream(side)
        assert exe.run_into(alien, dst) is dst and dst._version == ver + 1
        with pytest.raises(Unsupported, match="overlap"):
            exe.run_into((exe._static[1], alien[0]), dst)
        del alien
        side.wait_stream(owner)
        with torch.cuda.stream(side):
            readback = dst.clone()
        owner.wait_stream(side)
        assert same(readback, exe(*other))


@torch.inference_mode()
def test_output_slots_plain_process_sources_not_pinned():
    # Instrumented CUDA tools retain Python operation arguments. Test actual storage expiry in a
    # normal process; the separate async-lifetime, version and reuse tests also run under sanitizer.
    import gc
    from torch.multiprocessing.reductions import StorageWeakRef

    m, a, b, exe = setup("one")
    r = req(90)
    with exe:
        dst = torch.empty_like(a[0])
        exe.run_into(r, dst)
        torch.cuda.synchronize()
        refs = [StorageWeakRef(t.untyped_storage()) for t in (m.bias, *a, *b, dst)]
        del m, a, b, dst
        gc.collect()
        assert all(v.expired() for v in refs)
        assert torch.isfinite(exe(*r)).all().item()
