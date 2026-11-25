"""Test observable request ownership, snapshot lifetime and fail-closed capture on a real GPU."""

import copy
import gc
import threading
import weakref

import pytest
import torch
from torch import nn

from sparkfuse import Unsupported
from sparkfuse.contextual import _exact
from sparkfuse.resident import capture_export

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")


def inputs(seed, n=3):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return tuple(torch.randn(4, 128, device="cuda", generator=gen) for _ in range(n))


class Visible(nn.Module):
    def forward(self, x, a, b):
        y = (x + a) * b
        return {"input": (x, x[:, 1:], x.view_as(x)), "fresh": [y[:, 2:], y, y, y.view_as(y)]}


@pytest.mark.parametrize("backend", ["aten", "mixed"])
@torch.inference_mode()
def test_rolling_visible_outputs(backend):
    m = Visible()
    ex, audit = inputs(1), inputs(2)
    ep = torch.export.export(m, ex)
    options = {"host_backend": "native"} if backend == "mixed" else None
    with capture_export(ep, ex, audit, backend=backend, partition_options=options) as run:
        assert run.metadata["fresh_output_bytes"] == ex[0].nbytes
        if backend == "mixed":
            assert any("module" in r for r in run.metadata["regions"])
        pending = []
        for i in range(32):
            req = inputs(i + 12)
            out = run(*req)
            want = m(*req)
            x, sl, same = out["input"]
            cut, y, repeat, alias = out["fresh"]
            assert x is req[0] and same is not x and sl.storage_offset() == 1
            assert repeat is y and alias is not y and cut.storage_offset() == 2
            for v, owner in ((sl, x), (same, x), (alias, y), (cut, y)):
                assert v.untyped_storage().data_ptr() == owner.data_ptr()
            pending.append((out, want))
        assert len({o["fresh"][1].data_ptr() for o, _ in pending}) == len(pending)
        for out, want in pending:
            _exact(out, want, "queued")
        first = pending[0][0]
        first["fresh"][0].fill_(17)
        assert bool((first["fresh"][1][:, 2:] == 17).all())
        ex[0].add_(21)
        _exact(run(*ex), m(*ex), "new input contents")
    _exact(pending[-1][0], pending[-1][1], "outlives handle")
    run.close()
    with pytest.raises(RuntimeError, match="closed"):
        run(*ex)


class Weights(nn.Module):
    def __init__(self):
        super().__init__()
        w = inputs(523, 1)[0]
        self.register_buffer("a", w)
        self.register_buffer("b", w.detach())

    def forward(self, x):
        return (x + self.a) * self.b


@pytest.mark.parametrize("backend", ["aten", "mixed"])
@torch.inference_mode()
def test_weight_snapshot_and_dropped_graph(backend):
    m = Weights()
    ex, audit, req = (inputs(i, 1) for i in [1, 2, 3])
    ep = torch.export.export(m, ex)
    want = m(*req)
    old = m.a.clone()
    run = capture_export(ep, ex, audit, backend=backend)
    assert run.metadata["staging_and_state_bytes"] == 2 * ex[0].nbytes
    m.a.zero_()
    m.a = ex[0].clone()
    m.b = audit[0].clone()
    del ep, m
    gc.collect()
    with pytest.raises(TypeError):
        copy.copy(run)
    with pytest.raises(TypeError):
        copy.deepcopy(run)
    for i in range(20):
        inp = inputs(i, 1)
        got = run(*inp)
        _exact(got, (inp[0] + old) * old, "snapshot")
    final = run(*req)
    ref = weakref.ref(run)
    del run
    gc.collect()
    assert ref() is None
    _exact(final, want, "after dropped handle")


@torch.inference_mode()
def test_stream_and_thread_and_precision_guard():
    m = Visible()
    ex, audit = inputs(8), inputs(9)
    ep = torch.export.export(m, ex)
    caller = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    side.wait_stream(caller)
    with torch.cuda.stream(side):
        run = capture_export(ep, ex, audit, backend="aten")
        with torch.autocast("cuda"), pytest.raises(ValueError, match="precision"):
            run(*ex)
    with pytest.raises(RuntimeError, match="stream"):
        run(*ex)
    with pytest.raises(RuntimeError, match="stream"):
        run.close()
    errors = []

    def other():
        try:
            run(*ex)
        except Exception as e:
            errors.append(e)

    th = threading.Thread(target=other)
    th.start()
    th.join()
    assert len(errors) == 1 and "thread" in str(errors[0])
    req = inputs(10)
    side.wait_stream(caller)
    with torch.cuda.stream(side):
        out = run(*req)
        run.close()
    _exact(out, m(*req), "nondefault owner")


@torch.inference_mode()
def test_invalid_request_is_rejected_before_any_copy():
    m = Visible()
    ex, audit = inputs(1), inputs(2)
    ep = torch.export.export(m, ex)
    with pytest.raises(Unsupported, match="CUDA request"):
        capture_export(ep, tuple(x.cpu() for x in ex), audit)
    with pytest.raises(ValueError, match="content-distinct"):
        capture_export(ep, ex, tuple(x.clone() for x in ex))
    with pytest.raises(ValueError, match="byte limit"):
        capture_export(ep, ex, audit, backend="aten", max_static_bytes=1)
    with capture_export(ep, ex, audit, backend="aten") as run:
        x, a, b = ex
        for invalid in [
            (x, x, b),
            (x.T, a, b),
            (x.flatten()[1:], a, b),
            (x.view(2, -1), a, b),
            (x.half(), a, b),
            (x[:0], a, b),
            (x, a),
            (x.cpu(), a, b),
            (False, a, b),
        ]:
            with pytest.raises((Unsupported, ValueError)):
                run(*invalid)
        _exact(run(*audit), m(*audit), "after invalid calls")


class Bad(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.register_buffer("w", inputs(91, 1)[0])

    def forward(self, x):
        if self.kind == "hidden_base":
            return (x + 1)[:, 1:]
        if self.kind == "exposed_state":
            return self.w.view(-1), x + 1
        if self.kind == "identity":
            return x, x.view(-1)
        if self.kind == "random":
            return torch.rand_like(x) + x
        if self.kind == "mutation":
            return x.add_(1)
        if self.kind == "cpu":
            return x.to("cpu")
        return x + 1


@pytest.mark.parametrize(
    "kind,why",
    [
        ("hidden_base", "no full"),
        ("exposed_state", "model state"),
        ("identity", "alias-only"),
        ("random", "allowlist"),
        ("mutation", "functional|allowlist"),
        ("cpu", "CUDA|allowlist"),
        ("dynamic", "static"),
    ],
)
@torch.inference_mode()
def test_unsupported_graphs(kind, why):
    m = Bad(kind)
    ex, audit = inputs(3, 1), inputs(4, 1)
    ds = ({0: torch.export.Dim("rows", min=2, max=12)},) if kind == "dynamic" else None
    ep = torch.export.export(m, ex, dynamic_shapes=ds)
    with pytest.raises(Unsupported, match=why):
        capture_export(ep, ex, audit, backend="aten")


@torch.inference_mode()
def test_replay_error_poison_and_close(monkeypatch):
    ex, audit = inputs(5), inputs(6)
    ep = torch.export.export(Visible(), ex)
    run = capture_export(ep, ex, audit, backend="aten")

    def fail():
        raise RuntimeError("injected replay error")

    with monkeypatch.context() as patch:
        patch.setattr(run._graph, "replay", fail)
        with pytest.raises(RuntimeError, match="injected"):
            run(*ex)
    with pytest.raises(RuntimeError, match="closed or failed"):
        run(*ex)
    run.close()


class FourInputs(nn.Module):
    def forward(self, x, a, b, c):
        return ((x + a) * b) + c


@torch.inference_mode()
def test_every_argument_staged_independently_and_no_nested_capture():
    m = FourInputs()
    ex = inputs(81, 4)
    audit = inputs(191, 4)
    ep = torch.export.export(m, ex)
    default = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    side.wait_stream(default)
    with (
        torch.cuda.stream(side),
        capture_export(
            ep, ex, audit, backend="mixed", partition_options={"host_backend": "native"}
        ) as run,
    ):
        for i in range(4):
            req = tuple(t if k != i else audit[k] for k, t in enumerate(ex))
            out = run(*req)
            _exact(out, m(*req), "one different input")
            assert not torch.equal(out, m(*ex))
        outer = torch.cuda.CUDAGraph()
        with torch.cuda.graph(outer, stream=side):
            placeholder = ex[0] + 1
            with pytest.raises(RuntimeError, match="nested"):
                run(*ex)
        outer.replay()
        _exact(placeholder, ex[0] + 1, "unrelated capture")
        _exact(run(*audit), m(*audit), "after nested rejection")


@torch.inference_mode()
def test_setup_export_contract_and_source_alias():
    m = Weights()
    ex, audit = inputs(81, 1), inputs(191, 1)
    ep = torch.export.export(m, ex)
    with pytest.raises(Unsupported, match="aliases model state"):
        capture_export(ep, (m.a,), audit, backend="aten")
    with pytest.raises(Unsupported, match="aliases model state"):
        capture_export(ep, ex, (m.b,), backend="aten")
    with pytest.raises(AssertionError, match="Guard failed"):
        capture_export(
            ep,
            tuple(x.view(8, 64) for x in ex),
            tuple(x.view(8, 64) for x in audit),
            backend="aten",
        )
    with torch.autocast("cuda"), pytest.raises(Unsupported, match="autocast"):
        capture_export(ep, ex, audit, backend="aten")
