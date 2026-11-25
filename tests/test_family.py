"""Immutable cross-shape state, explicit admission, transactional failure and fresh results."""

import copy
import gc
import pytest
import torch
from torch import nn
from sparkfuse.contextual import _exact
from sparkfuse.family import capture_family
from sparkfuse.ir import Unsupported
from sparkfuse.resident import _capture_export, _get, capture_export

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def req(rows, seed):
    return (
        torch.randn(
            rows, 128, device="cuda", generator=torch.Generator(device="cuda").manual_seed(seed)
        ),
    )


def pairs(rows=(2, 7, 17)):
    return [(req(n, 111 + i), req(n, 881 + i)) for i, n in enumerate(rows)]


class Layer(nn.Module):
    def __init__(self, tied=True):
        super().__init__()
        w = req(1, 478)[0].view(128)
        self.register_buffer("a", w)
        self.register_buffer("b", w.detach() if tied else w.clone())

    def forward(self, x):
        y = (x + self.a) * self.b
        return {"h": y, "slice": y[:, 3:], "x": x}


@pytest.mark.parametrize("backend", ["mixed", "aten"])
@torch.inference_mode()
def test_frozen_family_interleaved_kept_results_and_single_storage(backend):
    model = Layer().eval()
    rs = pairs()
    expected = sum(x[0].nbytes for x, _ in rs) + 512
    old = model.a.clone()
    family = capture_family(
        model,
        rs,
        backend=backend,
        max_static_bytes=expected,
        partition_options={"host_backend": "native"} if backend == "mixed" else None,
    )
    with family as f:
        assert (
            f.metadata["weight_snapshot_bytes"] == 512
            and f.metadata["named_state"] == 2
            and f.metadata["static_bytes"] == expected
        )
        states = [
            [_get(h._fn, n.target) for n in h._fn.graph.nodes if n.op == "get_attr"]
            for h in f._handles.values()
        ]
        assert (
            len({t.untyped_storage().data_ptr() for ts in states for t in ts}) == 1
            and states[0][0].data_ptr() != model.a.data_ptr()
        )
        model.b.add_(83)
        model.a = old.clone()
        del model
        gc.collect()
        held = []
        for j in range(30):
            x = req((2, 17, 7)[j % 3], j)[0]
            out = f(x)
            expected = (x + old) * old
            held.append((out, expected))
            assert out["x"] is x
            assert (
                out["slice"].storage_offset() == 3
                and out["slice"].untyped_storage().data_ptr() == out["h"].data_ptr()
            )
        assert len({o["h"].data_ptr() for o, _ in held}) == 30
        for o, e in held:
            _exact(o["h"], e, "interleaved")
        with pytest.raises(TypeError):
            copy.copy(f)
        with pytest.raises(TypeError):
            copy.deepcopy(f)
        with torch.cuda.stream(torch.cuda.Stream()), pytest.raises(RuntimeError, match="stream"):
            f.close()
        _exact(f(held[0][0]["x"])["h"], held[0][1], "after wrong close")
    family.close()
    _exact(held[-1][0]["h"], held[-1][1], "after close")
    with pytest.raises(RuntimeError, match="closed"):
        family(held[0][0]["x"])


@torch.inference_mode()
def test_budget_duplicates_and_miss_never_export(monkeypatch):
    model = Layer().eval()
    rs = pairs((2, 7))
    cap = 512 + sum(x[0].nbytes for x, _ in rs)
    with pytest.raises(ValueError, match="byte limit"):
        capture_family(model, rs, backend="aten", max_static_bytes=cap - 1)
    with pytest.raises(ValueError, match="duplicate"):
        capture_family(model, rs + rs[:1], backend="aten")
    with pytest.raises(ValueError, match="1 and 16"):
        capture_family(model, rs * 9)
    with pytest.raises(Unsupported, match="eval model"):
        capture_family(model.train(), rs)
    with capture_family(model.eval(), rs, backend="aten", max_static_bytes=cap) as f:

        def unexpected(*a, **kw):
            raise AssertionError("cannot export on service path")

        monkeypatch.setattr(torch.export, "export", unexpected)
        for x in (req(3, 19)[0], rs[0][0][0].half()):
            with pytest.raises(Unsupported, match="not admitted"):
                f(x)
        with pytest.raises(Unsupported, match="contiguous"):
            f(torch.empty(2, 129, device="cuda")[:, :128])
        with pytest.raises(Unsupported, match="capture device"):
            f(rs[0][0][0].cpu())
        y = f(*rs[0][1])
        _exact(y, model(*rs[0][1]), "after misses")


@pytest.mark.parametrize("mode", ["value", "signed_zero", "untying", "tying"])
@torch.inference_mode()
def test_subsequent_snapshot_requires_bits_and_alias_groups(mode):
    m = Layer(tied=mode != "tying").eval()
    m.a[0] = 0.0
    rs = pairs((2, 7))
    (ex, audit), (ex2, audit2) = rs
    with capture_export(torch.export.export(m, ex), ex, audit, backend="aten") as run:
        shared = {
            n.target: _get(run._fn, n.target) for n in run._fn.graph.nodes if n.op == "get_attr"
        }
        if mode == "value":
            m.a[1] += 1
        elif mode == "signed_zero":
            m.a[0] = -0.0
        elif mode == "untying":
            m.b = m.a.clone()
        else:
            m.b = m.a.detach()
        with pytest.raises(Unsupported, match="state (contents|alias groups)"):
            _capture_export(torch.export.export(m, ex2), ex2, audit2, "aten", None, 1024**2, shared)
        _exact(run(*audit)["h"], (audit[0] + shared["a"]) * shared["b"], "original still valid")


class ShapeBranch(Layer):
    def forward(self, x):
        return x + (self.a if x.shape[0] < 5 else self.b)


@torch.inference_mode()
def test_failed_admission_closes_preceding_graph(monkeypatch):
    from sparkfuse.resident import ResidentExecutable

    calls = []
    close = ResidentExecutable.close

    def watch(self):
        close(self)
        calls.append(self._graph is None)

    monkeypatch.setattr(ResidentExecutable, "close", watch)
    with pytest.raises(Unsupported, match="named state"):
        capture_family(ShapeBranch(tied=False).eval(), pairs((2, 7)), backend="aten")
    assert calls == [True]
