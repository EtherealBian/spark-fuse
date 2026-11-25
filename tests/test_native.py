import gc
import pytest
import torch
from torch import nn
from examples.graphs import case
from sparkfuse import compile_export
from sparkfuse.measure import compare
from sparkfuse.native import native_dispatch


@pytest.mark.parametrize("name", ["gate", "mean", "resnorm", "resnorm2", "decomp"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_native_fresh_outputs_and_kernel_lifetime(name, dtype):
    m, a = case(name, 13, 113, dtype)
    fn = native_dispatch(compile_export(torch.export.export(m, a)))
    gc.collect()
    y = fn(*a)
    other = fn(*a)
    compare(y, m(*a), exact=(1,) if name == "resnorm2" else ())
    old = y[0] if isinstance(y, tuple) else y
    new = other[0] if isinstance(other, tuple) else other
    assert old.data_ptr() != new.data_ptr()


def test_native_guards():
    m, a = case("gate", 8, 16, torch.float32)
    fn = native_dispatch(compile_export(torch.export.export(m, a)))
    x, y = a
    for bad in (
        x.cpu(),
        x.half(),
        x[:4],
        x.t(),
        x[:, ::2],
        x[0:1].expand_as(x),
        x.detach().requires_grad_(),
        17,
    ):
        with pytest.raises(ValueError):
            fn(bad, y)
    with pytest.raises(ValueError):
        fn(x)


def test_native_lifted_and_nested():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(64, device="cuda"))

        def forward(self, d, *, skip):
            z = d["x"] + skip[0]
            return {"main": z * self.w, "skip": [z]}

    m = M()
    a = {"x": torch.randn(3, 64, device="cuda")}
    kw = {"skip": [torch.randn_like(a["x"])]}
    ep = torch.export.export(m, (a,), kw)
    fn = native_dispatch(compile_export(ep))
    with torch.inference_mode():
        compare(fn(a, **kw), m(a, **kw))


def test_native_capture_and_other_stream():
    m, a = case("resnorm2", 17, 96, torch.bfloat16)
    fn = native_dispatch(compile_export(torch.export.export(m, a)))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        side = fn(*a)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = fn(*a)
        a[0].add_(2)
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    compare(out, m(*a), exact=(1,))


@torch.inference_mode()
def test_mixed_native_region():
    from sparkfuse.partition import partition_export

    class M(nn.Module):
        def forward(self, x, y):
            return torch.nn.functional.silu(x @ y) * x

    m = M()
    a = tuple(torch.randn(16, 16, device="cuda") for _ in range(2))
    gm, report = partition_export(torch.export.export(m, a), host_backend="native")
    assert any(r.get("host_backend") == "native" for r in report)
    compare(gm(*a), m(*a))
