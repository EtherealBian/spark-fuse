import pytest
import torch
from torch import nn
from sparkfuse import Schedule, compile_export
from sparkfuse.measure import compare
from sparkfuse.partition import partition_export


class Gate(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, x, up):
        v = torch.nn.functional.gelu(x, approximate=self.mode)
        return v * up, v


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["none", "tanh"])
@pytest.mark.parametrize("shift", [0, 1])
def test_fused_gelu_observable_activation(dtype, mode, shift):
    g = torch.Generator(device="cuda").manual_seed(7243)
    args = tuple(
        torch.randn(13 * 129 + shift, device="cuda", dtype=dtype, generator=g)[shift:].reshape(
            13, 129
        )
        for _ in range(2)
    )
    args[0][0, :9] = args[0].new_tensor(
        [float("-inf"), -15.0, -5.0, -0.0, 0.0, 5.0, 15.0, float("inf"), float("nan")]
    )
    m = Gate(mode)
    k = compile_export(torch.export.export(m, args), schedule=Schedule(128, "flat", False, 4))
    out = k(*args)
    ref = m(*args)
    compare(out, ref, exact=(0, 1) if dtype != torch.float32 else ())
    assert torch.equal(out[1][0, :9].signbit()[1:7], ref[1][0, :9].signbit()[1:7])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cubic_rounds_each_product(dtype):
    class Cube(nn.Module):
        def forward(self, x):
            return x.pow(3.0), x.pow(3.0) + 1

    g = torch.Generator(device="cuda").manual_seed(32769)
    x = torch.randn(37, 769, dtype=dtype, device="cuda", generator=g)
    ep = torch.export.export(Cube(), (x,))
    gm, d = partition_export(ep)
    assert len([r for r in d if "module" in r]) == 1
    compare(gm(x), Cube()(x), exact=(0, 1))
    if dtype != torch.float32:
        assert (x.pow(3.0) != x.float().pow(3.0).to(dtype)).any()
