import pytest
import torch
from examples.graphs import case
from sparkfuse import Schedule, compile_export
from sparkfuse.measure import compare


@pytest.mark.parametrize(
    "name,rows,width,sch",
    [
        ("gate", 13, 3, Schedule(128, "flat", False, 8)),
        ("gate", 47, 1537, Schedule(64, "flat", False, 8)),
        ("resnorm2", 13, 769, Schedule(128, "block", True, 8)),
        ("decomp", 9, 127, Schedule(64, "warp", True, 4)),
        ("mean", 35, 1, Schedule(64, "warp", False, 8)),
        ("mean", 65, 113, Schedule(128, "block", False, 4)),
    ],
)
def test_masked_unaligned_tail(name, rows, width, sch):
    m, orig = case(name, rows, width, torch.bfloat16)
    a = tuple(torch.cat((t.new_full((1,), float("nan")), t.flatten()))[1:].view_as(t) for t in orig)
    k = compile_export(torch.export.export(m, a), schedule=sch)
    compare(k(*a), m(*a), exact=(1,) if name == "resnorm2" else ())


@pytest.mark.parametrize("bias_shape", ["channel", "scalar", "full"])
def test_linear_packets_only_across_legal_domains(bias_shape):
    class M(torch.nn.Module):
        def forward(self, x, b):
            return x.relu() + b

    x = torch.randn(33, 15, device="cuda")
    b = torch.randn({"channel": (15,), "scalar": (), "full": (33, 15)}[bias_shape], device="cuda")
    m = M()
    k = compile_export(torch.export.export(m, (x, b)), schedule=Schedule(64, "flat", False, 4))
    assert ("first=packet*4" in k.source) == (bias_shape != "channel")
    compare(k(x, b), m(x, b))
