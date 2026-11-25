import pytest, torch
from torch import nn
from examples.graphs import case, ResidualNorm
from sparkfuse import compile_export, Schedule, Unsupported, lower, generate
from sparkfuse.measure import compare


@pytest.mark.parametrize("name", ["resnorm", "resnorm2", "gate", "decomp", "mean"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "sch",
    [
        Schedule(64, "block"),
        Schedule(128, "warp"),
        Schedule(128, "warp", False, 4),
        Schedule(256, "block", False, 4),
    ],
)
def test_fused(name, dtype, sch):
    m, a = case(name, 11, 128, dtype)
    ep = torch.export.export(m, a)
    k = compile_export(ep, schedule=sch)
    compare(k(*a), m(*a), exact=(1,) if name == "resnorm2" else ())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("name", ["resnorm2", "gate", "decomp"])
def test_offset_and_stream(name, dtype):
    m, a = case(name, 7, 128, dtype)
    a = tuple(torch.cat((x.new_zeros(1), x.flatten()))[1:].view(x.shape) for x in a)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    ep = torch.export.export(m, a)
    s = Schedule(64, "flat" if name == "gate" else "warp", name != "gate", 4)
    k = compile_export(ep, schedule=s)
    with torch.cuda.stream(stream):
        y = k(*a)
    torch.cuda.current_stream().wait_stream(stream)
    compare(y, m(*a), exact=(1,) if name == "resnorm2" else ())


@pytest.mark.parametrize("name", ["resnorm", "resnorm2", "decomp", "mean", "gate"])
def test_tail(name):
    m, a = case(name, 7, 113, torch.float32)
    ep = torch.export.export(m, a)
    k = compile_export(ep, schedule=Schedule(64, "warp", name != "gate"))
    compare(k(*a), m(*a))
    with pytest.raises(Unsupported):
        generate(k.plan, Schedule(64, "warp", False, 4))


def test_visibility():
    _, a = case("resnorm", 3, 64, torch.float32)
    plans = [lower(torch.export.export(ResidualNorm(show), a)) for show in [False, True]]
    p, q = plans
    assert p.vals == q.vals and len(p.outputs) == 1 and len(q.outputs) == 2
    assert "out1[" not in generate(p, Schedule()) and "out1[" in generate(q, Schedule())


def test_unsupported_and_guard():
    class MM(nn.Module):
        def forward(self, x):
            return x @ x

    x = torch.randn(16, 16, device="cuda")
    with pytest.raises(Unsupported, match="unsupported operator"):
        lower(torch.export.export(MM(), (x,)))
    m, a = case("gate", 2, 16, torch.float32)
    k = compile_export(torch.export.export(m, a))
    for bad in (a[0].cpu(), a[0].half(), a[0].t(), a[0][:, :-1], a[0][:1].expand_as(a[0])):
        with pytest.raises(ValueError):
            k(bad, a[1])


def test_weights_and_trees():
    class W(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(64, device="cuda"))

        def forward(self, x):
            return {"gate": x * self.w, "other": (-x,)}

    m = W()
    a = torch.randn(3, 4, 64, device="cuda")
    k = compile_export(torch.export.export(m, (a,)))
    with torch.inference_mode():
        compare(k(a), m(a))
