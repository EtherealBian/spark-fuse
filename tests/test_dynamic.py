import pytest
import torch
from examples.graphs import case
from sparkfuse.dynamic import DynamicRows
from sparkfuse import Schedule, Unsupported
from sparkfuse.measure import compare


@pytest.mark.parametrize("name", ["gate", "resnorm2", "mean", "decomp"])
def test_one_binary_for_all_row_counts(name, monkeypatch):
    m, a = case(name, 7, 128, torch.bfloat16)
    dim = torch.export.Dim("tokens", min=1, max=129)
    dyn = tuple({0: dim} if x.ndim == 2 else None for x in a)
    ep = torch.export.export(m, a, dynamic_shapes=dyn)
    runner = DynamicRows(
        ep,
        schedule=Schedule(
            128, "flat" if name == "gate" else "warp", name not in ("mean", "gate"), 4
        ),
    )

    def no_build(*args, **kwargs):
        raise AssertionError("recompilation")

    monkeypatch.setattr("sparkfuse.runtime.build", no_build)
    for rows in (1, 2, 31, 32, 47, 48, 97, 129):
        _, inputs = case(name, rows, 128, torch.bfloat16, seed=400 + rows)
        compare(runner(*inputs), m(*inputs), exact=(1,) if name == "resnorm2" else ())
    _, inputs = case(name, 130, 128, torch.bfloat16)
    with pytest.raises(ValueError, match="bounds"):
        runner(*inputs)
    if name == "gate":
        with pytest.raises(ValueError, match="shape"):
            runner(a[0], a[1][:-1])


def test_reject_two_axes():
    m, a = case("gate", 7, 128, torch.float32)
    dim = torch.export.Dim("tokens", min=1, max=129)
    h = torch.export.Dim("h", min=8, max=256)
    s = {0: dim, 1: h}
    with pytest.raises(Unsupported, match="exactly one"):
        DynamicRows(torch.export.export(m, a, dynamic_shapes=(s, s)))
