import pytest
import torch
from examples.graphs import case
from sparkfuse import Schedule, Unsupported, compile_export, generate, lower
from sparkfuse.dynamic import DynamicRows
from sparkfuse.measure import compare


def offset(x):
    return torch.cat([x.new_zeros(1), x.flatten()])[1:].view_as(x)


class ReducedStatistics(torch.nn.Module):
    def forward(self, x, weight, bias):
        # Only one reduction. Reduce a broadcast expression; feed both compact outputs.
        mu = (x * weight).float().square().mean(-1, keepdim=True)
        return mu + bias, torch.rsqrt(mu + 1e-5)


@pytest.mark.parametrize(
    "h,t,vec,dt",
    [
        (7, 32, 1, torch.float32),
        (8, 32, 4, torch.float16),
        (33, 64, 4, torch.bfloat16),
        (769, 128, 2, torch.float32),
        (1536, 256, 4, torch.bfloat16),
        (4096, 1024, 4, torch.float16),
    ],
)
@torch.inference_mode()
def test_reduced_epilogues_multiple_input_alignments(h, t, vec, dt):
    m = ReducedStatistics()
    torch.manual_seed(911)
    a = (
        torch.randn(2, 3, h, device="cuda", dtype=dt),
        torch.randn(h, device="cuda", dtype=dt),
        torch.randn(2, 3, 1, device="cuda"),
    )
    k = compile_export(torch.export.export(m, a), schedule=Schedule(t, "compact_mean", False, vec))
    for mask in (0, 1, 2, 3):
        args = tuple(offset(x) if (1 << i) & mask else x for i, x in enumerate(a))
        y = k(*args)
        compare(y, m(*args))
        assert y[0].shape == y[1].shape == (2, 3, 1)


@torch.inference_mode()
def test_compact_mean_symbolic_rows():
    m, a = case("mean", 7, 2048, torch.bfloat16, seed=9881)
    dim = torch.export.Dim("rows", min=1, max=129)
    ep = torch.export.export(m, a, dynamic_shapes=({0: dim},))
    run = DynamicRows(ep, schedule=Schedule(256, "compact_mean", False, 8)).native_dispatch()
    for rows in (1, 31, 129):
        _, b = case("mean", rows, 2048, torch.bfloat16, seed=1002 + rows)
        x = offset(b[0])
        compare(run(x), m(x))


@pytest.mark.parametrize(
    "case_name,width,cache",
    [("gate", 32, False), ("resnorm2", 32, False), ("mean", 1, False), ("mean", 32, True)],
)
def test_compact_reject_noncompact(case_name, width, cache):
    m, a = case(case_name, 5, width, torch.float32)
    p = lower(torch.export.export(m, a))
    with pytest.raises(Unsupported, match="compact_mean"):
        generate(p, Schedule(128, "compact_mean", cache, 1))


@torch.inference_mode()
def test_compact_special_rows_and_candidates():
    from sparkfuse.tuning import candidates

    m, a = case("mean", 7, 14336, torch.bfloat16, seed=151)
    a[0][0, 0] = float("nan")
    a[0][1, -1] = float("-inf")
    a[0][2, 13] = float("inf")
    a[0][3].zero_()
    a[0][4].fill_(0.25)
    ep = torch.export.export(m, a)
    p = lower(ep)
    schedules = candidates(p)
    assert (
        Schedule(1024, "block", False, 8) in schedules
        and Schedule(1024, "compact_mean", False, 8) in schedules
    )
    kernel = compile_export(ep, schedule=Schedule(1024, "compact_mean", False, 8))
    y = kernel(*a)
    ref = m(*a)
    quality = compare(y, ref)[0]
    assert quality["matched_nan"] == 1 and quality["matched_posinf"] == 2
    m2, b = case("resnorm2", 3, 128, torch.bfloat16)
    assert not any(
        s.layout == "compact_mean" for s in candidates(lower(torch.export.export(m2, b)))
    )
