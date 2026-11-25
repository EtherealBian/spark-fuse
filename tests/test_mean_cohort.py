import pytest
import torch
from torch import nn
from examples.graphs import case
from sparkfuse import Schedule, Unsupported, compile_export, generate, lower
from sparkfuse.dynamic import DynamicRows
from sparkfuse.mean_tree import cohort_eligible
from sparkfuse.tuning import candidates
from test_mean_tree import FullTrace, same_bits, shifted


@pytest.mark.parametrize(
    "h,dt",
    [
        (1024, torch.float16),
        (1536, torch.bfloat16),
        (4096, torch.bfloat16),
        (8156, torch.bfloat16),
        (8160, torch.float16),
    ],
)
@pytest.mark.parametrize("threads,cache", [(128, False), (128, True), (256, True), (512, True)])
@torch.inference_mode()
def test_cohort_all_observable_rounding_boundaries(h, dt, threads, cache):
    m = FullTrace()
    _, a = case("decomp", 17, h, dt, 5471)
    ep = torch.export.export(m, a)
    k = compile_export(ep, schedule=Schedule(threads, "mean_cohort", cache, 4))
    assert len(candidates(k.plan)) == 10
    for seed, offsets in [(61343, (0, 0, 0)), (84397, (1, 2, 3))]:
        _, a = case("decomp", 17, h, dt, seed)
        v = tuple(shifted(x, o) for x, o in zip(a, offsets))
        same_bits(k(*v), m(*v))


class Products(nn.Module):
    def forward(self, x, w, scale):
        m = (x * w).mean(-1, keepdim=True)
        return (m + scale), x * m


@torch.inference_mode()
def test_rank3_broadcasts_and_mixed_output():
    g = torch.Generator(device="cuda").manual_seed(1397)
    x, w, scale = tuple(
        torch.randn(*s, device="cuda", generator=g) for s in [(3, 7, 1540), (1, 7, 1540), (3, 7, 1)]
    )
    m = Products()
    v = tuple(shifted(a, i) for i, a in enumerate((x, w, scale), 1))
    ep = torch.export.export(m, v)
    for t, c in [(128, False), (256, True)]:
        same_bits(compile_export(ep, schedule=Schedule(t, "mean_cohort", c, 4))(*v), m(*v))


@pytest.mark.parametrize(
    "rows,w,threads,cache",
    [
        (17, 1020, 256, True),
        (17, 8164, 256, True),
        (17, 1024, 256, False),
        (15, 1024, 128, True),
        (32, 1537, 128, True),
    ],
)
def test_unsupported_cohort(rows, w, threads, cache):
    m, a = case("decomp", rows, w, torch.bfloat16)
    p = lower(torch.export.export(m, a))
    with pytest.raises(Unsupported, match="mean_cohort"):
        generate(p, Schedule(threads, "mean_cohort", cache, 4))
    if threads != 256 or cache:
        assert not cohort_eligible(p)


@torch.inference_mode()
def test_dynamic_native_guard_and_bitwise():
    m, a = case("decomp", 19, 1536, torch.bfloat16, 5411)

    def exp(hi):
        rows = torch.export.Dim("rows", min=16, max=hi)
        return torch.export.export(m, a, dynamic_shapes=({0: rows}, {0: rows}, None))

    k = DynamicRows(exp(48))
    assert k.kernel.schedule == Schedule(256, "mean_cohort", True, 4)
    fast = k.native_dispatch()
    for n in (16, 17, 48):
        _, v = case("decomp", n, 1536, torch.bfloat16, 9097 + n)
        same_bits((fast(*v),), (m(*v),))
    with pytest.raises(Unsupported, match="row bound"):
        DynamicRows(exp(12000), schedule=Schedule(256, "mean_cohort", True, 4))
