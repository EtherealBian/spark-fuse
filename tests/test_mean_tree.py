import pytest
import torch
from torch import nn
from examples.graphs import case
from sparkfuse import Schedule, Unsupported, compile_export, generate, lower
from sparkfuse.dynamic import DynamicRows
from sparkfuse.mean_tree import eligible, sensitive
from sparkfuse.partition import ordered_collectives
from sparkfuse.tuning import candidates


class FullTrace(nn.Module):
    def forward(self, x, skip, w):
        r = x + skip
        f = r.float()
        mu = f.square().mean(-1, keepdim=True)
        sc = torch.rsqrt(mu + 1e-6)
        n = f * sc
        low = n.to(x.dtype)
        return mu, sc, n, low, w * low, r


def shifted(x, offset):
    return torch.cat([x.new_zeros(offset), x.flatten()])[offset:].view_as(x) if offset else x


def same_bits(actual, ref):
    for v, t in zip(actual, ref, strict=True):
        assert v.dtype == t.dtype and v.shape == t.shape
        typ = torch.int32 if v.element_size() == 4 else torch.int16
        assert torch.equal(v.view(typ), t.view(typ))


@pytest.mark.parametrize(
    "rows,width,dt,t,cache",
    [
        (48, 8960, torch.bfloat16, 128, False),
        (48, 8960, torch.bfloat16, 256, True),
        (48, 8960, torch.bfloat16, 512, False),
        (17, 8160, torch.float16, 128, True),
        (17, 8164, torch.float16, 128, True),
        (31, 65536, torch.bfloat16, 512, True),
        (16, 128, torch.bfloat16, 512, False),
    ],
)
@torch.inference_mode()
def test_order_and_observability(rows, width, dt, t, cache):
    m = FullTrace()
    _, a = case("decomp", rows, width, dt, 1753)
    ep = torch.export.export(m, a)
    k = compile_export(ep, schedule=Schedule(t, "mean_tree", cache, 4))
    assert sensitive(k.plan) and all(s.layout == "mean_tree" for s in candidates(k.plan))
    for seed, offsets in [(1753, (0, 0, 0)), (29172, (1, 2, 3))]:
        _, v = case("decomp", rows, width, dt, seed)
        v = tuple(shifted(x, off) for x, off in zip(v, offsets))
        same_bits(k(*v), m(*v))


@torch.inference_mode()
def test_original_default_reproduces_midpoint_new_default_does_not():
    m, a = case("decomp", 48, 8960, torch.bfloat16, 1753)
    ep = torch.export.export(m, a)
    fix = compile_export(ep)
    old = compile_export(ep, schedule=Schedule(128, "block", True, 1))
    ref = m(*a)
    if torch.__version__ == "2.13.0+cu130" and "GB10" in torch.cuda.get_device_name():
        assert old(*a)[25, 3533].item() == -2.578125 and ref[25, 3533].item() == -2.546875
    assert fix.schedule == ordered_collectives(fix.plan)
    same_bits((fix(*a),), (ref,))


@pytest.mark.parametrize(
    "kind,rows,width",
    [
        ("decomp", 15, 8960),
        ("decomp", 48, 8961),
        ("resnorm2", 48, 8960),
        ("mean", 48, 8960),
        ("plain", 48, 8960),
        ("decomp", 16, 68),
    ],
)
def test_mean_tree_guards(kind, rows, width):
    if kind == "plain":

        class Plain(nn.Module):
            def forward(self, x):
                return x.mean(-1, keepdim=True)

        m = Plain()
        a = (torch.empty(rows, width, dtype=torch.float32),)
    else:
        m, a = case(kind, rows, width, torch.bfloat16)
    p = lower(torch.export.export(m, a))
    assert not eligible(p)
    with pytest.raises(Unsupported, match="mean_tree"):
        generate(p, Schedule(512, "mean_tree", False, 4))


@torch.inference_mode()
def test_bounded_rows_and_source_layout():
    m, a = case("decomp", 19, 512, torch.float16, 257)
    dim = torch.export.Dim("rows", min=16, max=49)

    def make(d):
        return torch.export.export(m, a, dynamic_shapes=({0: d}, {0: d}, None))

    k = DynamicRows(make(dim), schedule=Schedule(256, "mean_tree", True, 4)).native_dispatch()
    for r in (16, 19, 49):
        _, v = case("decomp", r, 512, torch.float16, r)
        v = tuple(shifted(x, 1) for x in v)
        same_bits((k(*v),), (m(*v),))
    with pytest.raises(Unsupported, match="row bound"):
        DynamicRows(
            make(torch.export.Dim("many", min=16, max=40000)),
            schedule=Schedule(256, "mean_tree", True, 4),
        )
