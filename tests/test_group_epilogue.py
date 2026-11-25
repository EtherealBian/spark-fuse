"""Horizontal fusion of independent signed dequantization branches; no GEMM or ABI changes."""

import pytest
import torch
from torch import nn
from sparkfuse.partition import partition_export


def check(ys, ref):
    assert len(ys) == len(ref)
    for a, b in zip(ys, ref, strict=True):
        assert (a.shape, a.dtype, a.stride(), a.storage_offset()) == (
            b.shape,
            b.dtype,
            b.stride(),
            b.storage_offset(),
        )
        assert torch.equal(
            a.contiguous().reshape(-1).view(torch.uint8),
            b.contiguous().reshape(-1).view(torch.uint8),
        )
    stores = [t.untyped_storage().data_ptr() for t in ys]
    old = [t.untyped_storage().data_ptr() for t in ref]
    assert [[a == b for b in stores] for a in stores] == [[a == b for b in old] for a in old]


class Many(nn.Module):
    def __init__(self, n, visible=False):
        super().__init__()
        self.n = n
        self.visible = visible

    def forward(self, row, *branches):
        out = []
        for j in range(self.n):
            acc, ch = branches[j * 2 : j * 2 + 2]
            cast = acc.float()
            b = cast * row
            y = b * ch
            if self.visible:
                if j % 2 == 0:
                    out.append(cast)
                if j % 3 == 0:
                    out.append(b)
            out.append(y)
        return tuple(out)


def inputs(count, rows=11):
    gen = torch.Generator(device="cuda").manual_seed(1171397 + count)
    row = torch.rand(rows + 1, device="cuda", generator=gen)[1:, None]
    args = [row]
    row[:5, 0] = torch.tensor(
        [0, -(2**31), 1, 0x7FC12345, 0x7F800000], device="cuda", dtype=torch.int32
    ).view(torch.float32)
    for j, n in enumerate([7, 131, 65, 1, 211, 17][:count]):
        acc = torch.randint(
            -(2**31), 2**31 - 1, (rows * n + 1,), device="cuda", dtype=torch.int32, generator=gen
        )[1:].view(rows, n)
        acc[0, 0] = 2**24 + 1
        ch = torch.randn(n + 1, device="cuda", generator=gen)[1:]
        ch[0] = torch.tensor(0x7F800123, device="cuda", dtype=torch.int32).view(torch.float32)
        args.extend([acc, ch if j % 2 else ch[None, :]])
    return tuple(args)


@pytest.mark.parametrize("count", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("host", ["native", "python"])
@torch.inference_mode()
def test_grouped_live_outputs_replay(host, count):
    mod = Many(count, True)
    args = inputs(count)
    ep = torch.export.export(mod, args)
    gm, recs = partition_export(
        ep, host_backend=host, projection_epilogue=True, group_epilogues=True
    )
    group = [r for r in recs if r.get("kind") == "grouped_int32_epilogue"]
    single = [r for r in recs if r.get("kind") == "ordered_int32_epilogue"]
    assert [len(x["branches"]) for x in group] == (
        {2: [2], 3: [3], 4: [4], 5: [4], 6: [4, 2]}[count]
    )
    assert len(single) == (count == 5)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        before = mod(*args)
        old = gm(*args)
        check(old, before)
        cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cg, stream=stream):
            out = gm(*args)
        for step in (-0.5, 3.0):
            args[0].fill_(step)
            cg.replay()
            check(out, mod(*args))
            check(old, before)
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.parametrize(
    "case", ["dependent_channel", "early_reader", "overlap", "other_row", "one"]
)
@torch.inference_mode()
def test_decline_grouping_without_independent_legal_region(case):
    class Bad(nn.Module):
        def forward(self, a, b, row, row2, c):
            f = a.float()
            t = f * row
            y = t * c
            if case == "one":
                return (y,)
            if case == "dependent_channel":
                channel = y[0]
                src = b
                other = row
            elif case == "early_reader":
                tmp = y + 1
                src = (tmp + b).to(torch.int32)
                channel = c
                other = row
            else:
                channel = c
                src = b
                other = row2 if case == "other_row" else row
            z = t * channel if case == "overlap" else (src.float() * other) * channel
            return y, z

    shapes = (7, 67)
    g = torch.Generator(device="cuda").manual_seed(1058413)
    a, b = [
        torch.randint(-1000, 1000, shapes, dtype=torch.int32, device="cuda", generator=g)
        for _ in range(2)
    ]
    r, s = [torch.rand(7, 1, device="cuda", generator=g) for _ in range(2)]
    c = torch.rand(67, device="cuda", generator=g)
    mod = Bad()
    args = (a, b, r, s, c)
    gm, recs = partition_export(
        torch.export.export(mod, args), host_backend="native", group_epilogues=True
    )
    assert not any(z.get("kind") == "grouped_int32_epilogue" for z in recs)
    check(gm(*args), mod(*args))
