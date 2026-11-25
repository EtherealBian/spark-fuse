import pytest
import torch
from torch import nn
from sparkfuse.partition import partition_export


def bits(t):
    return t.contiguous().reshape(-1).view(torch.uint8)


def exact(xs, ys):
    for x, y in zip(xs, ys, strict=True):
        assert (x.shape, x.stride(), x.storage_offset(), x.dtype) == (
            y.shape,
            y.stride(),
            y.storage_offset(),
            y.dtype,
        ) and torch.equal(bits(x), bits(y))
    for i in range(len(xs)):
        for j in range(i):
            assert (xs[i].untyped_storage().data_ptr() == xs[j].untyped_storage().data_ptr()) == (
                ys[i].untyped_storage().data_ptr() == ys[j].untyped_storage().data_ptr()
            )
            assert (xs[i] is xs[j]) == (ys[i] is ys[j])


class Tail(nn.Module):
    def __init__(self, vis):
        super().__init__()
        self.vis = vis

    def forward(self, acc, row, chan):
        f = acc.float()
        p = f * row
        y = p * chan
        return (
            *((f,) if "f" in self.vis else ()),
            *((p,) if "p" in self.vis else ()),
            y,
            y[:, 1:],
            acc,
            y,
        )


def build(m, args, host="native"):
    gm, rs = partition_export(
        torch.export.export(m, args), host_backend=host, projection_epilogue=True
    )
    return gm, [r for r in rs if r.get("kind") == "ordered_int32_epilogue"]


def inputs(m, n, vec=False):
    g = torch.Generator(device="cuda").manual_seed(321791)
    acc = torch.randint(
        -(2**31), 2**31 - 1, (m * n + 1,), device="cuda", generator=g, dtype=torch.int32
    )[1:].view(m, n)
    row = torch.randn(m + 1, device="cuda", generator=g)[1:, None]
    chan = torch.randn(n + 3, device="cuda", generator=g)[3:]
    return (acc, row, chan if vec else chan[None, :])


@pytest.mark.parametrize("host", ["native", "python"])
@pytest.mark.parametrize("vis", ["", "f", "p", "fp"])
@torch.inference_mode()
def test_visible_exact_order_and_replay(host, vis):
    a, r, c = inputs(11, 67, vis in ("f", "p"))
    m = Tail(vis)
    gm, (site,) = build(m, (a, r, c), host)
    assert site["output_roles"] == (["cast"] if "f" in vis else []) + (
        ["row_scaled"] if "p" in vis else []
    ) + ["output"]
    a[:, 0:7] = torch.tensor(
        [-(2**31), 2**31 - 1, 2**24 + 1, 2**24 + 3, -(2**24) - 1, 0, 1],
        device="cuda",
        dtype=torch.int32,
    )
    sr = torch.tensor(
        [0, -(2**31), 1, 0x7F800000, -0x00800000, 0x7F800001, 0x7FC12345, -0x003FFFFF],
        device="cuda",
        dtype=torch.int32,
    ).view(torch.float32)
    r[: len(sr), 0] = sr
    c.reshape(-1)[: len(sr)] = sr.flip([0])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        untouched = a.clone()
        orig = gm(a, r, c)
        expected = m(a, r, c)
        exact(orig, expected)
        assert torch.equal(a, untouched)
        cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cg, stream=stream):
            cap = gm(a, r, c)
        for s in (0.25, -3.5):
            c.fill_(s)
            cg.replay()
            exact(cap, m(a, r, c))
        exact(orig, expected)
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.parametrize(
    "reason",
    [
        "reassociate",
        "first_operand",
        "last_operand",
        "strided",
        "wrongdtype",
        "transpose",
        "two_rows",
        "early",
        "dependency",
        "self_row",
        "self_channel",
        "delayed_channel",
    ],
)
@torch.inference_mode()
def test_reject_unproven_order_or_topology(reason):
    class M(nn.Module):
        def forward(self, a, r, c):
            f = a.float()
            if reason == "reassociate":
                return (f * (r * c),)
            if reason == "dependency":
                return ((f * r) * f[:1, :],)
            if reason == "self_row":
                return ((f * f) * c,)
            p = r * f if reason == "first_operand" else f * r
            if reason == "self_channel":
                return (p * p,)
            if reason == "early":
                v = p[:, 1:]
            if reason == "delayed_channel":
                c = c + 1.25
            y = c * p if reason == "last_operand" else p * c
            return (y, v) if reason == "early" else (y,)

    a, r, c = inputs(1 if reason == "self_channel" else 8, 1 if reason == "self_row" else 35)
    m = M()
    if reason == "strided":
        c = torch.ones(1, 2 * c.numel(), device="cuda")[:, ::2]
    if reason == "wrongdtype":
        a = a.to(torch.int64)
    if reason == "transpose":
        a = a.t().contiguous().t()
    if reason == "two_rows":
        r = r.expand_as(a).contiguous()
    gm, sites = build(m, (a, r, c))
    assert bool(sites) == (reason == "delayed_channel")
    exact(gm(a, r, c), m(a, r, c))


def test_storage_dtype_guards():
    a, r, c = inputs(9, 17)
    gm, (site,) = build(Tail(""), (a, r, c))
    op = getattr(gm, site["module"])
    for bad in (a.float(), a.to(torch.int8), a[:1], a.cpu()):
        with pytest.raises(ValueError):
            op(bad, r, c)
    with pytest.raises(ValueError):
        op(a, r.detach().requires_grad_(True), c)
    from sparkfuse.ir import DT

    assert torch.int32 not in DT and torch.int8 not in DT


@torch.inference_mode()
def test_exact_independent_ieee_rounding_and_denormals():
    # Two binary32 operands have an exact product in binary64; pack rounds only at the specified points.
    import struct

    def rn(t):
        return struct.unpack("<f", struct.pack("<f", t))[0]

    vals = [2**24 - 1, 2**24 + 1, 2**24 + 3, -(2**24 + 3), 2**31 - 1, -(2**31), 0]
    rows = [rn(0.10000001), rn(1.0000001192092896), -0.0, rn(2.0**-149)]
    cols = [
        rn(0.3333333432674408),
        rn(3.125),
        rn(-0.0078125),
        rn(1.0000001192092896),
        rn(1.9999998807907104),
        rn(0.5),
        rn(-0.0),
    ]
    acc = torch.tensor([vals] * len(rows), device="cuda", dtype=torch.int32)
    r = torch.tensor(rows, device="cuda", dtype=torch.float32)[:, None]
    c = torch.tensor(cols, device="cuda", dtype=torch.float32)
    gm, (site,) = build(Tail("fp"), (acc, r, c))
    f, first, y, *_ = gm(acc, r, c)
    ref = [[], [], []]
    different_order = False
    for rv in rows:
        for av, cv in zip(vals, cols, strict=True):
            a = rn(av)
            b = rn(a * rv)
            v = rn(b * cv)
            different_order |= struct.pack("<f", v) != struct.pack("<f", rn(a * rn(rv * cv)))
            for out, vv in zip(ref, (a, b, v), strict=True):
                out.extend(struct.pack("<f", vv))
    assert different_order
    for t, expected in zip((f, first, y), ref, strict=True):
        assert bytes(bits(t).cpu().tolist()) == bytes(expected)
