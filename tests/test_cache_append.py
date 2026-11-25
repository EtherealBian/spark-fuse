"""A cache is an observable value. Check all returned bits, aliases, inputs and layouts."""

from math import prod
import pytest
import torch
from sparkfuse.partition import partition_export


class Cats(torch.nn.Module):
    def __init__(self, mode="normal"):
        super().__init__()
        self.mode = mode

    def forward(self, k, kt, v, vt):
        k1 = torch.cat((k, kt), 2)
        if self.mode == "barrier":
            v = v + torch.sin(k1[:, :, : v.shape[2]])
        if self.mode == "strided":
            v = v[..., ::2]
            vt = vt[..., ::2]
        if self.mode == "head":
            v1 = torch.cat((v, v), 1)
        else:
            v1 = torch.cat((v, vt), -2)
        if self.mode == "chain":
            k2 = torch.cat((k1, kt), 2)
            v2 = torch.cat((v1, vt), 2)
            return k1, v1, k2, v2, k2[:, :, 1:]
        return k1, v1, k1[:, :, 1:], v1[:, :, :-1]


def inputs(dt, d, layout, offsets, old=7, t=3):
    # Arbitrary bit patterns include negative zeros, NaN payloads and infinities.
    it = torch.int32 if dt == torch.float32 else torch.int16
    g = torch.Generator(device="cuda").manual_seed(90331)
    out = []
    for i, (sw, off, n) in enumerate(zip(layout, offsets, (old, t, old, t), strict=True)):
        sh = (2, n, 3, d) if sw else (2, 3, n, d)
        flat = torch.randint(
            torch.iinfo(it).min,
            torch.iinfo(it).max,
            (prod(sh) + off,),
            device="cuda",
            dtype=it,
            generator=g,
        )
        u = flat[off:]
        u[::19] = -32768 if it == torch.int16 else -2147483648
        u[2::23] = (
            0x7FC1 if dt == torch.bfloat16 else (0x7E01 if dt == torch.float16 else 0x7FC00001)
        )
        v = u.view(dt).reshape(sh)
        out.append(v.transpose(1, 2) if sw else v)
    return tuple(out)


def check(got, ref):
    for x, y in zip(got, ref, strict=True):
        it = torch.int32 if x.dtype == torch.float32 else torch.int16
        assert (
            x.shape == y.shape
            and x.stride() == y.stride()
            and torch.equal(x.contiguous().view(it), y.contiguous().view(it))
        )


def addr(t):
    return t.untyped_storage().data_ptr()


@pytest.mark.parametrize("host", ["python", "native"])
@pytest.mark.parametrize("dt,d", [(torch.float32, 9), (torch.float16, 14), (torch.bfloat16, 32)])
@pytest.mark.parametrize(
    "layout", [(False, False, False, False), (True, False, True, False), (False, True, True, True)]
)
@torch.inference_mode()
def test_append_bitpatterns(host, dt, d, layout):
    a = inputs(dt, d, layout, [0] * 4)
    model = Cats()
    gm, rec = partition_export(
        torch.export.export(model, a), paired_appends=True, host_backend=host
    )
    (r,) = [r for r in rec if r.get("kind") == "paired_append"]
    assert r["logical_shape"] == [2, 3, 7, 3, d] and len(r["ops"]) == 2
    assert r["physical_order"] == ["BSHD" if x else "BHSD" for x in layout]
    for offs in [(0, 0, 0, 0), (1, 2, 3, 5), (2, 2, 2, 2), (4, 4, 4, 4)]:
        a = inputs(dt, d, layout, offs)
        before = [x.clone() for x in a]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            got = gm(*a)
            ref = model(*a)
        torch.cuda.current_stream().wait_stream(stream)
        check(got, ref)
        check(a, before)
        assert (
            addr(got[0]) != addr(got[1])
            and {addr(y) for y in got}.isdisjoint(map(addr, a))
            and addr(got[2]) == addr(got[0])
            and addr(got[3]) == addr(got[1])
        )
        check(got, ref)


@pytest.mark.parametrize("mode", ["barrier", "strided", "head"])
@torch.inference_mode()
def test_append_rejects(mode):
    a = inputs(torch.float16, 14, (False,) * 4, (0,) * 4)
    m = Cats(mode)
    g, r = partition_export(torch.export.export(m, a), paired_appends=True)
    assert not any(x.get("kind") == "paired_append" for x in r)


@torch.inference_mode()
def test_append_carry_chain_and_alias_input():
    k, kt, v, vt = inputs(torch.bfloat16, 32, (True, False, True, False), (1, 2, 3, 5))
    a = (k, kt, v, kt)
    m = Cats("chain")
    ep = torch.export.export(m, (k, kt, v, vt))
    gm, rs = partition_export(ep, paired_appends=True)
    assert sum(r.get("kind") == "paired_append" for r in rs) == 2
    out = gm(*a)
    reference = m(*a)
    check(out, reference)
    assert len({addr(z) for z in out[:4]}) == 4
    for i in range(len(reference)):
        for j in range(len(reference)):
            assert (addr(out[i]) == addr(out[j])) == (addr(reference[i]) == addr(reference[j]))
    assert addr(out[-1]) == addr(out[2])  # the last returned view slices K2, not V2
    _, base = partition_export(ep)
    assert not any(r.get("kind") == "paired_append" for r in base)


@pytest.mark.parametrize("host", ["python", "native"])
def test_append_runtime_guards(host):
    a = inputs(torch.float32, 9, (False,) * 4, (0,) * 4)
    gm, r = partition_export(torch.export.export(Cats(), a), paired_appends=True, host_backend=host)
    p = gm.get_submodule(next(x["module"] for x in r if x.get("kind") == "paired_append"))
    for bad in [
        a[0].transpose(0, 1),
        a[0][..., :-1],
        a[0].half(),
        a[0].cpu(),
        a[0].clone().requires_grad_(True),
        torch.empty_strided(a[0].shape, (400, 100, 10, 1), device="cuda"),
    ]:
        with pytest.raises((RuntimeError, ValueError)):
            p(bad, *a[1:])
