"""Tile boundaries, pointer alignment, byte identity, ownership and native guard coverage."""

import ctypes
import hashlib
import pytest
import torch
from sparkfuse.ir import Unsupported
from sparkfuse.resident import capture_export
from sparkfuse.runtime import build
from sparkfuse.slot_copy import generate


@pytest.mark.parametrize(
    "bad", [(), (0,), (-1,), (True,), (2.0,), (1,) * 33, (4096 * (2**31 - 1), 1)]
)
def test_slot_copy_reject_sizes(bad):
    with pytest.raises(Unsupported):
        generate(bad)


@pytest.mark.parametrize("offsets", [(0, 0), (0, 1), (3, 0), (1, 3)])
@torch.inference_mode()
def test_slot_copy_byte_tails_and_halos(offsets):
    sizes = (1, 2, 3, 15, 16, 17, 127, 128, 255, 4095, 4096, 4097, 8195, 9, 32, 33) * 2
    lib, _ = build(generate(sizes), torch.device("cuda"))
    a, b = offsets
    raw = [torch.randint(0, 256, (n + 64,), device="cuda", dtype=torch.uint8) for n in sizes]
    dst = [torch.full((n + 64,), 91, device="cuda", dtype=torch.uint8) for n in sizes]
    refs = [t.clone() for t in dst]
    for n, r, x in zip(sizes, refs, raw, strict=True):
        r[b : b + n].copy_(x[a : a + n])
    ptr = (ctypes.c_void_p * (2 * len(sizes)))(
        *[t.data_ptr() + b for t in dst], *[t.data_ptr() + a for t in raw]
    )
    stream = torch.cuda.current_stream()
    assert lib.launch(ptr, len(sizes) - 1, stream.cuda_stream) != 0
    assert lib.launch(ptr, len(sizes), stream.cuda_stream) == 0
    assert all(torch.equal(r, t) for r, t in zip(refs, dst, strict=True))


class Out(torch.nn.Module):
    def __init__(self, count):
        super().__init__()
        self.count = count

    def forward(self, x):
        if self.count == 1:
            return x + 1
        return tuple((x + i).to(torch.bfloat16) if i % 2 else x + i for i in range(self.count))


def leaves(tree):
    return tree if type(tree) is tuple else (tree,)


def same(a, b):
    return type(a) is type(b) and all(
        x.dtype == y.dtype
        and x.shape == y.shape
        and x.stride() == y.stride()
        and torch.equal(x.reshape(-1).view(torch.uint8), y.reshape(-1).view(torch.uint8))
        for x, y in zip(leaves(a), leaves(b), strict=True)
    )


@pytest.mark.parametrize("count", [1, 3, 21, 32])
@pytest.mark.parametrize("side", [False, True])
@torch.inference_mode()
def test_resident_fused_slot_ring_and_aliases(count, side):
    owner = torch.cuda.Stream() if side else torch.cuda.default_stream()
    with torch.cuda.stream(owner):
        req = [(torch.randn(3, 341, device="cuda"),) for _ in range(6)]
        m = Out(count).eval()
        ep = torch.export.export(m, req[0])
        refs = [m(*r) for r in req[2:]]
        with capture_export(
            ep, *req[:2], backend="aten", request_backend="native", slot_copy_backend="cuda"
        ) as f:
            sizes = [y.nbytes for y in leaves(refs[0])]
            meta = f.metadata["slot_copy"]
            assert (
                meta["cuda_sha256"] == hashlib.sha256(generate(sizes).encode()).hexdigest()
                and meta["output_bytes"] == sizes
            )
            with torch.inference_mode(False):
                # Distinct DLPack storages with shifted pointers and real sentinel bytes on each side.
                rings = [
                    [
                        torch.full((t.numel() + 2,), -83, dtype=t.dtype, device="cuda")
                        for t in leaves(y)
                    ]
                    for y in refs
                ]
                flat = [
                    tuple(
                        torch.from_dlpack(buf[1:-1]).view_as(t)
                        for buf, t in zip(row, leaves(y), strict=True)
                    )
                    for row, y in zip(rings, refs, strict=True)
                ]
                slots = [v if count > 1 else v[0] for v in flat]
            versions = [t._version for v in flat for t in v]
            old = f(*req[2])
            got = [f.run_into(r, slot) for r, slot in zip(req[2:], slots, strict=True)]
            assert all(x is y for x, y in zip(got, slots, strict=True)) and all(
                same(x, y) for x, y in zip(got, refs, strict=True)
            )
            assert [v + 1 for v in versions] == [t._version for v in flat for t in v]
            assert all(
                buf[0].item() == -83 and buf[-1].item() == -83 for row in rings for buf in row
            )
            snapshot = f._static[0].clone()
            keep = leaves(slots[0])[0].clone()
            bad = (*flat[0][:-1], f._outputs[-1]) if count > 1 else f._outputs[-1]
            with pytest.raises(Unsupported):
                f.run_into(req[2], bad)
            assert (
                torch.equal(snapshot, f._static[0])
                and torch.equal(keep, flat[0][0])
                and not f._failed
            )
            other = torch.cuda.Stream()
            other.wait_stream(owner)
            with torch.cuda.stream(other):
                with pytest.raises(RuntimeError, match="stream"):
                    f.run_into(req[2], slots[0])
                check = tuple(t.clone() for t in leaves(slots[0]))
            owner.wait_stream(other)
            assert same(check, leaves(refs[0]))
            assert (
                f.run_into(req[-1], slots[0]) is slots[0]
                and same(slots[0], refs[-1])
                and same(old, refs[0])
            )
        assert same(slots[-1], refs[-1])
        with pytest.raises(RuntimeError, match="closed"):
            f.run_into(req[-1], slots[0])


@torch.inference_mode()
def test_slot_backend_limits_and_explicit_optin():
    a, b = [(torch.randn(37, device="cuda"),) for _ in range(2)]
    m = Out(33)
    ep = torch.export.export(m, a)
    with pytest.raises(Unsupported, match="1..32"):
        capture_export(ep, a, b, backend="aten", request_backend="native", slot_copy_backend="cuda")
    for req, copy in [("python", "cuda"), ("native", "guess")]:
        with pytest.raises(ValueError, match="slot"):
            capture_export(ep, a, b, backend="aten", request_backend=req, slot_copy_backend=copy)
    with capture_export(ep, a, b, backend="aten", request_backend="native") as f:
        assert "slot_copy" not in f.metadata and same(m(*a), f(*a))
