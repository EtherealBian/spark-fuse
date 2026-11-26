"""GGUF directory accounting for the audited expert layouts; do not modify the 235B baseline type table."""

import hashlib
from math import prod
from gguf_budget import Header, LIMIT
from shard_headers import BLOCKS as BASE_BLOCKS, check

BLOCKS = {**BASE_BLOCKS, 8: ("q8_0", 32, 34), 13: ("q5_K", 256, 176), 39: ("mxfp4", 32, 17)}


def packed_size(shape, kind):
    check(kind in BLOCKS, "unsupported GGML block layout")
    _, block, size = BLOCKS[kind]
    check(shape[0] % block == 0, "quantized dimension not divisible by block size")
    return (shape[0] // block) * size * prod(shape[1:])


def part(data, size):
    check(type(size) is int and 24 <= len(data) <= min(size, LIMIT), "bounded directory required")
    r = Header(data)
    check(bytes(r.take(4)) == b"GGUF" and r.read("I") == 3, "expected GGUF v3")
    nt, nm = r.read("Q"), r.read("Q")
    check(0 < nt < 100000 and 0 < nm < 100000, "unreasonable record count")
    m = {}
    for _ in range(nm):
        key = r.string(True)
        check(key not in m, "duplicate key")
        m[key] = r.value(r.read("I"), key == "general.architecture")
    align = m.get("general.alignment", 32)
    check(type(align) is int and 1 <= align <= 4096 and not align & (align - 1), "bad alignment")
    entries = {}
    offsets = {}
    for _ in range(nt):
        name = r.string(True)
        nd = r.read("I")
        check(name and name not in entries and 0 < nd <= 4, "bad tensor identity")
        shape = tuple(r.read("Q") for _ in range(nd))
        kind = r.read("I")
        off = r.read("Q")
        check(
            all(0 < x <= 2**32 for x in shape) and off % align == 0 and off not in offsets,
            "bad tensor descriptor",
        )
        entries[name] = (shape, kind)
        offsets[off] = packed_size(shape, kind)
    start = (r.pos + align - 1) // align * align
    check(min(offsets) == 0 and max(offsets) + start < size, "tensor outside shard")
    end = 0
    for off, amount in sorted(offsets.items()):
        check(off >= end and start + off + amount <= size, "overlap or truncated tensor payload")
        end = off + amount
    return m, entries, start, hashlib.sha256(data[: r.pos]).hexdigest()
