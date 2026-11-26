"""Plan a single-file GGUF run from a bounded header. Never claims that weights were loaded."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct

LIMIT = 32 * 1024**2
TYPES = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class Header:
    def __init__(self, data):
        self.data = memoryview(data)
        self.pos = 0

    def take(self, n):
        if n < 0 or self.pos + n > len(self.data):
            raise ValueError("incomplete GGUF directory")
        start = self.pos
        self.pos += n
        return self.data[start : self.pos]

    def read(self, fmt):
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))[0]

    def string(self, keep):
        data = self.take(self.read("Q"))
        return bytes(data).decode("utf-8") if keep else None

    def value(self, kind, keep):
        if kind in TYPES:
            return self.read(TYPES[kind])
        if kind == 8:
            return self.string(keep)
        if kind != 9:
            raise ValueError("unsupported metadata value")
        inner = self.read("I")
        n = self.read("Q")
        if inner in TYPES:
            self.take(n * struct.calcsize("<" + TYPES[inner]))
        elif inner == 8 and n <= 1_000_000:
            for _ in range(n):
                self.string(False)
        else:
            raise ValueError("unsupported or excessive metadata array")
        return None


def inspect(data, object_bytes, *, slots=1, ctx=4096):
    if (
        not 24 <= len(data) <= LIMIT
        or type(object_bytes) is not int
        or object_bytes < len(data)
        or any(type(v) is not int or v <= 0 for v in (slots, ctx))
    ):
        raise ValueError("invalid file size or context geometry")
    r = Header(data)
    if bytes(r.take(4)) != b"GGUF" or r.read("I") != 3:
        raise ValueError("requires little-endian GGUF v3")
    tensors = r.read("Q")
    nmeta = r.read("Q")
    meta = {}
    if not 0 < tensors <= 100000 or not 0 < nmeta <= 100000:
        raise ValueError("invalid directory size")
    for _ in range(nmeta):
        key = r.string(True)
        if key in meta:
            raise ValueError("duplicate metadata key")
        meta[key] = r.value(r.read("I"), key == "general.architecture")
    arch = meta.get("general.architecture")
    align = meta.get("general.alignment", 32)
    if (
        not isinstance(arch, str)
        or not arch
        or type(align) is not int
        or not (1 <= align <= 4096)
        or align & (align - 1)
        or meta.get("split.count", 1) != 1
    ):
        raise ValueError("unsupported architecture/alignment or sharded file")
    dims = {}
    offsets = set()
    kinds = Counter()
    numel = 0
    for _ in range(tensors):
        name = r.string(True)
        nd = r.read("I")
        if name in dims or not (1 <= nd <= 4):
            raise ValueError("invalid tensor name or rank")
        shape = tuple(r.read("Q") for _ in range(nd))
        typ = r.read("I")
        off = r.read("Q")
        if not all(0 < s <= 2**32 for s in shape) or off % align or off in offsets:
            raise ValueError("invalid tensor header")
        dims[name] = shape
        offsets.add(off)
        kinds[typ] += 1
        numel += math.prod(shape)
    start = ((r.pos + align - 1) // align) * align
    if min(offsets) != 0 or start + max(offsets) >= object_bytes:
        raise ValueError("tensor offset beyond object")

    def get(k):
        return meta.get(arch + "." + k)

    layers = get("block_count")
    kv = get("attention.head_count_kv")
    key = get("attention.key_length")
    val = get("attention.value_length")
    if type(kv) is int and kv > 0:

        def width(k, side):
            shape = dims.get("blk.0.attn_" + side + ".weight", ())
            return (
                k
                if k is not None
                else (shape[1] // kv if len(shape) == 2 and shape[1] % kv == 0 else None)
            )

        key = width(key, "k")
        val = width(val, "v")
    geometry = all(type(x) is int and x > 0 for x in (layers, kv, key, val))
    estimate = slots * ctx * layers * kv * (key + val) * 2 if geometry else None
    return dict(
        architecture=arch,
        source_object_bytes=object_bytes,
        tensor_data_start=start,
        payload_and_padding_bytes=object_bytes - start,
        tensors=tensors,
        logical_gguf_elements=numel,
        tensor_types=dict(sorted(kinds.items())),
        directory_sha256=hashlib.sha256(r.data[: r.pos]).hexdigest(),
        layers=layers,
        kv_heads=kv,
        key_width=key,
        value_width=val,
        total_experts=get("expert_count"),
        active_experts=get("expert_used_count"),
        slots=slots,
        context_per_slot=ctx,
        uniform_dense_fp16_kv_bytes=estimate,
        scope="Header-only estimate. No full-file checksum, GPU run, allocator overhead, quality or throughput measurement. KV assumes full context at every layer; sliding windows and backend padding can change it.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("header", type=Path)
    p.add_argument("--object-bytes", required=True, type=int)
    p.add_argument("--slots", type=int, default=1)
    p.add_argument("--ctx", type=int, default=4096)
    a = p.parse_args()
    with a.header.open("rb") as f:
        data = f.read(LIMIT)
    print(json.dumps(inspect(data, a.object_bytes, slots=a.slots, ctx=a.ctx), indent=2))
