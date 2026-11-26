"""Inspect bounded GGUF directories for all shards; never loads tensor data or claims a GPU run."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from math import prod
import re
from gguf_budget import Header, LIMIT


def check(ok, msg):
    if not ok:
        raise ValueError(msg)


# (name, logical values per quantization block, packed bytes per block).
# Verified against ggml_type_name, ggml_blck_size and ggml_type_size in the pinned build.
BLOCKS = {
    0: ("f32", 1, 4),
    10: ("q2_K", 256, 84),
    11: ("q3_K", 256, 110),
    12: ("q4_K", 256, 144),
    14: ("q6_K", 256, 210),
}


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


def inspect(lock, headers, ctx=4096, slots=1):
    f = lock["files"]
    check(
        type(ctx) is int
        and ctx > 0
        and type(slots) is int
        and slots > 0
        and len(f) == len(headers)
        and 1 < len(f) <= 100,
        "incomplete split or invalid context",
    )
    whole = {}
    kinds = Counter()
    info = []
    m0 = None
    total = None
    family = None
    for i, (rec, raw) in enumerate(zip(f, headers, strict=True)):
        path = rec["name"]
        p = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", path)
        check(
            p and int(p[2]) == i + 1 and int(p[3]) == len(f) and (i == 0 or p[1] == family),
            "invalid shard filename/order",
        )
        family = p[1]
        check(re.fullmatch(r"[0-9a-f]{64}", rec["sha256"]) is not None, "missing pinned digest")
        size = rec["bytes"]
        m, entries, start, sha = part(raw, size)
        check(
            type(m.get("split.no")) is int
            and m["split.no"] == i
            and m.get("split.count") == len(f),
            "split identity mismatch",
        )
        if i == 0:
            m0 = m
            total = m.get("split.tensors.count")
            check(
                type(total) is int and total > 0 and isinstance(m.get("general.architecture"), str),
                "primary split metadata missing",
            )
        else:
            check(
                m.get("split.tensors.count") == total
                and m.get("general.architecture", m0["general.architecture"])
                == m0["general.architecture"],
                "conflicting split metadata",
            )
        check(whole.keys().isdisjoint(entries.keys()), "tensor occurs in multiple shards")
        whole.update(entries)
        kinds.update(k for _, k in entries.values())
        info.append(
            dict(
                name=path,
                source_object_bytes=size,
                tensor_data_start=start,
                directory_sha256=sha,
                tensors=len(entries),
            )
        )
    check(len(whole) == total, "aggregate tensor count mismatch")
    arch = m0["general.architecture"]

    def get(key):
        return m0.get(arch + "." + key)

    layers = get("block_count")
    kv = get("attention.head_count_kv")
    key = get("attention.key_length")
    val = get("attention.value_length")
    experts = get("expert_count")
    active = get("expert_used_count")

    def width(value, side):
        if value is not None:
            return value
        dims = whole.get("blk.0.attn_" + side + ".weight", ((), None))[0]
        return (
            dims[1] // kv
            if len(dims) == 2 and type(kv) is int and kv > 0 and dims[1] % kv == 0
            else None
        )

    key = width(key, "k")
    val = width(val, "v")
    geo = all(type(v) is int and v > 0 for v in (layers, kv, key, val))
    budget = 2 * slots * ctx * layers * kv * (key + val) if geo else None
    by_kind = Counter()
    for shape, kind in whole.values():
        by_kind[BLOCKS[kind][0]] += packed_size(shape, kind)
    payload = sum(by_kind.values())
    object_bytes = sum(row["bytes"] for row in f)
    headers = sum(row["tensor_data_start"] for row in info)
    return dict(
        packed_tensor_bytes=payload,
        packed_tensor_bytes_by_type=dict(sorted(by_kind.items())),
        aligned_directory_bytes=headers,
        tensor_padding_or_trailer_bytes=object_bytes - headers - payload,
        architecture=arch,
        model=lock["model"],
        revision=lock["revision"],
        format=lock["format"],
        source_object_bytes=sum(f["bytes"] for f in lock["files"]),
        shards=info,
        tensors=len(whole),
        logical_gguf_elements=sum(prod(shape) for shape, _ in whole.values()),
        tensor_types=dict(sorted(kinds.items())),
        layers=layers,
        kv_heads=kv,
        key_width=key,
        value_width=val,
        total_experts=experts,
        active_experts=active,
        slots=slots,
        context_per_slot=ctx,
        uniform_dense_fp16_kv_bytes=budget,
        scope="Header-only: verifies directory consistency across shards, not tensor payloads, whole-object SHA, model quality, GPU loading, or achievable latency. KV assumes full-context FP16 at every layer.",
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("lock", type=Path)
    p.add_argument("headers", nargs="+", type=Path)
    a = p.parse_args()
    print(
        json.dumps(
            inspect(json.loads(a.lock.read_bytes()), [p.read_bytes() for p in a.headers]), indent=2
        )
    )
