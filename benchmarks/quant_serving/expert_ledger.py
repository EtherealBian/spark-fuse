"""Account for expert tensors in three named GGUF layouts without reading full payloads."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from gguf_budget import LIMIT
from expert_directory import BLOCKS, packed_size, part

BATCHES = (1, 2, 8, 32)
EXPERT = re.compile(r"blk\.(0|[1-9]\d*)\.ffn_(gate|up|down)_exps\.(weight|bias)")
LAYER = re.compile(r"blk\.(0|[1-9]\d*)\.(.+)")
SPLIT = re.compile(r"(.+)-(\d{5})-of-(\d{5})\.gguf")
SCHEMA = "sparkfuse.gguf.expert-ledger.v1"


def need(test, reason):
    if not test:
        raise ValueError(reason)


def attribute(meta, entries, batches=BATCHES):
    arch = meta["general.architecture"]
    need(
        arch in ("qwen2", "gpt-oss", "qwen3moe"), "only audited architecture layouts are supported"
    )
    layers = meta[arch + ".block_count"]
    n, k = (meta.get(arch + "." + a) for a in ("expert_count", "expert_used_count"))
    need(type(layers) is int and 0 < layers <= 1024, "invalid layer count")
    need(
        all(type(v) is int and 0 < v <= 65536 for v in batches)
        and list(batches) == sorted(set(batches))
        and batches,
        "invalid simultaneous token counts",
    )
    moe = arch != "qwen2"
    need(
        (moe and type(n) is int and type(k) is int and 0 < k <= n <= 65536)
        or (not moe and n is None and k is None),
        "unexpected expert geometry",
    )
    types = Counter()
    categories = Counter()
    witnesses = defaultdict(set)
    seen = set()
    suffixes = {"bias", "weight"} if arch == "gpt-oss" else {"weight"}
    expected = {(p, s) for p in ("gate", "up", "down") for s in suffixes}
    for name, (shape, kind) in entries.items():
        match = LAYER.fullmatch(name)
        if match:
            layer = int(match[1])
            need(layer < layers, "layer outside configured model")
            seen.add(layer)
        else:
            need(not name.startswith("blk."), "malformed block name")
        value = packed_size(shape, kind)
        types[BLOCKS[kind][0]] += value
        e = EXPERT.fullmatch(name)
        if e:
            li, proj, suffix = int(e[1]), e[2], e[3]
            need(
                moe
                and (proj, suffix) in expected
                and len(shape) == (3 if suffix == "weight" else 2)
                and shape[-1] == n
                and value % n == 0,
                "invalid expert tensor layout",
            )
            witnesses[li].add((proj, suffix))
            group = "routed_expert"
        else:
            need("exps" not in name and "expert" not in name, "unrecognized expert tensor")
            if match and match[2] in ("ffn_gate_inp.weight", "ffn_gate_inp.bias"):
                need(
                    moe and shape[-1] == n and len(shape) == (2 if name.endswith(".weight") else 1),
                    "router shape",
                )
                group = "router"
                witnesses[layer].add(("router", name.rsplit(".", 1)[1]))
            elif match and match[2].startswith("attn_"):
                group = "attention"
            elif match and match[2] in ("ffn_up.weight", "ffn_gate.weight", "ffn_down.weight"):
                group = "dense_ffn"
            elif name in ("token_embd.weight", "output.weight"):
                group = "embedding_or_output"
            else:
                group = "other"
        categories[group] += value
    need(seen == set(range(layers)), "missing transformer layers")
    if moe:
        per_layer = expected | {("router", s) for s in suffixes}
        need(
            all(witnesses[i] == per_layer for i in range(layers)),
            "incomplete expert or router projection set",
        )
    expert = categories["routed_expert"]
    per = expert // n if moe else 0
    bounds = (
        {str(b): {"min_bytes": per * k, "max_bytes": per * min(n, b * k)} for b in batches}
        if moe
        else {}
    )
    need(sum(categories.values()) == sum(types.values()), "packed byte reconciliation failure")
    return dict(
        architecture=arch,
        layers=layers,
        experts_per_layer=n,
        selected_per_token_per_layer=k,
        packed_tensor_bytes=sum(types.values()),
        packed_bytes_by_group=dict(sorted(categories.items())),
        packed_bytes_by_type=dict(sorted(types.items())),
        routed_expert_union_bounds=bounds,
    )


def inspect(lock, root, batches=BATCHES):
    files = lock["files"]
    need(
        type(files) is list
        and 1 <= len(files) <= 100
        and re.fullmatch("[0-9a-f]{40}", lock["revision"]),
        "invalid model lock",
    )
    all_entries = {}
    inventory = []
    names = set()
    m0 = None
    family = None
    limit = None
    for i, f in enumerate(files):
        name = f["name"]
        size = f["bytes"]
        dig = f["sha256"]
        need(
            isinstance(name, str)
            and re.fullmatch("[A-Za-z0-9][A-Za-z0-9_.-]*\\.gguf", name)
            and name not in names
            and type(size) is int
            and size > 24
            and re.fullmatch("[0-9a-f]{64}", dig),
            "invalid file record",
        )
        names.add(name)
        p = root / name
        need(
            not p.is_symlink() and p.is_file() and p.stat().st_size == size,
            "missing/incorrect-size local shard",
        )
        with p.open("rb") as fp:
            raw = fp.read(min(LIMIT, size))
        m, entries, align, header_hash = part(raw, size)
        split = SPLIT.fullmatch(name)
        if len(files) > 1:
            need(
                split
                and (i == 0 or split[1] == family)
                and int(split[2]) == i + 1
                and int(split[3]) == len(files),
                "invalid shard family or ordinal",
            )
            family = split[1]
            need(
                all(
                    type(m.get(s)) is int
                    for s in ("split.no", "split.count", "split.tensors.count")
                )
                and m["split.no"] == i
                and m["split.count"] == len(files)
                and m["split.tensors.count"] > 0,
                "invalid split metadata",
            )
        else:
            need(
                not split and m.get("split.count", 1) == 1 and m.get("split.no", 0) == 0,
                "expected one complete object",
            )
        if i == 0:
            m0 = m
            limit = m.get("split.tensors.count", len(entries))
        else:
            arch = m0["general.architecture"]
            need(
                m.get("split.tensors.count") == limit
                and all(
                    m[q] == m0.get(q)
                    for q in (
                        "general.architecture",
                        arch + ".block_count",
                        arch + ".expert_count",
                        arch + ".expert_used_count",
                    )
                    if q in m
                ),
                "shards disagree about model geometry",
            )
        need(all_entries.keys().isdisjoint(entries.keys()), "tensor occurs in two shards")
        all_entries.update(entries)
        packed = sum(packed_size(shape, kind) for shape, kind in entries.values())
        need(align + packed <= size, "impossible packed size")
        inventory.append(
            dict(
                name=name,
                object_bytes=size,
                aligned_directory_bytes=align,
                padding_or_trailer_bytes=size - align - packed,
                directory_sha256=header_hash,
                declared_object_sha256=dig,
                tensors=len(entries),
            )
        )
    need(type(limit) is int and len(all_entries) == limit, "missing tensor entries")
    result = attribute(m0, all_entries, batches)
    obj = sum(f["bytes"] for f in files)
    need(
        result["packed_tensor_bytes"]
        + sum(s["aligned_directory_bytes"] + s["padding_or_trailer_bytes"] for s in inventory)
        == obj,
        "bytes do not reconcile",
    )
    return dict(
        schema=SCHEMA,
        model=lock["model"],
        revision=lock["revision"],
        format=lock["format"],
        source_object_bytes=obj,
        tensor_count=len(all_entries),
        shards=inventory,
        **result,
        scope="Directory accounting only: an expert union is packed weight/bias slices selected by a group of simultaneous tokens, assuming arbitrary top-k overlap at every layer. Not whole-file verification, measured RAM/VRAM, DRAM traffic, a router trace, or a latency/throughput prediction.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("lock", type=Path)
    p.add_argument("directory", type=Path)
    p.add_argument("--simultaneous-tokens", type=int, nargs="+", default=BATCHES)
    a = p.parse_args()
    print(
        json.dumps(
            inspect(json.loads(a.lock.read_bytes()), a.directory, a.simultaneous_tokens),
            sort_keys=True,
            indent=2,
        )
    )
