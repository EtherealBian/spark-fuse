import copy
from itertools import combinations, product
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from expert_ledger import attribute, inspect
from expert_directory import packed_size


def toy(layers=2, n=4, k=2):
    arch = "qwen3moe"
    m = {
        "general.architecture": arch,
        arch + ".block_count": layers,
        arch + ".expert_count": n,
        arch + ".expert_used_count": k,
    }
    es = {}
    for i in range(layers):
        for p in ("gate", "down", "up"):
            es[f"blk.{i}.ffn_{p}_exps.weight"] = ((256, 2, n), 10 + i % 2)
        es[f"blk.{i}.ffn_gate_inp.weight"] = ((8, n), 0)
    return m, es


def binary(meta, entries):
    def u(x):
        return struct.pack("<I", x)

    def q(x):
        return struct.pack("<Q", x)

    def text(x):
        b = x.encode()
        return q(len(b)) + b

    def pad(x):
        return (x + 31) // 32 * 32

    h = b"GGUF" + u(3) + q(len(entries)) + q(len(meta))
    payload = 0
    for k, v in meta.items():
        h += text(k) + (u(8) + text(v) if isinstance(v, str) else u(4) + u(v))
    for n, (s, t) in entries.items():
        payload = pad(payload)
        h += text(n) + u(len(s)) + b"".join(q(x) for x in s) + u(t) + q(payload)
        payload += packed_size(s, t)
    return h + bytes(pad(len(h)) - len(h)) + bytes(payload)


class ExpertLedger(unittest.TestCase):
    def test_shared_and_disjoint_independent_exhaustive_oracle(self):
        m, e = toy()
        r = attribute(m, e, (1, 2, 3))
        pools = list(combinations(range(4), 2))
        per_layer = [
            sum(
                packed_size(s, t) // 4
                for nm, (s, t) in e.items()
                if nm.startswith(f"blk.{i}.") and "_exps." in nm
            )
            for i in range(2)
        ]
        self.assertNotEqual(*per_layer)
        for b in (1, 2, 3):
            independent = [
                len(set.union(*(set(c) for c in combo))) for combo in product(pools, repeat=b)
            ]
            self.assertEqual(
                r["routed_expert_union_bounds"][str(b)],
                dict(
                    min_bytes=sum(per_layer) * min(independent),
                    max_bytes=sum(per_layer) * max(independent),
                ),
            )
        self.assertGreater(r["packed_bytes_by_group"]["router"], 0)
        self.assertGreater(r["packed_tensor_bytes"], r["packed_bytes_by_group"]["routed_expert"])

    def test_refuses_unknown_geometry_or_partial_projections(self):
        m, e = toy()
        n = "blk.0.ffn_down_exps.weight"
        for update in (
            "missing",
            "bad-shape",
            "unknown",
            "geometry",
            "arch",
            "bool",
            "extra-layer",
        ):
            a, b = copy.deepcopy(m), dict(e)
            if update == "missing":
                del b[n]
            if update == "bad-shape":
                b[n] = ((256, 2, 8), 10)
            if update == "unknown":
                b["blk.0.some_expert.weight"] = ((1,), 0)
            if update == "geometry":
                a["qwen3moe.expert_used_count"] = 5
            if update == "arch":
                a["general.architecture"] = "other"
            if update == "bool":
                a["qwen3moe.expert_count"] = True
            if update == "extra-layer":
                b["blk.02.attn_q.weight"] = ((1,), 0)
            with self.subTest(update=update), self.assertRaises(ValueError):
                attribute(a, b)
        for batches in [(), (1, 1), (2, 1), (True,), (0,)]:
            with self.assertRaises(ValueError):
                attribute(m, e, batches)

    def test_dense_never_assigns_expert_bytes(self):
        a = {"general.architecture": "qwen2", "qwen2.block_count": 1}
        e = {"blk.0.ffn_down.weight": ((256, 2), 13)}
        r = attribute(a, e)
        self.assertEqual(r["routed_expert_union_bounds"], {})
        self.assertEqual(r["packed_bytes_by_group"], {"dense_ffn": 352})
        e["blk.0.ffn_down_exps.weight"] = ((256, 2, 4), 13)
        with self.assertRaises(ValueError):
            attribute(a, e)

    def test_mxfp4_bias_is_part_of_expert_slice_not_router(self):
        m = {
            "general.architecture": "gpt-oss",
            "gpt-oss.block_count": 1,
            "gpt-oss.expert_count": 4,
            "gpt-oss.expert_used_count": 1,
        }
        e = {}
        for p in ("gate", "up", "down"):
            e[f"blk.0.ffn_{p}_exps.weight"] = ((32, 2, 4), 39)
            e[f"blk.0.ffn_{p}_exps.bias"] = ((2, 4), 0)
        e["blk.0.ffn_gate_inp.weight"] = ((32, 4), 8)
        e["blk.0.ffn_gate_inp.bias"] = ((4,), 0)
        r = attribute(m, e)
        self.assertEqual(r["routed_expert_union_bounds"]["1"]["max_bytes"], 3 * (2 * 17 + 2 * 4))
        self.assertEqual(r["packed_bytes_by_group"]["router"], 34 * 4 + 4 * 4)
        del e["blk.0.ffn_up_exps.bias"]
        with self.assertRaises(ValueError):
            attribute(m, e)

    def test_split_directory_and_declared_vs_verified(self):
        m, e = toy()
        lock = {"revision": "a" * 40, "format": "unit", "model": "unit", "files": []}
        with TemporaryDirectory() as td:
            root = Path(td)
            records = []
            for i in range(2):
                shard = dict(m, **{"split.no": i, "split.count": 2, "split.tensors.count": len(e)})
                group = {k: v for k, v in e.items() if k.startswith(f"blk.{i}.")}
                b = binary(shard, group)
                n = f"toy-{i + 1:05}-of-00002.gguf"
                (root / n).write_bytes(b)
                records.append((n, shard, group))
                lock["files"].append(dict(name=n, bytes=len(b), sha256="b" * 64))
            r = inspect(lock, root)
            self.assertEqual(r["tensor_count"], len(e))
            self.assertEqual(
                sum(
                    s["aligned_directory_bytes"] + s["padding_or_trailer_bytes"]
                    for s in r["shards"]
                )
                + r["packed_tensor_bytes"],
                r["source_object_bytes"],
            )
            self.assertNotIn("verified_weight_bytes", r)
            bad = copy.deepcopy(lock)
            bad["files"].reverse()
            with self.assertRaises(ValueError):
                inspect(bad, root)
            bad = copy.deepcopy(lock)
            bad["files"][0]["name"] = "../a.gguf"
            with self.assertRaises(ValueError):
                inspect(bad, root)
            n, shard, group = records[1]
            shard["qwen3moe.expert_count"] = 8
            (root / n).write_bytes(binary(shard, group))
            with self.assertRaises(ValueError):
                inspect(lock, root)
            (root / n).write_bytes(b"x")
            with self.assertRaises(ValueError):
                inspect(lock, root)
