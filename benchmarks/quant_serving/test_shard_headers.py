import struct
import unittest
from shard_headers import inspect


def u(n):
    return struct.pack("<I", n)


def q(n):
    return struct.pack("<Q", n)


def string(s):
    return q(len(s.encode())) + s.encode()


def fake(i, *, name=None, total=2, offset=0, arch="toy"):
    vals = [("split.no", 4, u(i)), ("split.count", 4, u(2)), ("split.tensors.count", 4, u(total))]
    if i == 0 or arch != "toy":
        vals.append(("general.architecture", 8, string(arch)))
    if i == 0:
        vals.extend(
            [
                ("toy.block_count", 4, u(2)),
                ("toy.attention.head_count_kv", 4, u(1)),
                ("toy.attention.key_length", 4, u(32)),
                ("toy.attention.value_length", 4, u(16)),
            ]
        )
    return (
        b"GGUF"
        + u(3)
        + q(1)
        + q(len(vals))
        + b"".join(string(k) + u(t) + v for k, t, v in vals)
        + string(name or "tensor" + str(i))
        + u(2)
        + q(4)
        + q(6)
        + u(0)
        + q(offset)
    )


class ShardHeaders(unittest.TestCase):
    def setUp(self):
        self.lock = dict(
            model="toy",
            revision="a" * 40,
            format="mock",
            files=[
                dict(name=f"toy-{i:05d}-of-00002.gguf", bytes=2048, sha256="a" * 64) for i in (1, 2)
            ],
        )

    def test_reassembles_metadata_poor_secondary(self):
        b = inspect(self.lock, [fake(0), fake(1)], ctx=128, slots=2)
        self.assertEqual(
            (
                b["tensors"],
                b["logical_gguf_elements"],
                b["source_object_bytes"],
                b["uniform_dense_fp16_kv_bytes"],
            ),
            (2, 48, 4096, 49152),
        )

    def test_mutations(self):
        for raws in (
            [fake(0)],
            [fake(1), fake(0)],
            [fake(0), fake(1, name="tensor0")],
            [fake(0), fake(1, total=1)],
            [fake(0, total=3), fake(1, total=3)],
            [fake(0), fake(1, arch="other")],
            [fake(0), fake(1, offset=2048)],
            [fake(0), fake(1)[:-1]],
        ):
            with self.assertRaises(ValueError):
                inspect(self.lock, raws)
        self.lock["files"][1]["name"] = "mixed-00002-of-00002.gguf"
        with self.assertRaises(ValueError):
            inspect(self.lock, [fake(0), fake(1)])
