import struct
import unittest
from gguf_budget import inspect


def make(*, offset=0, shards=1):
    def u(n):
        return struct.pack("<I", n)

    def q(n):
        return struct.pack("<Q", n)

    def st(s):
        return q(len(s)) + s.encode()

    meta = [
        ("general.architecture", 8, st("toy")),
        ("toy.block_count", 4, u(4)),
        ("toy.attention.head_count_kv", 4, u(2)),
        ("toy.attention.key_length", 4, u(8)),
        ("toy.attention.value_length", 4, u(4)),
        ("split.count", 4, u(shards)),
        ("tokens", 9, u(8) + q(2) + st("a") + st("b")),
    ]
    b = (
        b"GGUF"
        + u(3)
        + q(1)
        + q(len(meta))
        + b"".join(st(k) + u(t) + v for k, t, v in meta)
        + st("weight")
        + u(2)
        + q(2)
        + q(3)
        + u(0)
        + q(offset)
    )
    return b


class Budget(unittest.TestCase):
    def test_header_only(self):
        b = make()
        v = inspect(b, 1024, slots=2, ctx=16)
        self.assertEqual(v["logical_gguf_elements"], 6)
        self.assertEqual(v["uniform_dense_fp16_kv_bytes"], 6144)
        self.assertEqual(v["tensor_data_start"] % 32, 0)

    def test_incomplete_and_unsupported(self):
        for b, n in [
            (make()[:-1], 1024),
            (make(offset=1), 1024),
            (make(offset=1024), 1024),
            (make(shards=2), 1024),
            (b"BAD!" + make()[4:], 1024),
            (make(), 16),
        ]:
            with self.assertRaises(ValueError):
                inspect(b, n)
