import unittest
from shard_headers import packed_size, part
from test_shard_headers import q, u, string


def raw(spec):
    return (
        b"GGUF"
        + u(3)
        + q(len(spec))
        + q(1)
        + string("general.architecture")
        + u(8)
        + string("x")
        + b"".join(
            string(str(i)) + u(1) + q(n) + u(kind) + q(off) for i, (n, kind, off) in enumerate(spec)
        )
    )


class PackedRanges(unittest.TestCase):
    def test_exact_and_quantized_widths(self):
        spec = [(256, 10, 0), (256, 11, 96), (256, 12, 224), (256, 14, 384), (4, 0, 608)]
        h = raw(spec)
        start = (len(h) + 31) // 32 * 32
        self.assertEqual(part(h, start + 624)[2], start)
        self.assertEqual(packed_size((256, 2, 3), 10), 504)
        with self.assertRaises(ValueError):
            part(h, start + 623)

    def test_reject_overlap_and_unknown_packing(self):
        for spec in [[(256, 10, 0), (1, 0, 32)], [(128, 10, 0)], [(256, 999, 0)]]:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                part(raw(spec), 2048)
