from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
import struct
from projection_witness import checked_index, counts, explicit, load_explicit, pack, sha


class Compact(TestCase):
    def test_equal_commitment_needs_no_disclosed_bytes(self):
        h = sha(b"not redistributed")
        self.assertEqual(counts(h, h, 100, {}), [0, 0, 0])

    def test_distinct_commitments_require_and_compare_each_word(self):
        a = struct.pack("<6H", 0, 0, 5, 9, 0, 1)
        b = struct.pack("<6H", 0x8000, 0, 5, 9, 1, 0)
        k, second_hash = sha(a), sha(b)
        self.assertEqual(counts(k, second_hash, 2, {k: a, second_hash: b}), [1, 0, 2])
        with self.assertRaises(KeyError):
            counts(k, second_hash, 2, {k: a})

    def test_reject_extra_missing_or_altered_compact_bytes(self):
        a, b = b"aa", b"bb"
        d = {sha(a): 2, sha(b): 2}
        with TemporaryDirectory() as x:
            p = Path(x) / "w"
            p.write_bytes(a + b)
            self.assertEqual(load_explicit(p, d), {sha(a): a, sha(b): b})
            for bad in [a + b + b"!", a, b + a, a + b"a!"]:
                p.write_bytes(bad)
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    load_explicit(p, d)

    def test_never_disclose_unapproved_stages(self):
        with self.assertRaisesRegex(ValueError, "disclose"):
            explicit({((128, 1, 0), "embedding"): {"a", "b"}}, {})

    def test_wrong_index_and_wrong_private_source_cannot_repack(self):
        with TemporaryDirectory() as x:
            p = Path(x) / "index"
            p.write_bytes(b"{}")
            self.assertEqual(checked_index(p, sha(b"{}")), {})
            with self.assertRaisesRegex(ValueError, "index"):
                checked_index(p, sha(b"[]"))
            with self.assertRaisesRegex(ValueError, "source"):
                pack(p, {}, {}, Path(x) / "new")
            self.assertFalse((Path(x) / "new").exists())
