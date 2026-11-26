"""Synthetic receipts exercise only the auditor; they are never timing evidence."""

import copy
import hashlib
import unittest

from audit_replay import TAGS, audit
from prefix_replay import common, digest


def sample():
    rows = []
    for length in (256, 2048):
        for i, name in enumerate(TAGS):
            order = ["cold", "reuse"] if (i + int(length == 2048)) % 2 == 0 else ["reuse", "cold"]
            shared = length - 20
            target = [i + 1] * shared + [21] * (length - shared)
            primer = [i + 1] * shared + [22] * (length - 9 - shared)
            w = dict(
                ids=list(range(24)),
                stop="limit",
                predicted=24,
                prompt_ms=100.0,
                decode_ms=23.0,
                wall_ms=124.0,
            )
            rows.append(
                dict(
                    length=length,
                    case=name,
                    target_ids=target,
                    primer_ids=primer,
                    target=digest(target),
                    primer=digest(primer),
                    primer_length=length - 9,
                    shared=shared,
                    order=order,
                    cold=dict(w, processed=length, reused=0),
                    reuse=dict(
                        copy.deepcopy(w), processed=20, reused=shared, prompt_ms=10.0, wall_ms=34.0
                    ),
                )
            )
    return rows


class ReplayEvidence(unittest.TestCase):
    def test_exact(self):
        a = audit(sample())
        self.assertTrue(a["all_tokens_exact"])
        self.assertEqual(a["cases"]["2048"]["median_paired_wall_ms_saved"], 90.0)

    def test_numerical_difference_kept(self):
        rows = sample()
        rows[3]["reuse"]["ids"][0] += 1
        a = audit(rows)
        self.assertFalse(a["all_tokens_exact"])
        self.assertEqual(a["cases"]["256"]["exact_token_pairs"], 3)

    def test_refuses_bad_receipts(self):
        for key, val in [
            ("reused", 1),
            ("processed", 4),
            ("ids", [42]),
            ("predicted", 0),
            ("stop", "eos"),
            ("wall_ms", 1),
            ("prompt_ms", float("nan")),
        ]:
            with self.subTest(key=key):
                rows = sample()
                rows[0]["cold"][key] = val
                with self.assertRaises(ValueError):
                    audit(rows)
        rows = sample()
        rows[0]["reuse"].update(reused=0, processed=256)
        with self.assertRaises(ValueError):
            audit(rows)

    def test_balanced_order_and_coverage(self):
        for key, val in [
            ("case", "duplicate"),
            ("primer", "xxx"),
            ("order", ["reuse", "cold"]),
            ("primer_length", 1),
        ]:
            with self.subTest(key=key):
                rows = sample()
                rows[0][key] = val
                with self.assertRaises(ValueError):
                    audit(rows)
        rows = sample()
        rows[0] = rows[1]
        with self.assertRaises(ValueError):
            audit(rows)

    def test_prefix_and_framed_digest(self):
        self.assertEqual(common([13, 2, 9], [13, 8, 9]), 1)
        self.assertEqual(common([3], [3, 4]), 1)
        self.assertEqual(common([], []), 0)
        self.assertNotEqual(digest([1, 23]), digest([12, 3]))
        self.assertEqual(digest([1]), hashlib.sha256(b"[1]").hexdigest())
