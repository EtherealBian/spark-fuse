"""A small deterministic fake server tests the trace generator/auditor contract only."""

import copy
import hashlib
import unittest
from unittest.mock import patch

import routed_trace
from audit_routes import audit
from slot_policy import common_prefix


class TraceChecks(unittest.TestCase):
    def fake(self):
        slots = {}

        def tokenize(_port, _path, body):
            return {"tokens": list(body["content"].encode())}

        def complete(_port, ids, reuse, n, *, slot):
            used = common_prefix(ids, slots.get(slot, ())) if reuse else 0
            slots[slot] = ids
            digest = hashlib.sha256(bytes(ids)).digest()
            work = len(ids) - used
            return dict(
                ids=list(digest[:n]),
                stop="limit",
                processed=work,
                reused=used,
                predicted=n,
                prompt_ms=work / 10,
                decode_ms=20.0,
                wall_ms=21.0 + work / 10,
            )

        with (
            patch.object(routed_trace, "post", tokenize),
            patch.object(routed_trace, "complete", complete),
        ):
            return routed_trace.run(0)

    def test_controls_and_altered_dispatch(self):
        rows, inputs = self.fake()
        s = audit(rows, inputs)
        self.assertTrue(s["all_tokens_exact"])
        a = s["patterns"]["alternating"]
        b = s["patterns"]["switchback"]
        self.assertEqual(a["round_robin"], a["affinity"])
        self.assertLess(b["affinity"]["processed"], b["round_robin"]["processed"])
        bad = copy.deepcopy(rows)
        bad[0]["rows"][0]["slot"] = 1
        with self.assertRaises(ValueError):
            audit(bad, inputs)
        bad = copy.deepcopy(rows)
        bad[-1]["rows"][-1]["ids"][0] ^= 1
        self.assertFalse(audit(bad, inputs)["all_tokens_exact"])
        bad = copy.deepcopy(inputs)
        bad["alternating:0"][0][0] ^= 1
        with self.assertRaises(ValueError):
            audit(rows, bad)
