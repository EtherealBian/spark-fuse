"""Synthetic fail-closed audit fixtures, never model results."""

from pathlib import Path
from hashlib import sha256
import json
import struct
import tempfile
import unittest
from copy import deepcopy
import boundary_review as rv
import boundary_bytes as bb
from prefix_pair_costs import analyze


class Boundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = Path("reports/qwen32b-fork/raw.json")
        orig = json.loads(cls.ref.read_bytes())
        cls.o = rv.refs(Path("reports"))
        cls.template = dict(
            orig, schema="sparkfuse.boundary32.v1", experiment_source_sha256=rv.sources(), cases=[]
        )
        old = {c["family"]: c for c in orig["cases"] if c["length"] == 2048}
        for p, i in rv.CELLS:
            cell = deepcopy(old[i])
            cell.update(prefix=p, snapshot_bytes=p * rv.BYTES_PER_POSITION)
            for d in cell["diagnostics"]:
                d["step_sha256"] = [
                    sha256(f"fixture:{d['logits_sha256']}:{s}".encode()).hexdigest()
                    for s in range(8)
                ]
            cls.template["cases"].append(cell)

    def setUp(self):
        self.r = deepcopy(self.template)
        self.c = self.r["cases"][0]

    def replay(self):
        return rv.summarize(self.r, self.o, self.ref)

    def test_balanced_ledger(self):
        s = self.replay()
        self.assertEqual(
            (s["timed_waves"], s["diagnostic_trajectories"], s["timed_targets"]), (96, 48, 192)
        )
        self.assertEqual(s, self.replay())
        self.assertEqual({(p, i) for p in rv.CUTS for i in (1, 2)}, set(rv.CELLS))
        self.assertEqual(
            [p for p, i in rv.CELLS if i == 1], list(reversed([p for p, i in rv.CELLS if i == 2]))
        )
        self.assertEqual(len(analyze(self.r)), 12)

    def test_invalid_evidence(self):
        def nonzero(c):
            c["pairs"][0]["waves"][0]["prepare_ms"] = 1

        def wrongcold(c):
            c["diagnostics"][0]["logits_sha256"] = "9" * 64

        def wrongstep(c):
            c["diagnostics"][2]["step_sha256"][4] = "a" * 64

        def input(c):
            c["inputs"][1][1941] += 1

        def prefix(c):
            c["prefix"] -= 1

        def size(c):
            c["snapshot_bytes"] += 4

        def order(c):
            c["pairs"][2]["waves"].reverse()

        def negative(c):
            c["pairs"][0]["waves"][0]["wave_ms"] = 1

        def early(c):
            d = c["diagnostics"][2]
            d["tokens"][0] += 1
            d["logits_sha256"] = "b" * 64
            d["step_sha256"][4] = "8" * 64

        def anchor(c):
            d = c["diagnostics"][2]
            d["logits_sha256"] = "1" * 64
            d["step_sha256"][4] = "8" * 64

        for f in (
            nonzero,
            wrongcold,
            wrongstep,
            input,
            prefix,
            size,
            order,
            negative,
            early,
            anchor,
        ):
            with self.subTest(f=f.__name__):
                self.r = deepcopy(self.template)
                c = next(
                    c for c in self.r["cases"] if c["prefix"] == (1536 if f == anchor else 512)
                )
                f(c)
                with self.assertRaises(ValueError):
                    self.replay()

    def test_changed_logits_need_not_change_greedy(self):
        d = self.c["diagnostics"][2]
        d["logits_sha256"] = "1" * 64
        d["step_sha256"][3] = "2" * 64
        s = self.replay()
        c = s["cases"][0]
        self.assertFalse(s["boundaries"]["512"]["both_families_strict"])
        self.assertEqual(c["matching_token_trajectories"], 2)
        self.assertEqual(c["branch_divergence"][0]["first_logit_step"], 4)
        self.assertIsNone(c["branch_divergence"][0]["first_greedy_id_step"])

    def test_timed_ids_and_paired_slowdown(self):
        self.c["pairs"][0]["waves"][0]["targets"][0]["ids"][3] += 1
        s = self.replay()
        self.assertFalse(s["cases"][0]["passes_strict_gate"])
        self.assertFalse(s["cases"][0]["all_timed_rows_agree_with_own_diagnostic"])
        self.c["pairs"][0]["waves"][1]["wave_ms"] *= 4
        s = self.replay()
        self.assertFalse(s["cases"][0]["passes_timing_only"])


class Bytes(unittest.TestCase):
    def test_actual_bytes_ties_signed_zero_history_and_tampering(self):
        def pack(xs):
            return b"".join(struct.pack("<H", v) for v in xs)

        xs = pack([0, 0x3F80, 0x3F80, 0x4000, 0x4040, 0, 0, 0, 0])
        ys = pack([0x8000, 0x3F80, 0x3F80, 0x4080, 0x4040, 0, 0, 0, 0])
        with tempfile.TemporaryDirectory() as dir:
            run = Path(dir)
            di = []
            for arm, data, ids in (("cold", xs, [1, 1, 0]), ("snapshot", ys, [1, 0, 0])):
                h, rows = bb.capture(run, data, 3, 3)
                for k in (0, 1):
                    di.append(
                        dict(
                            arm=arm,
                            branch=k,
                            shape=[3, 3],
                            tokens=ids,
                            step_sha256=rows,
                            logits_sha256=h,
                        )
                    )
            raw = dict(cases=[dict(prefix=2, family=1, diagnostics=di)])
            s = bb.inventory(run, raw)
            self.assertEqual((s["distinct_blobs"], s["validated_full_vocab_steps"]), (2, 12))
            t = s["traces"][0]["steps"]
            self.assertEqual(
                [x["same_input_and_generated_history"] for x in t], [True, True, False]
            )
            self.assertEqual([x["unequal_bf16_words"] for x in t], [1, 1, 0])
            self.assertEqual(t[0]["max_absolute_logit_delta"], 0)
            self.assertEqual(t[0]["cold_top_two_gap"], 0)
            self.assertEqual(t[1]["max_absolute_logit_delta"], 2)
            di[0]["tokens"] = [0, 0, 0]
            with self.assertRaisesRegex(ValueError, "argmax"):
                bb.inventory(run, raw)
            di[0]["tokens"] = [1, 1, 0]
            (run / "logits" / (h + ".bf16le")).write_bytes(xs)
            with self.assertRaisesRegex(ValueError, "digest"):
                bb.inventory(run, raw)
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                bb.describe(pack([0x7F80, 0]))
