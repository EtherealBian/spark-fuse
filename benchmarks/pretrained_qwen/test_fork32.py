from hashlib import sha256
from pathlib import Path
from unittest import TestCase
from audit_fork32 import ARMS, BYTES_PER_POSITION, STEPS, V, case, order, refs, workload


class Panel(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.o = refs(Path("reports"))

    def example(self):
        n, i = 128, 1
        targets, gold = workload(self.o, n, i)
        c = dict(
            length=n,
            family=i,
            prefix=96,
            snapshot_bytes=96 * BYTES_PER_POSITION,
            capacity=160,
            inputs=list(map(list, targets)),
            diagnostics=[],
            pairs=[],
        )
        for a in ARMS:
            for k in (0, 1):
                c["diagnostics"].append(
                    dict(
                        arm=a,
                        branch=k,
                        tokens=gold[:],
                        shape=[STEPS, V],
                        dtype="torch.bfloat16",
                        finite=True,
                        logits_sha256=sha256(str(k).encode()).hexdigest(),
                    )
                )
        for j in range(4):
            arms, bs = order(j)
            ws = []
            for a in arms:
                cold = a == "cold"
                ws.append(
                    dict(
                        arm=a,
                        prepare_ms=0 if cold else 2,
                        targets=[
                            dict(branch=k, wall_ms=10 if cold else 6, ids=gold[:]) for k in bs
                        ],
                        wave_ms=20.1 if cold else 14.1,
                    )
                )
            c["pairs"].append(dict(pair=j, waves=ws))
        return c

    def test_two_branch_cost_includes_preparation(self):
        c = self.example()
        r = case(c, self.o, 128, 1)
        self.assertTrue(r["passes_strict_gate"])
        self.assertEqual(r["median_snapshot_prepare_ms"], 2)
        for z in c["pairs"]:
            w = next(w for w in z["waves"] if w["arm"] == "snapshot")
            w["prepare_ms"] = 20
            w["wave_ms"] = 32.1
        r = case(c, self.o, 128, 1)
        self.assertFalse(r["passes_timing_only"])
        self.assertLess(
            r["median_prepared_target_ms"]["snapshot"], r["median_prepared_target_ms"]["cold"]
        )

    def test_exactness_not_assumed_even_if_ids_match(self):
        c = self.example()
        c["diagnostics"][-1]["logits_sha256"] = "e" * 64
        r = case(c, self.o, 128, 1)
        self.assertEqual(r["matching_token_trajectories"], 2)
        self.assertEqual(r["byte_identical_full_logit_trajectories"], 1)
        self.assertFalse(r["passes_strict_gate"])
        self.assertTrue(r["passes_timing_only"])
        c = self.example()
        c["pairs"][0]["waves"][0]["targets"][0]["ids"][0] = 3
        r = case(c, self.o, 128, 1)
        self.assertFalse(r["all_timed_rows_agree_with_own_diagnostic"])
        self.assertFalse(r["passes_strict_gate"])

    def test_invalid_evidence_is_rejected(self):
        for name in (
            "below-charged",
            "mixed-prefix",
            "repeat-arm",
            "missing",
            "nan",
            "missing-logit",
            "capacity",
            "edited",
        ):
            c = self.example()
            w = c["pairs"][0]["waves"][0]
            if name == "below-charged":
                w["wave_ms"] = 19
            elif name == "mixed-prefix":
                c["inputs"][1][24] += 1
            elif name == "repeat-arm":
                w["arm"] = "snapshot"
            elif name == "missing":
                c["pairs"].pop()
            elif name == "nan":
                w["targets"][0]["wall_ms"] = float("nan")
            elif name == "missing-logit":
                c["diagnostics"].pop()
            elif name == "capacity":
                c["snapshot_bytes"] += 1
            else:
                c["inputs"][1] = c["inputs"][0]
            with self.subTest(case=name), self.assertRaises(ValueError):
                case(c, self.o, 128, 1)
