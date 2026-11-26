from copy import deepcopy
from unittest import TestCase
from bench_profiled import CASES, REPS, audit, oracle, order, fixture, one


def raw():
    rows = []
    for g, (n, family) in enumerate(CASES):
        for rep in range(REPS):
            seq = order(g, rep)
            q = oracle(n, family, rep)
            rows.append(
                dict(
                    n=n,
                    case=family,
                    rep=rep,
                    order=seq,
                    arms={m: dict(ns=5000000 if m == "scan" else 3500000, leases=q) for m in seq},
                )
            )
    return dict(schema=1, machine="aarch64", cpu_affinity_count=1, rows=rows)


class Paired(TestCase):
    def test_gates_require_all_bounds_and_two_independent_wins(self):
        r = raw()
        self.assertTrue(audit(r)["passes_opt_in_gate"])
        self.assertEqual(audit(r)["matched_pairs"], 80)
        for row in r["rows"]:
            if row["case"] == "empty" and row["order"][0] == "scan":
                row["arms"]["scoped16"]["ns"] = 7000000
        self.assertFalse(audit(r)["passes_opt_in_gate"])
        self.assertFalse(audit(r)["cases"]["empty"]["within_regression_bounds"])
        r = raw()
        for row in r["rows"]:
            if row["case"] != "empty":
                row["arms"]["scoped16"]["ns"] = 5000000
        self.assertFalse(audit(r)["passes_opt_in_gate"])
        for attr in ("order", "time", "lease"):
            q = deepcopy(r)
            a = q["rows"][0]
            if attr == "order":
                a["order"].reverse()
            elif attr == "time":
                a["arms"]["scan"]["ns"] = True
            else:
                a["arms"]["scan"]["leases"][0][1] = False
            with self.assertRaises(ValueError):
                audit(q)

    def test_actual_small_fallback_oracle(self):
        for rep in (0, 1, 5):
            seeds, queries = fixture(2, "fallback-control", rep)
            for m in ("scan", "scoped16"):
                self.assertEqual(
                    one(m, 2, seeds, queries)["leases"], oracle(2, "fallback-control", rep)
                )
