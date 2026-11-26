import copy
import unittest
from pathlib import Path
from admission_trace import MODES, PIN, inputs, sources, identity, schedule, audit


class Trace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = Path("reports/qwen72b/assignment")
        cases = inputs(cls.ref)
        rows = []
        for i, h, order in schedule():
            arms = {}
            for name in MODES:
                hot = name == "fifo"
                lat = 100 if hot else 200
                arms[name] = dict(
                    leases=identity(cases[i]),
                    api_calls=1 if hot else 3,
                    router_calls=1 if hot else 3,
                    waits=1 if hot else 2,
                    worker_cpu_ns=50 if hot else 100,
                    release_cpu_ns=20,
                    arrival_to_grant_ns=h + lat,
                    actual_hold_ns=h,
                    release_to_grant_ns=lat,
                )
            rows.append(dict(case=i, hold_ns=h, order=list(order), arms=arms))
        cls.raw = dict(
            schema=1, source_sha256=sources(), history_sha256=PIN, affinity_cpus=1, rows=rows
        )

    def test_known_synthetic_arithmetic_only(self):
        o = audit(self.raw, self.ref)
        self.assertTrue(o["tested_gate"])
        self.assertEqual(o["router_calls"], {"poll-1ms": 144, "fifo": 48})
        self.assertEqual(o["matching_target_leases"], 192)

    def test_never_promote_if_one_order_tail_slips(self):
        x = copy.deepcopy(self.raw)
        for r in x["rows"]:
            if r["order"][0] == "fifo":
                a = r["arms"]["fifo"]
                a["release_to_grant_ns"] += 200
                a["arrival_to_grant_ns"] += 200
        o = audit(x, self.ref)
        self.assertFalse(o["tested_gate"])
        self.assertTrue(o["by_first_mode"]["poll-1ms"]["meets_test_gate"])

    def test_mismatched_observations_rejected(self):
        for mut in (
            lambda a: a["rows"].pop(),
            lambda a: a.update(affinity_cpus=2),
            lambda a: a["rows"][0]["order"].reverse(),
            lambda a: a["rows"][0]["arms"]["fifo"].update(actual_hold_ns=1),
            lambda a: a["rows"][0]["arms"]["poll-1ms"].update(router_calls=1),
            lambda a: a["rows"][0]["arms"]["fifo"]["leases"][0].__setitem__(2, 1),
            lambda a: a["rows"][0]["arms"]["fifo"].update(worker_cpu_ns=True),
            lambda a: a.update(history_sha256="0" * 64),
        ):
            x = copy.deepcopy(self.raw)
            mut(x)
            with self.assertRaises(ValueError):
                audit(x, self.ref)
