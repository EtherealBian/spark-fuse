from unittest import TestCase
from lookup_account import account
from lookup235 import assess
from test_lookup235 import fixture


class Account(TestCase):
    def data(self, bs):
        return dict(launches=4, target_requests=24, comparison=assess(bs, [150000]))

    def test_denominator_is_proposals_not_delivered(self):
        s = account(self.data(fixture()))
        x = s["families"]["copy"]["orders"][0]
        self.assertEqual((x["accepted"], x["drafted"], x["lookup_delivered"]), (2, 10, 6))
        self.assertEqual(x["proposal_acceptance"], 0.2)
        self.assertIsNone(s["families"]["new"]["orders"][1]["proposal_acceptance"])

    def test_overhang_or_one_order_slow_prevents_strict_recommendation(self):
        bs = fixture()
        z = next(x for x in bs[2]["rows"] if x["key"] == "edit:1")["response"]
        z["predicted"] += 1
        z["wall_ms"] = 39
        d = account(self.data(bs))["families"]["edit"]
        self.assertFalse(d["both_cases_pass"])
        self.assertEqual(d["preregistered_case_passes"], 1)
        self.assertEqual(d["orders"][1]["backend_minus_delivered"], 1)
        self.assertLess(d["orders"][1]["observed_wall_saving_fraction"], 0)
        self.assertEqual(d["orders"][1]["identical_outputs"], 2)

    def test_sums_not_unweighted_mean_of_per_target_saving(self):
        bs = fixture()
        next(x for x in bs[0]["rows"] if x["key"] == "copy:1")["response"]["wall_ms"] = 90
        d = account(self.data(bs))["families"]["copy"]["orders"][0]
        self.assertAlmostEqual(d["observed_wall_saving_fraction"], 1 - 14 / 100)
