import unittest
from completion_meter import Counts, native


class Meter(unittest.TestCase):
    def rec(self, response=None, **kw):
        opts = dict(wall_ns=10**9, n_predict=5, draft_max=3, vocab_size=10, eog=[9])
        opts.update(kw)
        r = {
            "tokens": [6, 9],
            "stop_type": "eos",
            "timings": dict(predicted_n=4, draft_n=5, draft_n_accepted=3),
        }
        return native(r | (response or {}), **opts)

    def test_actual_ids_not_predicted(self):
        v = self.rec().report()
        self.assertEqual(
            (
                v["delivered_tokens_including_eog"],
                v["delivered_non_eog_tokens"],
                v["backend_minus_delivered"],
                v["delivered_per_sum_request_second"],
            ),
            (2, 1, 2, 2),
        )

    def test_no_proposals_is_not_zero_acceptance(self):
        r = self.rec({"timings": {"predicted_n": 2}, "tokens_predicted": 2}, draft_max=0)
        self.assertIsNone(r.report()["proposal_acceptance"])
        self.assertEqual(r.predicted, r.delivered)

    def test_cannot_hide_or_invent_work(self):
        for t in (
            {"predicted_n": 3},
            {"predicted_n": 3, "draft_n": 5},
            {"predicted_n": 3, "draft_n": 0, "draft_n_accepted": 2},
            {"predicted_n": True},
            {"predicted_n": 8},
        ):
            with self.subTest(t=t), self.assertRaises(ValueError):
                self.rec({"timings": t})

    def test_unsupported_or_ambiguous_observations(self):
        for p in (
            {"tokens": []},
            {"tokens": [9, 9]},
            {"tokens": [True, 9]},
            {"tokens": [10, 9]},
            {"stop_type": "word"},
            {"stop_type": "limit"},
            {"tokens_predicted": 3},
        ):
            with self.subTest(p=p), self.assertRaises(ValueError):
                self.rec(p)

    def test_addition_is_weighted_and_associative(self):
        a = self.rec()
        b = self.rec(
            {"tokens": [3] * 5, "stop_type": "limit", "timings": {"predicted_n": 5}},
            wall_ns=3 * 10**9,
        )
        self.assertEqual((Counts() + a) + b, a + (b + Counts()))
        self.assertEqual((a + b).report()["delivered_per_sum_request_second"], 1.75)
        self.assertEqual((a + b).report()["proposal_acceptance"], 0.6)
        self.assertIsNone(Counts().report()["delivered_per_sum_request_second"])

    def test_invalid_envelope_and_total(self):
        for kw in (
            {"wall_ns": 0},
            {"draft_max": 0},
            {"draft_max": 1},
            {"eog": []},
            {"n_predict": 3},
        ):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.rec(**kw)
        with self.assertRaises(ValueError):
            Counts(0, 0, 0, 0, 3, 0, 0, 0)
