from unittest import TestCase
from lookup235 import ARMS, CORPUS, GEN, ORDER, assess, check_row, digest, issue


def fixture():
    allb = []
    for i, m in enumerate(ARMS):
        model = m != "none"
        inputs = {}
        rows = []
        for k in ORDER if i < 2 else reversed(ORDER):
            text = " ".join(CORPUS[k])
            p = [1, list(ORDER).index(k) + 10, 2]
            inputs[k] = dict(rendered=text, ids=p, sha256=digest(p))
            can = model and not k.startswith("new:")
            r = dict(
                ids=[10, 11, 150000],
                text="reference",
                detokenized="reference",
                slot=0,
                truncated=False,
                stop="eos",
                evaluated=3,
                processed=3,
                cached=0,
                predicted=3,
                counters_present=can,
                drafted=5 if can else 0,
                accepted=1 if can else 0,
                prompt_ms=1,
                decode_ms=3,
                wall_ms=7 if can else 10,
            )
            rows.append(dict(key=k, response=r))
        allb.append(dict(block=i, mode=m, inputs=inputs, rows=rows))
    return allb


class Lookup(TestCase):
    def test_reverse_order_and_copy_vs_no_hit(self):
        r = assess(fixture(), [150000])
        self.assertTrue(r["workloads"]["copy"]["both_cases_pass"])
        self.assertFalse(r["workloads"]["new"]["both_cases_pass"])
        self.assertFalse(r["all_six_pass"])
        self.assertEqual(r["workloads"]["edit"]["cases_meeting_gate"], 2)

    def test_no_recommendation_for_drift_or_shorter_answer_or_wrong_order_effect(self):
        for case in ("drift", "shorter", "order", "backend_lookahead"):
            bs = fixture()
            r = bs[1]["rows"][0]["response"]
            if case == "drift":
                r["ids"][0] = 14
            elif case == "shorter":
                r["ids"] = [10, 150000]
                r["predicted"] = 2
            elif case == "backend_lookahead":
                r["predicted"] += 1
            else:
                r["wall_ms"] = 10.2
            with self.subTest(case=case):
                self.assertFalse(assess(bs, [150000])["cases"]["copy:0"]["passes"])

    def test_unsound_receipts_are_rejected(self):
        for typ in (
            "fake-cold",
            "offer",
            "counter",
            "arm",
            "missing",
            "different-input",
            "template",
            "nan",
            "text",
            "limit",
        ):
            bs = fixture()
            r = bs[1]["rows"][0]["response"]
            if typ == "fake-cold":
                r["cached"] = 1
            elif typ == "offer":
                r["accepted"] = 6
            elif typ == "counter":
                r["counters_present"] = False
            elif typ == "arm":
                bs[1]["mode"] = "none"
            elif typ == "missing":
                bs[1]["rows"].pop()
            elif typ == "different-input":
                p = bs[1]["inputs"]["copy:0"]
                p["ids"][0] = 100
                p["sha256"] = digest(p["ids"])
            elif typ == "template":
                bs[1]["inputs"]["copy:0"]["rendered"] = "foo"
            elif typ == "nan":
                r["wall_ms"] = float("nan")
            elif typ == "text":
                r["detokenized"] = "other"
            else:
                r["stop"] = "limit"
            with self.subTest(t=typ), self.assertRaises(ValueError):
                assess(bs, [150000])

    def test_natural_protocol_does_not_mask_or_inject_expected_output(self):
        seen = []

        def fn(k, d):
            seen.append((k, d))
            if k == "detokenize":
                return dict(content="text")
            return dict(
                tokens=[20, 150000],
                content="text",
                id_slot=0,
                truncated=False,
                stop_type="eos",
                tokens_evaluated=3,
                timings=dict(
                    prompt_n=3, predicted_n=2, cache_n=0, prompt_ms=0.001, predicted_ms=0.001
                ),
            )

        row = issue(fn, [1, 2, 3], [150000])
        self.assertEqual(
            seen[0][1],
            dict(
                prompt=[1, 2, 3],
                n_predict=GEN,
                cache_prompt=False,
                id_slot=0,
                temperature=0,
                seed=1513,
                stream=False,
                return_tokens=True,
            ),
        )
        self.assertEqual(seen[1], ("detokenize", {"tokens": [20]}))
        self.assertFalse(row["counters_present"])
        check_row(dict(ids=[1, 2, 3], sha256=digest([1, 2, 3])), row, [150000], "none")
