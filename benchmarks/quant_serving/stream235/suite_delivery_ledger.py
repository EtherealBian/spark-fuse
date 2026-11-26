from unittest import TestCase
from delivery_ledger import describe


def read(events, ids, text, *, n=None, stop="eos", wstop=None):
    return describe(
        dict(
            case="512:key",
            rep=0,
            events=events,
            end_ns=100,
            predicted=len(ids) if n is None else n,
            stop=stop,
        ),
        dict(ids=ids, content=text, stop=wstop or stop),
        [7],
    )


class Delivery(TestCase):
    def test_complete_unicode_and_whitespace_latency(self):
        v = read(
            [[8, 1, 2, "  "], [10, 2, 3, "λx"], [35, 3, 4, "\tπ"], [99, 4, 7, ""]],
            [2, 3, 4, 7],
            "  λx\tπ",
        )
        self.assertEqual(
            (
                v["first_id_ns"],
                v["first_visible_ns"],
                v["last_visible_ns"],
                v["tail_after_last_visible_ns"],
            ),
            (8, 10, 35, 65),
        )
        self.assertEqual(
            (v["coverage"], v["visible_nonspace_codepoints"], v["codepoint_arrival_sum_ns"]),
            ("fully-delivered", 3, 55),
        )

    def test_missing_eog_not_falsely_filled(self):
        v = read([[9, 1, 3, "AM"], [12, 2, 4, "BER"]], [3, 4, 7], "AMBER")
        self.assertEqual(v["missing_ordinals"], [3])
        self.assertEqual(v["coverage"], "only-final-eog-not-delivered")
        self.assertEqual(v["retrospective_exact_label_ns"], 12)

    def test_missing_in_middle_vs_cap(self):
        self.assertEqual(
            read([[5, 2, 3, "A"], [7, 3, 7, ""]], [2, 3, 7], "A")["coverage"],
            "interior-ordinal-missing",
        )
        self.assertEqual(
            read([[5, 1, 3, "A"]], [3, 4], "A", stop="limit")["coverage"], "other-missing-ordinal"
        )

    def test_disagreement_precedes_even_complete_counts(self):
        for ids, text in [([2, 7], "A"), ([3, 7], "B")]:
            self.assertEqual(
                read([[8, 1, 3, "A"], [9, 2, 7, ""]], ids, text)["coverage"], "witness-disagreement"
            )
        self.assertIsNone(
            read([[6, 1, 3, "AMBER"], [7, 2, 7, "!"]], [3, 7], "AMBER!")[
                "retrospective_exact_label_ns"
            ]
        )

    def test_empty_delivery_is_not_an_inferred_visible_answer(self):
        q = read([], [7], "")
        self.assertEqual(q["coverage"], "only-final-eog-not-delivered")
        self.assertIsNone(q["first_visible_ns"])
        self.assertIsNone(q["mean_visible_codepoint_arrival_ms"])
        self.assertIsNone(q["tail_after_last_visible_ns"])

    def test_invalid_chronology_and_ordinals_rejected(self):
        for es in [
            [[20, 1, 3, ""], [18, 2, 7, ""]],
            [[1, 2, 7, ""], [2, 2, 7, ""]],
            [[101, 1, 3, ""]],
        ]:
            with self.subTest(events=es), self.assertRaises(ValueError):
                read(es, [3, 7], "")
