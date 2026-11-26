"""No CUDA or third-party modules required."""

from pathlib import Path
import sys, unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from row_objectives import CLOCKS, GENERIC, TABLE, decide, evaluate, sample_medians


def sample(a, b, hit=True):
    rot = {"windows": [], "median_us": {}}
    for k, v in ((GENERIC, a), (TABLE, b)):
        rot["median_us"][k] = {CLOCKS[0]: v, CLOCKS[1]: v}
    for _ in range(7):
        rot["windows"].append(
            {
                "order": [GENERIC, TABLE],
                "samples": {
                    k: {CLOCKS[0]: [v] * 16, CLOCKS[1]: [v] * 5}
                    for k, v in ((GENERIC, a), (TABLE, b))
                },
            }
        )
    return {
        "rotating": rot,
        "gates": {k: {"admitted": True} for k in (GENERIC, TABLE)},
        "variants": [{"selected": {} if hit else None}],
    }


class ObjectiveTests(unittest.TestCase):
    def test_holdout_does_not_choose_route(self):
        for clock in CLOCKS:
            x = evaluate(sample(20, 10), sample(14, 24), clock)
            self.assertEqual(x["selected"], TABLE)
            self.assertGreater(x["heldout"]["selected_over_generic_us"], 1.03)
            x = evaluate(sample(15, 15), sample(20, 5), clock)
            self.assertEqual(x["selected"], GENERIC)
            self.assertEqual(x["heldout"]["selected_over_generic_us"], 1)
            self.assertEqual(
                evaluate(sample(20, 12, False), sample(20, 12, False), clock)["selected"], GENERIC
            )

    def test_thresholds_independent(self):
        self.assertEqual(
            decide({GENERIC: [10] * 7, TABLE: [6, 6, 6, 6, 12, 12, 12]}, True), GENERIC
        )
        self.assertEqual(decide({GENERIC: [10] * 7, TABLE: [9.8] * 7}, True), GENERIC)
        self.assertEqual(decide({GENERIC: [10] * 7, TABLE: [6] * 5 + [12] * 2}, True), TABLE)

    def test_tampered_reports(self):
        for kind in ("median", "length", "nonfinite", "unqualified", "order"):
            c = sample(20, 10)
            w = c["rotating"]["windows"][0]
            key = CLOCKS[0]
            if kind == "median":
                c["rotating"]["median_us"][TABLE][key] = 5
            if kind == "length":
                w["samples"][TABLE][key].pop()
            if kind == "nonfinite":
                w["samples"][TABLE][key][0] = float("nan")
            if kind == "unqualified":
                c["gates"][TABLE]["admitted"] = False
            if kind == "order":
                w["order"].append(GENERIC)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                sample_medians(c, key)


if __name__ == "__main__":
    unittest.main()
