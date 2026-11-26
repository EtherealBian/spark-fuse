"""Never rescue a partial answer or alter the prior whole-output contract."""

import json
from pathlib import Path
import unittest
from manifest_answer_boundary import boundary, score, inspect


class BoundaryTests(unittest.TestCase):
    def test_closed_current_body_never_counts(self):
        for s in ("<think>MAPLE</think>CEDAR", "<think>\nMAPLE\n</think> CEDAR "):
            self.assertEqual(score(boundary(s)[1], "MAPLE", "CEDAR"), "stale")

    def test_reject_unclosed_nested_and_repeated(self):
        for s in (
            "<think>MAPLE",
            "<think><think>x</think>MAPLE",
            "x</think>MAPLE",
            "a<think>b</think>MAPLE",
            "<think>a</think>MAPLE</think>",
        ):
            self.assertIsNone(boundary(s)[1])

    def test_plain_output_not_rewritten(self):
        self.assertEqual(boundary(" MAPLE. "), ("plain", "MAPLE."))
        self.assertEqual(score("MAPLE.", "MAPLE", "CEDAR"), "other")
        self.assertEqual(score("", "MAPLE", "CEDAR"), "other")

    def test_real_panel_retains_both_scorers(self):
        d = Path("reports/glm45-355b/manifest")
        r = inspect(
            *(
                json.loads((d / n).read_bytes())
                for n in ("rows.json", "runtime.json", "timing.json")
            )
        )
        self.assertEqual(
            [r["result"][m]["strict_whole"] for m in ("cold", "reuse")],
            [{"other": 8}, {"other": 8}],
        )
        self.assertEqual(
            [r["result"][m]["answer_field"] for m in ("cold", "reuse")],
            [
                {"current": 4, "other": 1, "unavailable": 3},
                {"current": 5, "other": 1, "unavailable": 2},
            ],
        )
        self.assertEqual(len(r["cases"]), 16)
