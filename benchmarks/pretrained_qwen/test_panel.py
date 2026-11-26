from pathlib import Path
import json
import sys
import unittest
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import p95, summarize
from audit import audit


class Rules(unittest.TestCase):
    def test_tail(self):
        self.assertEqual(p95(list(range(1, 101))), 95)
        for xs in ([], [float("nan")], [0], [-1], [float("inf")]):
            with self.assertRaises(ValueError):
                p95(xs)

    def test_criterion(self):
        rows = [{"a": {"decode_ms": [8, 8]}, "b": {"decode_ms": [10, 10]}} for _ in range(12)]
        self.assertTrue(summarize(rows, "a", "b", True)["target_met"])
        self.assertFalse(summarize(rows, "a", "b", False)["target_met"])
        rows[-1]["a"]["decode_ms"] = [100, 100]
        self.assertFalse(summarize(rows, "a", "b", True)["target_met"])
        rows[0]["b"]["decode_ms"] = []
        with self.assertRaises(ValueError):
            summarize(rows, "a", "b", True)


class Evidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(
            (Path(__file__).resolve().parents[2] / "reports/pretrained-qwen-gb10.json").read_bytes()
        )

    def test_genuine(self):
        audit(self.data)

    def test_duplicate_prompt(self):
        j = deepcopy(self.data)
        v = j["cases"]["32"]["inputs"]
        v[2] = v[0]
        with self.assertRaisesRegex(ValueError, "prompt coverage"):
            audit(j)

    def test_recompilation(self):
        j = deepcopy(self.data)
        j["inductor_graphs_after_windows"] += 1
        with self.assertRaisesRegex(ValueError, "compiled new"):
            audit(j)

    def test_replaced_reference(self):
        j = deepcopy(self.data)
        j["cases"]["128"]["validation"]["sparkfuse"][2]["reference"][1] = -1
        with self.assertRaisesRegex(ValueError, "reference tokens"):
            audit(j)

    def test_invented_speedup(self):
        j = deepcopy(self.data)
        j["deployment_target_met"] = not j["deployment_target_met"]
        with self.assertRaisesRegex(ValueError, "cross-length"):
            audit(j)

    def test_missing_backend(self):
        j = deepcopy(self.data)
        j["cases"]["32"]["windows"][0]["order"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete paired"):
            audit(j)


if __name__ == "__main__":
    unittest.main()
