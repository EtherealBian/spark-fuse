import copy
import unittest
from measure235 import audit, digest


class Feasibility(unittest.TestCase):
    def test_fixed_length_cold_outputs_and_reverse_order(self):
        keys = [f"{n}:{i}" for n in (128, 1024) for i in range(2)]
        inputs = {k: [i] * int(k.split(":")[0]) for i, k in enumerate(keys)}
        rows = [
            dict(
                rep=j // 4,
                case=k,
                input_sha256=digest(inputs[k]),
                processed=len(inputs[k]),
                reused=0,
                ids=[4] * 16,
                predicted=16,
                stop="limit",
                wall_ms=3,
                prompt_ms=1,
                decode_ms=1,
            )
            for j, k in enumerate(keys + keys[::-1])
        ]
        raw = dict(inputs=inputs, rows=rows)
        result = audit(raw, {10})
        self.assertEqual(result["cases"]["128"]["repeated_greedy_exact"], 2)
        raw["rows"][-1]["ids"][1] = 6
        self.assertEqual(audit(raw, {10})["cases"]["128"]["repeated_greedy_exact"], 1)
        for k, v in [
            ("ids", [10] * 16),
            ("predicted", 8),
            ("reused", 16),
            ("rep", 1),
            ("case", "1024:1"),
            ("wall_ms", float("nan")),
        ]:
            x = copy.deepcopy(raw)
            x["rows"][0][k] = v
            with self.assertRaises(ValueError):
                audit(x, {10})
