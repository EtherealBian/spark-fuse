from copy import deepcopy
from pathlib import Path
import hashlib
import json
import unittest
from tile235 import compare, TILES


def fixture():
    base = json.loads(Path("reports/qwen235b/rows.json").read_bytes())
    eog = json.loads(Path("reports/qwen235b/runtime.json").read_bytes())["eog_token_ids"]
    blocks = []
    for i, u in enumerate(TILES):
        rr = deepcopy(base)
        if u == 256:
            for r in rr["rows"]:
                for f in ("wall_ms", "prompt_ms", "decode_ms"):
                    r[f] *= 0.8
        blocks.append(
            (
                dict(
                    block=i, tile=u, process_fingerprint=hashlib.sha256(str(i).encode()).hexdigest()
                ),
                rr,
            )
        )
    return blocks, base, eog


class TileGate(unittest.TestCase):
    def test_two_orders_complete_and_baseline(self):
        b, r, e = fixture()
        self.assertTrue(compare(b, r, e)["recommend_for_this_workload"])
        b[2][1]["rows"][2]["ids"][0] ^= 1
        v = compare(b, r, e)
        self.assertFalse(v["meets_exact_output_gate"])
        self.assertTrue(v["meets_timing_gate"])

    def test_short_regression_or_one_unfavourable_order(self):
        for length in (128, 1024):
            b, r, e = fixture()
            for row in b[2][1]["rows"]:
                if len(r["inputs"][row["case"]]) == length:
                    row["wall_ms"] *= 1.45
            v = compare(b, r, e)
            self.assertTrue(v["meets_exact_output_gate"])
            self.assertFalse(v["meets_timing_gate"])

    def test_incomplete_or_duplicate_process(self):
        b, r, e = fixture()
        b[2][0]["process_fingerprint"] = b[1][0]["process_fingerprint"]
        with self.assertRaises(ValueError):
            compare(b, r, e)
        with self.assertRaises(ValueError):
            compare(b[:3], r, e)

    def test_controls_must_match_previous_public_baseline(self):
        b, r, e = fixture()
        for _, p in b:
            for x in p["rows"]:
                x["ids"][0] ^= 1
        v = compare(b, r, e)
        self.assertFalse(v["meets_exact_output_gate"])
        self.assertTrue(v["cases"]["128:0"]["cross_tile_exact"])
