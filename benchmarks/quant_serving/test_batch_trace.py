import copy
from contextlib import redirect_stdout
from io import StringIO
import os
from threading import Lock
import time
import unittest
from unittest.mock import patch
import batch_trace as b


class Trace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        slots = {}
        lock = Lock()

        def api(port, path, req):
            if path == "tokenize":
                return {"tokens": list(req["content"].encode())}
            slot = req["id_slot"]
            ids = req["prompt"]
            n = req["n_predict"]
            use = req["cache_prompt"]
            assert (
                req["temperature"] == 0
                and req["logit_bias"] == [[151643, False]]
                and req["return_tokens"]
                and not req["stream"]
            )
            with lock:
                match = b.common(slots.get(slot, []), ids) if use else 0
                slots[slot] = ids[:]
            time.sleep(0.015 if use else 0.001)
            return dict(
                id_slot=slot,
                stop_type="limit",
                tokens=[3] * n,
                timings=dict(
                    prompt_n=len(ids) - match,
                    cache_n=match,
                    predicted_n=n,
                    prompt_ms=0.1,
                    predicted_ms=0.1,
                ),
            )

        saves = []
        with (
            patch.dict(os.environ, SPARKFUSE_EOG_IDS="[151643]"),
            patch.object(b, "post", side_effect=api),
            redirect_stdout(StringIO()),
        ):
            cls.raw = b.run(0, lambda rows: saves.append(sum(len(c["arms"]) for c in rows)))
        assert saves == list(range(1, 13))

    def test_panel_and_divergence(self):
        d = copy.deepcopy(self.raw)
        d[0]["arms"][1]["requests"][0]["response"]["ids"][4] = 5
        g = b.audit(d, [151643])["groups"]
        w = g["contended"]
        a = w["repeats"][0]["arms"]
        self.assertEqual((a["greedy"]["processed"], a["joint"]["processed"]), (1088, 768))
        self.assertEqual(w["exact_pairs"], 3)
        self.assertEqual(w["repeats"][0]["first_different_token"], [5, None])
        self.assertEqual(g["disjoint"]["exact_pairs"], 4)

    def test_corruption_rejected(self):
        for field, val in [
            ("observed_slot", 1),
            ("cache_prompt", False),
            ("reused", 640.0),
            ("processed", 3),
            ("masked_eog", []),
            ("ids", [151643] * 16),
            ("input_sha256", "bad"),
        ]:
            with self.subTest(field=field):
                d = copy.deepcopy(self.raw)
                d[0]["arms"][0]["requests"][0]["response"][field] = val
                with self.assertRaises(ValueError):
                    b.audit(d, [151643])
        for kind in ("overlap", "slot", "order", "seed", "matrix", "missing"):
            with self.subTest(kind=kind):
                d = copy.deepcopy(self.raw)
                arm = d[0]["arms"][0]
                r = arm["requests"]
                if kind == "overlap":
                    delta = r[1]["end_ns"]
                    r[0]["start_ns"] += delta
                    r[0]["end_ns"] += delta
                elif kind == "slot":
                    r[0]["slot"] = 1
                elif kind == "order":
                    d[0]["arms"].reverse()
                elif kind == "matrix":
                    d[0]["inputs"]["seeds"][0][1] ^= 1
                elif kind == "missing":
                    d.pop()
                else:
                    arm["preparations"].reverse()
                with self.assertRaises(ValueError):
                    b.audit(d, [151643])
