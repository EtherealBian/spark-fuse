"""A revalidator must reject silently changed inputs, summaries and allocation claims."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit7b import check
from checkpoint7b import REPO, REV


def row():
    v = dict(
        reference=[1] * 128,
        output=[1] * 128,
        input_ids=[2] * 128,
        max_abs_by_step=[0.0] * 128,
        cosine_by_step=[1.0] * 128,
        passed=True,
    )
    w = dict(output=[1] * 128, ttft_ms=1.0, decode_ms=[2.0] * 127)
    return dict(
        format="sparkfuse.qwen2-7b.pilot.v1",
        model=REPO,
        revision=REV,
        backend="resident",
        length=128,
        generated=128,
        batch=1,
        dtype="bfloat16",
        attention="sdpa",
        cache_capacity=256,
        validation={"1": v, "2": deepcopy(v)},
        windows=[dict(deepcopy(w), prompt=i) for i in [1, 2, 2, 1]],
        graphs_after_setup=0,
        graphs_after_windows=0,
        ttft_p50_ms=1.0,
        decode_p50_ms=2.0,
        decode_p95_ms=2.0,
        eligible=True,
        model_storage_before=16_000_000_000,
        model_storage_after=16_000_000_000,
        warm_peak_allocated=17_000_000_000,
        warm_peak_reserved=18_000_000_000,
        setup_peak_allocated=17_000_000_000,
    )


class AuditTests(unittest.TestCase):
    def test_qualification(self):
        r = row()
        self.assertTrue(check(r, 128, "resident")["eligible"])
        r["validation"]["1"]["output"][10] = 4
        r["validation"]["1"]["passed"] = False
        r["eligible"] = False
        self.assertFalse(check(r, 128, "resident")["eligible"])
        self.assertEqual(check(r, 128, "resident")["first_token_mismatch"]["1"], 11)

    def test_invalid_evidence(self):
        for edit in (
            lambda r: r["windows"][0]["output"].pop(),
            lambda r: r["windows"][0]["decode_ms"].pop(),
            lambda r: r.update(decode_p95_ms=1.0),
            lambda r: r.update(model_storage_after=24_000_000_000),
            lambda r: r["validation"]["1"]["cosine_by_step"].__setitem__(0, float("nan")),
            lambda r: r.update(graphs_after_windows=2),
            lambda r: r["validation"]["1"]["reference"].__setitem__(0, 5),
        ):
            r = row()
            edit(r)
            with self.assertRaises(ValueError):
                check(r, 128, "resident")


if __name__ == "__main__":
    unittest.main()
