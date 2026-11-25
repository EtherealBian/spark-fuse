"""Recalculate request qualification and each reported serving clock from the raw GB10 panel."""

from hashlib import sha256
import json
import math
from pathlib import Path
from random import Random
from statistics import median

from tools.verify_sources import verify_reports

R = Path(__file__).resolve().parents[1] / "reports"


def test_four_fresh_inputs_held_outputs_and_resident_timings():
    path = R / "resident-qwen-gb10.json"
    doc = json.loads(path.read_bytes())
    assert verify_reports([path])["distinct_archives"] == 1
    frozen = (R / "qwen-packed-mean-heldout-gb10.json").read_bytes()
    assert doc["rms_report_sha256"] == sha256(frozen).hexdigest() and doc["smoke"] is False
    schedules = {
        c["sequence"]: c["kernels"]["packed"]["schedule"] for c in json.loads(frozen)["cases"]
    }
    assert [(c["sequence"], c["weight_seed"]) for c in doc["cases"]] == [
        (s, w) for s in (33, 65, 129) for w in (813157, 821641)
    ]
    weights = {}
    seen = set()
    for c in doc["cases"]:
        seq, w = c["sequence"], c["weight_seed"]
        assert c["schedule"] == schedules[seq]
        ws = c["weight_sha256"]
        assert len(ws) == 12 and len(set(ws.values())) == 12
        assert weights.setdefault(w, ws) == ws
        h = c["input_sha256"]
        assert len(h) == 8 and all(len(r) == 4 for r in h)
        assert all(len({r[i] for r in h}) == 8 for i in range(4))
        ref = c["reference_sha256"]
        assert (
            len(ref) == 8
            and all(len(p) == 2 for p in ref)
            and len({q for p in ref for q in p}) == 16
        )
        assert not seen.intersection(tuple(p) for p in ref)
        seen.update(tuple(p) for p in ref)
        d = c["resident"]
        bits = 2
        stage = bits * (2 * seq * 1536 + 2 * 2 * seq * 128 + 2 * seq * seq)
        fresh = bits * (2 * seq * 1536 + 2 * 12 * seq * seq)
        for n, m in d.items():
            assert (
                m["input_bytes"] == stage
                and m["fresh_output_bytes"] == fresh
                and m["staging_and_state_bytes"] - stage == 93595648
            )
            assert [(b["source"], b["base"], b["offset"]) for b in m["output_bindings"]] == [
                ("copy", 0, 0),
                ("copy", 1, 0),
            ]
            assert [b["shape"] for b in m["output_bindings"]] == [[2, seq, 1536], [2, 12, seq, seq]]
            assert (
                c["setup"][n]["wall_ms"] > 0 and c["setup"][n]["peak_allocator_increase_bytes"] >= 0
            )
        assert (
            not d["resident_aten"]["regions"]
            and sum(r.get("schedule") == c["schedule"] for r in d["resident_mixed"]["regions"]) == 1
        )
        passed = set()
        mandatory = {"eager", "aten", "mixed", "resident_aten", "resident_mixed"}
        assert c["gates"].keys() == mandatory | {"inductor", "inductor_casts"}
        for n, g in c["gates"].items():
            assert len(g["checks"]) == 8
            qualifies = g["distinct_outputs"]
            for q in g["checks"]:
                exact = all(
                    v["finite"] and v["tolerance_pass"] and v["unequal_bits"] == 0
                    for v in q["values"].values()
                )
                assert q["passed"] == (
                    exact and q["masked_nonzero"] == 0 and q["max_row_sum_error"] <= 0.004
                )
                qualifies = qualifies and q["passed"] and q["layout_match"] and q["alias_match"]
            assert qualifies == g["passed"]
            if qualifies:
                passed.add(n)
        tm = c["timing"]
        assert mandatory <= passed and passed == tm["median_us"].keys()
        rng = Random(tm["seed"])
        assert len(tm["windows"]) == 7
        for window in tm["windows"]:
            order = sorted(passed)
            rng.shuffle(order)
            assert window["order"] == order and window["samples"].keys() == passed
            for n, samples in window["samples"].items():
                assert set(samples) == {
                    "fixed_submit_us",
                    "fixed_sync_us",
                    "rotating_submit_us",
                    "rotating_sync_us",
                    "queued8_us_per_forward",
                }
                for k, vals in samples.items():
                    assert len(vals) == (5 if k.startswith("queued") else 16) and all(
                        math.isfinite(v) and v > 0 for v in vals
                    )
                for tag in ["fixed", "rotating"]:
                    assert all(
                        host <= sync
                        for host, sync in zip(
                            samples[tag + "_submit_us"], samples[tag + "_sync_us"], strict=True
                        )
                    )
        for n, values in tm["median_us"].items():
            for k, v in values.items():
                assert v == median(median(window["samples"][n][k]) for window in tm["windows"])
    assert (
        len(weights) == 2 and len({ws["layer.mlp.up_proj.weight"] for ws in weights.values()}) == 2
    )
