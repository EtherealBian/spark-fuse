"""Offline audit of the paired full-layer benchmark and single-call kernel census."""

import json
from pathlib import Path
from statistics import median
import pytest
from tools.verify_sources import verify_reports

REPORTS = Path(__file__).resolve().parents[1] / "reports"


@pytest.mark.parametrize("profile", [False, True])
def test_layer_evidence(profile):
    file = "qwen-decoder-kernel-census-gb10.json" if profile else "qwen-decoder-barrier-gb10.json"
    path = REPORTS / file
    d = json.loads(path.read_text())
    assert d["passed"] and not d["verify_only"] and d.get("profile", False) == profile
    assert verify_reports([path])["verified_reports"] == 1
    seeds = d["held_out_seeds"] + [d["timing_seed"]]
    assert len(set(seeds + [d["export_seed"]])) == 8
    assert [c["seq"] for c in d["cases"]] == [33, 65, 97]
    for c in d["cases"]:
        s = c["seq"]
        assert c["passed"] and c["witness_pass"]
        assert [r["seed"] for r in c["residual_witness"]] == seeds
        for r in c["residual_witness"]:
            for key in ("residual", "normalized"):
                assert (
                    r[key]["shape"] == [2, s, 1536]
                    and r[key]["finite"]
                    and r[key]["unequal_bits"] == 0
                )
        prior, split = (c["graphs"][k] for k in ("prior", "split"))
        assert prior["region_count"] == 3 and split["region_count"] == 4
        (hit,) = [r for r in split["records"] if r.get("module") == c["witnessed_module"]]
        assert hit["split"]["reason"] == "fallback_barrier" and hit["split"]["covered_ops"] == 9
        assert (
            len(hit["live_outs"]) == 2
            and hit["split"]["logical_output_bytes"] == 2 * 2 * s * 1536 * 2
        )
        assert "aten.linear.default" in split["fx"] and "aten.matmul.default" in split["fx"]
        backends = c["backends"]
        eligible = {k for k, v in backends.items() if v["passed"]}
        assert {"eager", "exported_aten", "prior", "split"} <= eligible
        assert "inductor" in backends
        assert len(c["orderings"]) == 7 and all(
            set(w) == eligible and len(w) == len(eligible) for w in c["orderings"]
        )
        for k, v in backends.items():
            assert [r["seed"] for r in v["samples"]] == seeds
            assert v["passed"] == all(r["passed"] for r in v["samples"])
            if not v["passed"]:
                assert (
                    k == "inductor"
                    and "windows" not in v
                    and "device_us" not in v
                    and "profile" not in v
                )
                assert any(
                    "runtime_error" in r
                    or not r["hidden"]["tolerance_pass"]
                    or not r["probabilities"]["tolerance_pass"]
                    for r in v["samples"]
                )
                continue
            for r in v["samples"]:
                assert r["masked_nonzero"] == 0 and 0 <= r["max_row_sum_error"] <= 0.004
                assert r["hidden"]["shape"] == [2, s, 1536] and r["probabilities"]["shape"] == [
                    2,
                    12,
                    s,
                    s,
                ]
                if k != "inductor":
                    assert r["hidden"]["unequal_bits"] == r["probabilities"]["unequal_bits"] == 0
            assert len(v["windows"]) == 7
            for metric, count in [("device", 7), ("host", 15)]:
                times = [w[metric]["samples_us"] for w in v["windows"]]
                assert all(len(t) == count and min(t) > 0 for t in times)
                assert v[metric + "_us"] == median(median(t) for t in times)
            if profile:
                census = v["profile"]
                assert (
                    census["calls"] == 1
                    and census["count"] == sum(census["kernel_names"].values()) > 0
                )
        if profile:
            assert backends["split"]["profile"]["count"] < backends["prior"]["profile"]["count"]
