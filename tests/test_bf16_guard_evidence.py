"""Recompute admission and timings; bind the isolated measurement to the full layer."""

from hashlib import sha256
import json
from pathlib import Path
from statistics import median

from tools.verify_sources import verify_reports

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "examples/generated/guarded-scores"


def read(stem):
    path = ROOT / "reports" / (stem + "-gb10.json")
    assert verify_reports([path])["verified_reports"] == 1
    return json.loads(path.read_text())


def test_quotient_boundary_encodings_and_production_source():
    from sparkfuse.score_rounding import GUARDED_QUOTIENT

    audit = read("bf16-quotient-rounding")
    src = (SOURCES / "quotient-audit.cu").read_bytes()
    assert sha256(src).hexdigest() == audit["source_sha256"] and GUARDED_QUOTIENT.encode() in src
    assert audit["passed"] and set(audit["cases"]) == {
        "bf16_midpoints",
        "random_positive_bits",
        "special",
    }
    for name, row in audit["cases"].items():
        assert row["eager_reference_differences"] == row["guarded_differences"] == 0
        assert row["elements"] == sum(row["branches"]) > 0
        if name != "special":
            assert row["unguarded_differences"] >= len(row["witness"]) > 0
        for ex in row["witness"]:
            assert ex["guard"] == ex["exact"] != ex["mul"] and ex["branch"] in (1, 2)


def check_windows(wins, allowed, key):
    assert len(wins) == 7
    for w in wins:
        assert len(w["order"]) == len(allowed) and set(w["order"]) == set(w[key]) == allowed
    return {k: median(median(w[key][k]["samples_us"]) for w in wins) for k in allowed}


def test_real_qwen_fx_slice_and_trajectory_admission():
    layer = read("qwen-decode-guarded-scores")
    slice_ = read("qwen-score-slice-guarded")
    assert layer["passed"] and slice_["passed"] and layer["score_panel"] and layer["guard_panel"]
    for k in (
        "weight_seed",
        "export_seed",
        "seeds",
        "timing_seed",
        "device",
        "torch",
        "transformers",
    ):
        assert layer[k] == slice_[k]
    seeds = layer["seeds"]
    assert (
        len(set(seeds)) == 5
        and slice_["timing_seed"] == seeds[-1]
        and slice_["export_seed"] not in seeds
    )
    assert [c["prefix"] for c in slice_["cases"]] == [1024, 1535, 2047]
    for c, full in zip(slice_["cases"], layer["cases"], strict=True):
        past = c["prefix"]
        assert full["initial_prefix"] == past and full["passed"] and c["passed"]
        assert c["export_graph_sha256"] == sha256(full["graphs"][0]["aten"].encode()).hexdigest()
        assert [s["seed"] for s in c["samples"]] == seeds
        for s in c["samples"]:
            assert len(s["whole_layer"]) == 4 and all(x["passed"] for x in s["whole_layer"])
            q, mask = s["live_ins"]
            assert q["shape"] == s["probability"]["shape"] == [2, 12, 1, past + 1]
            assert mask["shape"] == [2, 1, 1, past + 1] and all(
                len(x["sha256"]) == 64 for x in (q, mask, s["probability"])
            )
            for x in s["methods"].values():
                if "error" not in x:
                    assert x["passed"] == (not x["unequal_bf16"] and x["stride_match"])
        assert len({s["live_ins"][0]["sha256"] for s in c["samples"]}) == len(seeds)
        allowed = {
            k
            for k in c["samples"][0]["methods"]
            if all(s["methods"][k]["passed"] for s in c["samples"])
        }
        assert {"slots", "guarded", "eager"} <= allowed and set(c["profiles"]) == allowed
        for metric in ("device", "host"):
            assert check_windows(c["windows"], allowed, metric) == c["median_us"][metric]
        for step, g in enumerate(full["graphs"], 1):
            for name in ("slots", "guarded"):
                recs = g["partitions"][name]["records"]
                score = [r for r in recs if r.get("module") and "softmax" in r.get("ops", [])]
                if past + step == 2049:
                    assert not score and any(
                        r.get("reason") == "outside_ordered_score_domain" for r in recs
                    )
                else:
                    (r,) = score
                    src = (SOURCES / f"p{past}-step{step}-{name}.cu").read_bytes()
                    digest = sha256(src).hexdigest()
                    assert r["cuda_sha256"] == digest
                    if step == 1:
                        b = c["builds"][name]
                        assert b["source_sha256"] == digest
                        assert (
                            b["record"]["live_ins"] == c["live_ins"]
                            and b["record"]["live_outs"] == c["live_outs"]
                        )
        for name, backend in full["backends"].items():
            assert backend["passed"] == all(t["passed"] for t in backend["trajectories"])
            assert ("timing" in backend) == backend["passed"]
            if name in ("slots", "guarded"):
                for t in backend["trajectories"]:
                    assert len(t["steps"]) == 2
                    for row in t["steps"]:
                        assert (
                            row["passed"]
                            and row["new_storage"]
                            and row["input_unchanged"]
                            and row["prefix_exact"]
                        )
                        assert (
                            row["output_strides"] == row["expected_strides"]
                            and not row["masked_nonzero"]
                        )
                        assert len(row["outputs"]) == 4 and all(
                            v["unequal_bits"] == 0 for v in row["outputs"].values()
                        )
            for timing in backend.get("timing", []):
                assert len(timing["windows"]) == 7
                for metric in ("device", "host"):
                    assert timing[metric + "_us"] == median(
                        median(w[metric]["samples_us"]) for w in timing["windows"]
                    )
        control = c["mask_control"]
        pairs = set()
        assert set(control["checks"]) == {"observed", "periodic", "prefix", "window"}
        for kind, samples in control["checks"].items():
            assert [s["seed"] for s in samples] == seeds
            for s, ref in zip(samples, c["samples"], strict=True):
                assert (s["mask_sha256"] == ref["live_ins"][1]["sha256"]) == (kind == "observed")
                assert min(s["masked_per_batch"]) >= 1
            for name in ("slots", "guarded"):
                if all(s["checks"][name]["passed"] for s in samples):
                    pairs.add(kind + ":" + name)
        assert control["passed"] and len(pairs) == 8
        assert check_windows(control["windows"], pairs, "device") == control["median_us"]
