"""Recompute the paired layout experiment's numerical gates, controls and timing summaries offline."""

import json
from math import isfinite, prod
from pathlib import Path
from random import Random
from statistics import median
from tools.verify_sources import verify_reports

R = Path(__file__).resolve().parents[1] / "reports/frozen-rhs-gb10.json"


def samples(v, n):
    assert len(v) == n and all(isfinite(x) and x > 0 for x in v)
    return median(v)


def test_frozen_projection_paired_layout_evidence():
    d = json.loads(R.read_bytes())
    assert not d["smoke"] and verify_reports([R])["distinct_archives"] == 1
    cases = {(c["seed"], c["label"]): c for c in d["cases"]}
    assert len(cases) == len(d["cases"]) == 8
    combos = {
        "h33": ([33, 1536], 64),
        "h33-control": ([33, 1536], 64),
        "b32": ([2, 16, 1536], 128),
        "b65": ([65, 3072], 256),
    }
    for seed in [987143, 1002857]:
        row = cases[seed, "h33"]
        col = cases[seed, "h33-control"]
        for key in ["inputs_sha256", "weight_sha256", "reference_sha256", "cpu_int32_sha256"]:
            assert row[key] == col[key]
        assert (
            len({cases[seed, name]["weight_sha256"]["weight"] for name in ("h33", "b32", "b65")})
            == 3
        )
        for label, (shape, n) in combos.items():
            c = cases[seed, label]
            assert c["shape"] == shape and c["channels"] == n
            k = shape[-1]
            m = prod(shape[:-1])
            assert c["source_layout"] == ("column" if label == "h33-control" else "row")
            st = [1, k] if label == "h33-control" else [n, 1]
            assert c["source_weight_stride"] == st
            assert (
                len(c["inputs_sha256"]) == 10
                and len({h for r in c["inputs_sha256"] for h in r}) == 20
            )
            assert (
                len(c["reference_sha256"]) == 8
                and all(len(row) == 5 for row in c["reference_sha256"])
                and c["cpu_int32_sha256"] == [out[3] for out in c["reference_sha256"]]
            )
            assert set(c["captures"]) == {
                b + "_" + policy
                for b in ["mixed", "aten"]
                for policy in ["preserve", "column_major"]
            }
            assert set(c["setup"]) == set(c["captures"]) | {"partition_mixed"}
            for name, v in c["captures"].items():
                backend, layout = name.split("_", 1)
                md = v["metadata"]
                (w,) = md["frozen_int8_weights"]
                assert w == dict(
                    name="weight",
                    shape=[k, n],
                    source_stride=st,
                    snapshot_stride=(st if layout == "preserve" else [1, k]),
                    storage_group=0,
                    bytes=k * n,
                )
                assert (
                    md["backend"] == backend
                    and md["int8_rhs_layout"] == layout
                    and md["staging_and_state_bytes"] == 4 * m * k + k * n + 4 * n
                    and md["input_bytes"] == 4 * m * k
                )
                assert (
                    v["private_distinct"]
                    and v["actual_stride"] == w["snapshot_stride"]
                    and v["actual_weight_sha256"] == c["weight_sha256"]["weight"]
                )
                assert (
                    c["setup"][name]["wall_ms"] > 0
                    and c["setup"][name]["peak_allocator_increase_bytes"]
                    >= c["setup"][name]["retained_allocator_delta_bytes"]
                )
            assert set(c["gates"]) == set(c["captures"]) | {
                "aten",
                "eager",
                "mixed",
                "cast_inductor",
                "inductor",
            }
            for name, g in c["gates"].items():
                if "error" in g:
                    assert name in ("cast_inductor", "inductor") and not g["admitted"]
                    continue
                assert len(g["outputs"]) == len(g["code_witnesses"]) == 8 and all(
                    len(v) == 5 for v in g["outputs"]
                )
                assert all(
                    (w is None) == (v[1]["different_bytes"] == 0)
                    for w, v in zip(g["code_witnesses"], g["outputs"], strict=True)
                )
                assert g["admitted"] == (
                    g["fresh"]
                    and all(
                        r["different_bytes"] == 0 and r["geometry_match"]
                        for v in g["outputs"]
                        for r in v
                    )
                )
            keep = {n for n, g in c["gates"].items() if g["admitted"]}
            assert set(c["gates"]) - {"cast_inductor", "inductor"} <= keep
            t = c["serving"]
            rng = Random(seed + n + 911)
            assert (
                len(t["windows"]) == 7
                and t["seed"] == seed + n + 911
                and set(t["median_us"]) == keep
            )
            for w in t["windows"]:
                order = sorted(keep)
                rng.shuffle(order)
                assert w["order"] == order and set(w["samples"]) == keep
            lens = {
                k: 16
                for k in [
                    "fixed_sync_us",
                    "rotating_sync_us",
                    "fixed_submit_us",
                    "rotating_submit_us",
                ]
            }
            lens["queued8_us_per_forward"] = 5
            for name, stats in t["median_us"].items():
                assert set(stats) == set(lens)
                for key, num in lens.items():
                    assert stats[key] == median(
                        samples(w["samples"][name][key], num) for w in t["windows"]
                    )
            dev = c["matched_device"]
            if label == "h33-control":
                assert dev is None
                continue
            methods = ["int_mm_row", "projection_row", "int_mm_column", "projection_column"]
            rng = Random(seed + n + 1871)
            assert len(dev["windows"]) == 7 and set(dev["profiles"]) == set(methods)
            for w in dev["windows"]:
                order = methods.copy()
                rng.shuffle(order)
                assert w["order"] == order
            for name, p in dev["profiles"].items():
                assert p["calls"] == 1 and sum(p["kernel_names"].values()) == p["count"] == (
                    3 if name.startswith("projection") else 1
                )
                if name.startswith("projection"):
                    for marker in ["ordered_int32_epilogue", "residual_token_quant"]:
                        assert sum(v for k, v in p["kernel_names"].items() if marker in k) == 1
                for mode, num in [("device", 7), ("host", 15)]:
                    assert dev["median_us"][mode][name] == median(
                        samples(w[mode][name]["samples_us"], num) for w in dev["windows"]
                    )
    assert cases[987143, "h33"]["weight_sha256"] != cases[1002857, "h33"]["weight_sha256"]
