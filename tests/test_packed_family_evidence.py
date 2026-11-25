"""Recompute physical/logical byte accounting, gates and raw window summaries, offline."""

import json
from math import isfinite
from pathlib import Path
from random import Random
from statistics import median
from tools.verify_sources import verify_reports

REPORT = Path(__file__).resolve().parents[1] / "reports/packed-family-gb10.json"


def test_packed_family_ablation_has_three_graphs_and_one_private_weight():
    r = json.loads(REPORT.read_bytes())
    assert not r["smoke"] and verify_reports([REPORT])["distinct_archives"] == 1
    rows = r["rows"]
    ns = r["widths"]
    k = r["k"]
    assert (rows, ns, k) == ([17, 33, 129], [512, 256, 256], 1536)
    matrix = k * sum(ns)
    state = matrix + 4 * sum(ns)
    stage = 4 * k * sum(rows)
    assert [c["seed"] for c in r["cases"]] == [1556219, 1569401]
    inputs = set()
    weightsets = set()
    for c in r["cases"]:
        wh = c["weights_sha256"]
        assert len(wh) == 6 and weightsets.isdisjoint(wh.values())
        weightsets.update(wh.values())
        iq = [v for a in c["held_inputs_sha256"] for v in a] + [
            v for pair in c["admission_sha256"] for a in pair for v in a
        ]
        assert len(iq) == len(set(iq)) == 28 and inputs.isdisjoint(iq)
        inputs.update(iq)
        assert c["held_rows"] == [17, 33, 129, 33, 17, 129, 17, 129]
        ref = c["reference_sha256"]
        assert (
            len(ref) == 8
            and all(len(y) == 9 for y in ref)
            and c["cpu_int32_sha256"] == [y[3::2] for y in ref]
        )
        i, s = (c["isolated_allocator"][name] for name in ("independent", "shared"))
        assert i["baseline_allocated"] == s["baseline_allocated"] > 0
        for name, copies in [("independent", 3), ("shared", 1)]:
            mem = c["isolated_allocator"][name]
            b = c["inventory"][name]
            assert mem["kind"] == name and mem["inventory"] == b and mem["weight_sha256"] == wh
            assert (
                mem["warm_setup_ms"] > 0
                and mem["peak_build_delta"] >= mem["retained_delta"] >= b["static_bytes"]
                and mem["after_close_delta"] == 0
            )
            assert (
                b["private_state_stores"] == 4 * copies
                and b["private_packed_rhs"] == copies
                and b["private_packed_rhs_bytes"] == matrix * copies
            )
            assert (
                b["state_bytes"] == state * copies
                and b["input_staging_bytes"] == stage
                and b["input_stores"] == 6
                and b["static_bytes"] == stage + state * copies
            )
            assert [v["shape"] for v in b["members"]] == [[m, k, sum(ns)] for m in rows]
            for m, v in zip(rows, b["members"], strict=True):
                assert (
                    v["staging"] == 4 * m * k
                    and v["state"] == state
                    and len(v["cuda_sha256"]) == 64
                )
                desc = v["source_weights"]
                assert [(t["name"], t["shape"], t["packed_column_offset"]) for t in desc] == [
                    (f"weight_{j}", [k, n], sum(ns[:j])) for j, n in enumerate(ns)
                ]
        assert i["inventory"]["members"] == s["inventory"]["members"]
        assert i["retained_delta"] - s["retained_delta"] == 2 * state
        gates = c["gates"]
        assert set(gates) == {"inductor", "cast_inductor", "eager", "independent", "shared"}
        for g in gates.values():
            assert g["fresh"] and len(g["outputs"]) == 8 and all(len(y) == 9 for y in g["outputs"])
            assert g["admitted"] == all(
                v["geometry_match"] and v["different_bytes"] == 0
                for row in g["outputs"]
                for v in row
            )
        passed = {n for n, g in gates.items() if g["admitted"]}
        assert {"eager", "shared", "independent"} <= passed
        t = c["serving"]
        assert (
            set(t["median_us"]) == passed
            and len(t["windows"]) == 7
            and t["seed"] == c["seed"] + 811
        )
        rng = Random(t["seed"])
        lens = {
            "fixed_sync_us": 16,
            "fixed_submit_us": 16,
            "rotating_sync_us": 16,
            "rotating_submit_us": 16,
            "queued8_us_per_forward": 5,
        }
        for w in t["windows"]:
            order = sorted(passed)
            rng.shuffle(order)
            assert w["order"] == order and set(w["samples"]) == passed
        for n, stats in t["median_us"].items():
            assert set(stats) == set(lens)
            for key, num in lens.items():
                windows = [w["samples"][n][key] for w in t["windows"]]
                assert all(len(v) == num and all(isfinite(a) and a > 0 for a in v) for v in windows)
                assert stats[key] == median(map(median, windows))
