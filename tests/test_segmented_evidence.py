"""Recompute two-step admissions, dynamic-cache properties and medians from raw data."""

import json
from pathlib import Path
from statistics import median


def test_segmented_real_layer_evidence():
    root = Path(__file__).resolve().parents[1] / "reports"
    d = json.loads((root / "segmented-qwen-gb10.json").read_bytes())
    assert not d["smoke"] and (root / d["provenance"]["archive"]).is_file()
    cases = d["cases"]
    assert (
        len(cases) == len({(c["past"], c["seed"]) for c in cases}) == 6
        and len({c["seed"] for c in cases}) == 2
    )
    for c in cases:
        expected = {"hf_eager", "exported_aten", "two_cats", "paired_append", "segmented"}
        assert len(c["input_sha256"]) == 10 and len({r[0][0] for r in c["input_sha256"]}) == 10
        assert expected <= set(c["gates"]) and (
            "inductor" in c["gates"] or "inductor" in c["build_failures"]
        )
        assert len(c["steps"]) == 2
        for i, r in enumerate(c["steps"]):
            rec = r["segmented"]
            live = rec["segmented"]
            s = c["past"] + i + 1
            assert (
                rec["plan"]["sequence"] == s
                and live["published"] == [True, True]
                and live["private_output_bytes_saved"] == 0
                and len(set(live["appends"])) == 2
                and len(r["paired"]) == 1
            )
            assert (
                live["input_strides"][1]
                == live["input_strides"][3]
                == (
                    [2 * c["past"] * 64, 64, 128, 1]
                    if i == 0
                    else [2 * (s - 1) * 64, (s - 1) * 64, 64, 1]
                )
            )
            assert len(r["binary"]) == 2
            z = r["held_intermediates"]
            assert len(z) == len({v["cpu_fp64_sha256"] for v in z}) == 8
            for v in z:
                assert (
                    v["cpu_error"]["structure"]
                    and not v["cpu_error"]["outside_tolerance"]
                    and len(v["cache_sha256"]) == 2
                    and all(
                        a == b
                        for a, b in [v["partial_sha256"], v["attention_sha256"], *v["cache_sha256"]]
                    )
                )
        for n, g in c["gates"].items():
            assert len(g["requests"]) == 8 and all(len(row) == 2 for row in g["requests"])
            rs = [r for row in g["requests"] for r in row]
            assert all(
                r["within_tolerance"] == all(o["outside"] == 0 for o in r["outputs"])
                and r["exact_cache"]
                == all(
                    o["different_bytes"] == 0 and o["sha256"] == o["ref_sha256"]
                    for o in r["outputs"][1:]
                )
                for r in rs
            )
            ok = g["all_live"] and all(
                r["within_tolerance"] and r["geometry"] and r["prefix_bitwise"] and r["fresh"]
                for r in rs
            )
            assert g["admitted"] == ok and g["exact_cache"] == (
                ok and all(r["exact_cache"] for r in rs)
            )
            assert n == "inductor" or g["exact_cache"]
            assert len(g["bitwise_vs_compact"]) == 8 and all(
                len(row) == 6 for row in g["bitwise_vs_compact"]
            )
            if n in ("two_cats", "paired_append", "segmented"):
                assert all(all(row) for row in g["bitwise_vs_compact"])
        timed = {n for n, g in c["gates"].items() if g["admitted"]}
        h, t = c["hot"], c["requests"]
        assert (
            set(c["allocator"]) == set(t["median_us"]) == set(h["profiles"]) == timed
            and len(t["windows"]) == len(h["windows"]) == 7
        )
        for n, a in c["allocator"].items():
            assert a["all_six_fresh"] and a["peak_increment_bytes"] >= a["retained_bytes"] > 0
            for k in ("device", "host"):
                assert h["median_us"][k][n] == median(
                    median(w[k][n]["samples_us"]) for w in h["windows"]
                )
            for k, num in t["median_us"][n].items():
                assert num == median(median(w["samples"][n][k]) for w in t["windows"])
