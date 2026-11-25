"""Recompute cache-layout, ablation and timing evidence from the published samples."""

import json
from pathlib import Path
from statistics import median


def verify_timing(t, paths):
    assert set(t["profiles"]) == paths and len(t["windows"]) == 7
    for k, prof in t["profiles"].items():
        assert prof["count"] == sum(prof["kernel_names"].values()) and prof["calls"] == 1
    for mode in ("host", "device"):
        for n in paths:
            assert t["median_us"][mode][n] == median(
                median(w[mode][n]["samples_us"]) for w in t["windows"]
            )


def test_functional_append_report():
    root = Path(__file__).resolve().parents[1] / "reports"
    r = json.loads((root / "dynamic-append-gb10.json").read_bytes())
    assert not r["smoke"] and (root / r["provenance"]["archive"]).is_file()
    assert len(r["cases"]) == 12 and {
        (c["family"], c["masked"], c["seed"]) for c in r["cases"]
    } == {
        (f, m, s)
        for f in ("qwen2", "llama", "mistral")
        for m in (True, False)
        for s in (3823703, 3836107)
    }
    for c in r["cases"]:
        assert (
            c["qualification"] == [17, 257]
            and c["past"] == [2, 15, 31, 125, 513, 2045, 4093, 8189]
            and c["steps"] == 4
        )
        assert c["inductor_backend_calls"] >= 1 and set(c["gates"]) == {
            "hf_eager",
            "joined",
            "separate",
            "inductor_dynamic",
        }
        assert len(c["binary"]["sass"]) == 3 and len(c["binary"]["binary_sha256"]) == 64
        a, b = c["variants"]["joined"], c["variants"]["separate"]
        assert (
            not b["appends"]
            and a["attention"] == b["attention"]
            and a["rotary"] == b["rotary"]
            and a["host_calls"] == b["host_calls"]
        )
        (rec,) = a["appends"]
        assert (
            rec["source_sha256"] == c["binary"]["source_sha256"]
            and rec["past"] == [2, 8192]
            and len(rec["ops"]) == 2
        )
        plan = a["attention"][0]["plan"]
        assert len({xs[0] for row in c["inputs"] for xs in row}) == 32
        for k, g in c["gates"].items():
            assert len(g["steps"]) == 8 and all(len(x) == 4 for x in g["steps"])
            flat = [z for row in g["steps"] for z in row]
            assert g["admitted"] == (
                g["all_live"]
                and all(
                    z["within_tolerance"] and z["geometry"] and z["fresh"] and z["prefix_bitwise"]
                    for z in flat
                )
            )
            assert g["exact_caches"] == (g["admitted"] and all(z["exact_cache"] for z in flat))
            if k != "inductor_dynamic":
                assert g["exact_caches"]
        for x, y in zip(
            c["gates"]["separate"]["steps"], c["gates"]["joined"]["steps"], strict=True
        ):
            assert x == y
        for row, (v, layout) in zip(
            c["cache_sites"],
            [(2, "BSHD"), (3, "BHSD"), (8189, "BSHD"), (8190, "BHSD")],
            strict=True,
        ):
            assert (
                row["past"] == v
                and row["layout"] == layout
                and row["bytes_read_plus_written"]
                == 8 * plan["batch"] * plan["kv_heads"] * (v + 1) * plan["width"]
            )
            assert len(row["result_sha256"]) == 2 and all(
                len(s) == 64 for s in row["result_sha256"]
            )
            t = row["timing"]
            verify_timing(t, {"paired", "two_cats"})
            assert t["profiles"]["paired"]["count"] == 1 and t["profiles"]["two_cats"]["count"] == 2
        admitted = {k for k, g in c["gates"].items() if g["admitted"]}
        mix = c["rotating"]
        assert len(mix["windows"]) == 7 and set(mix["median_us"]) == admitted
        for k, med in mix["median_us"].items():
            for m, n in med.items():
                assert n == median(median(w["samples"][k][m]) for w in mix["windows"])
        assert set(c["hot"]) == {"2", "8189"}
        for win in c["hot"].values():
            verify_timing(win["timing"], admitted)
            assert set(win["alloc"]) == admitted and all(
                x["peak_allocator_increase_bytes"] >= x["live_output_bytes"] > 0
                and x["distinct_storages"] == 12
                for x in win["alloc"].values()
            )
            pr = win["timing"]["profiles"]
            assert pr["separate"]["count"] - pr["joined"]["count"] == 4
