"""Independently reconstruct eligibility, per-island byte checks, and whole-request clocks."""

import json
from pathlib import Path
from statistics import median


def test_qualified_hybrid_decode_panel():
    root = Path(__file__).resolve().parents[1] / "reports"
    d = json.loads((root / "hybrid-qwen-gb10.json").read_bytes())
    assert not d["smoke"] and (root / d["provenance"]["archive"]).is_file()
    assert len(d["cases"]) == len({(c["seed"], c["masked"]) for c in d["cases"]}) == 4
    for c in d["cases"]:
        assert (
            c["qualification_past"] == [17, 65]
            and c["steps"] == 4
            and c["initial_past"] == [2, 29, 30, 31, 125, 253, 509, 2045]
        )
        assert len({x[0] for t in c["input_sha256"] for x in t}) == 32
        variants = c["variants"]
        assert (
            variants["split"]["binaries"] == variants["split_host"]["binaries"]
            and variants["split"]["regions"] == variants["split_host"]["regions"]
            and not variants["split"]["host_calls"]
            and variants["split_host"]["host_calls"]
        )
        assert all(key == v["binary_sha256"] and len(key) == 64 for key, v in c["binaries"].items())
        assert len(set(c["attention_binaries"])) == 2
        for v in variants.values():
            for rec in v["regions"]:
                if "module" in rec:
                    assert (
                        v["binaries"][rec["module"]] in c["binaries"]
                        and len(rec["output_shapes"]) == len(rec["live_outs"])
                        and rec["qualification"] == "two independent byte-exact traces"
                    )
        split = [r for r in variants["split"]["regions"] if "module" in r]
        names = {k for r in split for k in r["live_outs"]}
        assert (
            any(len(r["live_outs"]) == 2 and "split_from" in r for r in split) and len(split) >= 4
        )
        for start, rows in zip(c["initial_past"], c["checked_island_outputs"], strict=True):
            assert len(rows) == 4
            for i, row in enumerate(rows):
                assert (
                    row["sequence"] == start + i + 1
                    and row["attention"]["structure"]
                    and not row["attention"]["outside_tolerance"]
                )
                assert set(row["outputs"]) == names
                for v in row["outputs"].values():
                    assert v["stride_match"] and (v["sha256"] == v["ref_sha256"]) == (
                        v["different_bytes"] == 0
                    )
        assert c["gates"]["split"] == c["gates"]["split_host"]
        admitted = set()
        for name, g in c["gates"].items():
            flat = [x for tr in g["steps"] for x in tr]
            assert (
                len(flat) == 32 and len(g["steps"]) == 8 and all(len(tr) == 4 for tr in g["steps"])
            )
            for x in flat:
                assert x["within_tolerance"] == all(o["outside"] == 0 for o in x["outputs"])
                assert x["exact_cache"] == all(
                    o["sha256"] == o["ref_sha256"] for o in x["outputs"][1:]
                )
            assert g["admitted"] == (
                g["all_live"]
                and all(
                    x["fresh"] and x["geometry"] and x["prefix_bitwise"] and x["within_tolerance"]
                    for x in flat
                )
            )
            assert g["exact_caches"] == (g["admitted"] and all(x["exact_cache"] for x in flat))
            if g["admitted"]:
                admitted.add(name)
        assert {"attention_only", "hf_eager", "exported_aten"} <= admitted
        assert set(c["hot"]) == {"2", "2045"}
        r = c["rotating"]
        assert len(r["windows"]) == 7 and set(r["median_us"]) == admitted
        for k, m in r["median_us"].items():
            for mode, t in m.items():
                assert t == median(median(w["samples"][k][mode]) for w in r["windows"])
        for h in c["hot"].values():
            assert set(h["profiles"]) == admitted and len(h["windows"]) == 7
            for mode in ("device", "host"):
                for k, t in h["median_us"][mode].items():
                    assert t == median(median(w[mode][k]["samples_us"]) for w in h["windows"])
