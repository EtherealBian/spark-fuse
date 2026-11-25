"""Independently check gates, alias/layout facts and medians for the two-step layer test."""

from pathlib import Path
import json
from statistics import median


def test_qwen_sdpa_step_evidence():
    root = Path(__file__).resolve().parents[1] / "reports"
    d = json.loads((root / "qwen-sdpa-sites-gb10.json").read_bytes())
    assert not d["smoke"] and (root / d["provenance"]["archive"]).is_file()
    cases = d["cases"]
    assert (
        len(cases) == len({(c["seed"], c["past"], c["masked"]) for c in cases}) == 6
        and len({c["seed"] for c in cases}) == 2
    )
    weights = {}
    for c in cases:
        weights.setdefault(c["seed"], c["weights"])
        assert weights[c["seed"]] == c["weights"]
        assert (
            len(c["inputs"]) == 10
            and len({v[0][0] for v in c["inputs"]}) == len({v[1][0] for v in c["inputs"]}) == 10
        )
        assert len(c["steps"]) == 2
        expected = {"hf_eager", "exported_aten", "cuda_keep", "cuda_compact", "cuda_compact_g4"}
        assert expected <= set(c["gates"]) and (
            "inductor" in c["gates"] or "inductor" in c["build_failures"]
        )
        for j, step in enumerate(c["steps"]):
            assert "scaled_dot_product_attention" in step["graph"] and set(
                step["sites"]
            ) == expected - {"hf_eager", "exported_aten"}
            s = c["past"] + j + 1
            for name, k in step["sites"].items():
                compact = name != "cuda_keep"
                r = k["site"]
                p = r["plan"]
                assert (
                    p["sequence"] == s
                    and p["kv_heads"] == (2 if compact else 8)
                    and p["head_group"] == (4 if name.endswith("g4") else 1)
                )
                assert (
                    len(set(r["erased_repeats"])) == 2 * compact
                    and r["removed_materialized_bytes"] == 2 * 2 * 8 * s * 64 * 2 * compact
                )
                assert (
                    len(k["kernels"]) == 2
                    and len(k["site_checks"]) == 8
                    and len({m["fp64_sha256"] for m in k["site_checks"]}) == 8
                )
                for v in k["site_checks"]:
                    assert (
                        v["metrics"]["structure"]
                        and v["metrics"]["outside_tolerance"] == 0
                        and v["metrics"]["output_stride"]
                        == v["original_stride"]
                        == p["output_stride"]
                    )
                    assert len(v["operands"]) == 3 + int(c["masked"])
            assert [v["metrics"]["sha256"] for v in step["sites"]["cuda_keep"]["site_checks"]] == [
                v["metrics"]["sha256"] for v in step["sites"]["cuda_compact"]["site_checks"]
            ]
        for name, g in c["gates"].items():
            assert len(g["requests"]) == 8 and all(len(row) == 2 for row in g["requests"])
            rows = [r for row in g["requests"] for r in row]
            for r in rows:
                assert len(r["outputs"]) == 3 and r["within_tolerance"] == all(
                    o["outside"] == 0 for o in r["outputs"]
                )
                assert r["exact_cache"] == all(
                    o["different_bytes"] == 0 and o["sha256"] == o["ref_sha256"]
                    for o in r["outputs"][1:]
                )
            ok = g["all_live"] and all(
                r["geometry"] and r["fresh"] and r["prefix_bitwise"] and r["within_tolerance"]
                for r in rows
            )
            assert g["numerically_admitted"] == ok and g["exact_cache_admitted"] == (
                ok and all(r["exact_cache"] for r in rows)
            )
            assert name == "inductor" or g["exact_cache_admitted"]
        timed = {n for n, g in c["gates"].items() if g["numerically_admitted"]}
        h = c["hot"]
        t = c["requests"]
        assert set(h["profiles"]) == set(t["median_us"]) == timed
        assert len(h["windows"]) == len(t["windows"]) == 7
        for n in timed:
            assert h["profiles"][n]["count"] > 0
            for mode in ("host", "device"):
                assert h["median_us"][mode][n] == median(
                    median(w[mode][n]["samples_us"]) for w in h["windows"]
                )
            for mode, us in t["median_us"][n].items():
                assert us == median(median(w["samples"][n][mode]) for w in t["windows"])
    assert len({tuple(sorted(x.items())) for x in weights.values()}) == 2
