"""Recompute copy/layer eligibility; verify layout and emitted CUDA independently of timings."""

from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from tools.verify_sources import verify_reports

ROOT = Path(__file__).resolve().parents[1]


def windows(ws, profiles, summary, eligible):
    assert len(ws) == 7 and set(profiles) == eligible
    for w in ws:
        assert len(w["order"]) == len(eligible) and set(w["order"]) == eligible
        for kind, count in [("host", 15), ("device", 7)]:
            assert set(w[kind]) == eligible and all(
                len(w[kind][n]["samples_us"]) == count for n in eligible
            )
    for kind in ("host", "device"):
        assert summary[kind] == {
            n: median(median(w[kind][n]["samples_us"]) for w in ws) for n in eligible
        }


def source(rec):
    key = rec["cuda_sha256"]
    assert rec["widths"] == [1536, 256, 256] and rec["schedule"] == {
        "max_packet_elements": 8,
        "threads": 128,
    }
    assert (
        sha256((ROOT / "examples/generated/split-copy3" / (key + ".cu")).read_bytes()).hexdigest()
        == key
    )
    return key


def test_split_copy_evidence():
    paths = [ROOT / "reports" / f"qkv-layout-layer{s}-gb10.json" for s in ("", "-heldout")]
    assert verify_reports(paths)["verified_reports"] == 2
    reports = [json.loads(p.read_text()) for p in paths]
    assert (
        len({r["weight_seed"] for r in reports})
        == len({r["checkpoint"]["packed_sha256"][0] for r in reports})
        == len({r["checkpoint"]["packed_sha256"][1] for r in reports})
        == 2
    )
    for r in reports:
        assert (
            r["passed"]
            and not r["verify_only"]
            and r["checkpoint"]["parameter_bytes"] == 6295552
            and [c["prefix"] for c in r["cases"]] == [127, 1024]
        )
        seeds = r["layer_seeds"]
        assert len(set(seeds)) == 5 and r["export_seed"] not in seeds
        for c in r["cases"]:
            assert set(c["backends"]) == {
                "direct",
                "reexport",
                "views",
                "separate",
                "copies",
                "one_copy_kernel",
                "eager",
                "inductor",
            }
            for n, b in c["backends"].items():
                assert [t["seed"] for t in b["trajectories"]] == seeds
                for t in b["trajectories"]:
                    for step in t["steps"]:
                        if "error" in step:
                            assert n == "inductor" and not step["passed"]
                            continue
                        strict = n != "inductor"
                        assert set(step["outputs"]) == {"hidden", "probabilities", "key", "value"}
                        assert step["passed"] == (
                            step["new_storage"]
                            and step["input_unchanged"]
                            and step["prefix_exact"]
                            and step["masked_nonzero"] == 0
                            and step["max_row_sum_error"] <= 0.004
                            and (not strict or step["output_strides"] == step["expected_strides"])
                            and all(
                                o["tolerance_pass"] and (not strict or o["unequal_bits"] == 0)
                                for o in step["outputs"].values()
                            )
                        )
                    assert t["passed"] == (
                        len(t["steps"]) == 2 and all(z["passed"] for z in t["steps"])
                    )
                assert b["passed"] == all(t["passed"] for t in b["trajectories"])
                assert n == "inductor" or b["passed"]
            allowed = {n for n, b in c["backends"].items() if b["passed"]}
            for graph, ws, profiles, meds in zip(
                c["graphs"], c["windows"], c["profiles"], c["median_us"], strict=True
            ):
                windows(ws, profiles, meds, allowed)
                for n, g in graph.items():
                    kinds = [r.get("kind") for r in g["records"]]
                    assert kinds.count("split_copy3") == int(n == "one_copy_kernel")
                    for k in ["rotary_pair", "paired_append"]:
                        assert kinds.count(k) == int(n != "views")
                (hit,) = [
                    rr
                    for rr in graph["one_copy_kernel"]["records"]
                    if rr.get("kind") == "split_copy3"
                ]
                assert source(hit) == source(c["isolated"][1]["record"])
                assert profiles["copies"]["count"] - profiles["one_copy_kernel"]["count"] == 2
            assert [s["stage"] for s in c["isolated"]] == ["prefill", "decode1", "decode2"]
            for iso in c["isolated"]:
                source(iso["record"])
                assert len(iso["rows"]) == 3 and len({row["x_sha256"] for row in iso["rows"]}) == 3
                for row in iso["rows"]:
                    assert len(row["sliced_checkpoint_against_hf"]) == len(
                        row["wide_against_hf"]
                    ) == 3 and all(
                        z["unequal_bits"] == 0 and z["stride_match"]
                        for z in row["sliced_checkpoint_against_hf"]
                    )
                    if c["prefix"] == 127 and iso["stage"] == "prefill":
                        assert row["wide_against_hf"][1]["unequal_bits"] > 0
                    for n, b in row["copy_backends"].items():
                        if "error" in b:
                            assert n == "inductor" and not b["passed"]
                            continue
                        assert b["passed"] == (
                            b["independent"]
                            and len(b["tensors"]) == 3
                            and all(
                                z["unequal_bits"] == 0
                                and z["stride_match"]
                                and z["layout"]["storage_offset"] == 0
                                for z in b["tensors"]
                            )
                        )
                        assert n == "inductor" or b["passed"]
                if iso["stage"] == "decode2":
                    assert "windows" not in iso
                    continue
                eligible = {
                    n
                    for n in iso["rows"][0]["copy_backends"]
                    if all(row["copy_backends"][n]["passed"] for row in iso["rows"])
                }
                windows(iso["windows"], iso["profiles"], iso["median_us"], eligible)
                assert (
                    iso["profiles"]["three_clones"]["count"] == 3
                    and iso["profiles"]["fx_copy_kernel"]["count"]
                    == iso["profiles"]["native_copy_kernel"]["count"]
                    == 1
                )
