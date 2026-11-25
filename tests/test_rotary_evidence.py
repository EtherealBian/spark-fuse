"""Audit both layers of the experiment from the public files, without CUDA."""

from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from tools.verify_sources import verify_reports

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"


def test_rotary_iso_evidence():
    p = R / "paired-rotary-panel-gb10.json"
    d = json.loads(p.read_text())
    assert d["passed"] and verify_reports([p])["verified_reports"] == 1 and len(d["cases"]) == 5
    assert len(set([d["export_seed"], *d["seeds"]])) == 5 and d["timing_seed"] == d["seeds"][-1]
    for c in d["cases"]:
        assert c["record"]["kind"] == "rotary_pair" and len(c["record"]["ops"]) == 14
        assert [x["offsets"] for x in c["layouts"]] == [[0] * 4, [1, 2, 3, 5]]
        for layout in c["layouts"]:
            checks = layout["checks"]
            eligible = {n for n, samples in checks.items() if all(r["passed"] for r in samples)}
            assert {"eager", "sparkfuse"} <= eligible and {"eager", "sparkfuse", "inductor"} == set(
                checks
            )
            for rows in checks.values():
                assert [r["seed"] for r in rows] == d["seeds"]
            assert all(
                r["bit_mismatches"] == [0, 0] and r["strides_match"] for r in checks["sparkfuse"]
            )
            assert len(layout["windows"]) == 7
            for w in layout["windows"]:
                assert (
                    len(w["order"]) == len(eligible)
                    and set(w["order"]) == set(w["device"]) == set(w["host"]) == eligible
                )
            for metric in ("device", "host"):
                assert {
                    k: median(median(w[metric][k]["samples_us"]) for w in layout["windows"])
                    for k in eligible
                } == layout["median_us"][metric]


def test_rotary_layer_evidence():
    p = R / "qwen-decoder-rotary-gb10.json"
    d = json.loads(p.read_text())
    prior = json.loads((R / "qwen-decoder-barrier-gb10.json").read_text())
    assert d["passed"] and verify_reports([p])["verified_reports"] == 1
    seeds = [*d["held_out_seeds"], d["timing_seed"]]
    assert d["weight_seed"] != prior["weight_seed"] and set(seeds).isdisjoint(
        prior["held_out_seeds"] + [prior["timing_seed"], d["export_seed"]]
    )
    assert [c["seq"] for c in d["cases"]] == [33, 65, 97]
    for c in d["cases"]:
        s = c["seq"]
        g = c["graphs"]["rotary"]
        (hit,) = [r for r in g["records"] if r.get("kind") == "rotary_pair"]
        assert g["region_count"] == 5 and hit["physical_order"] == ["BSHD"] * 2
        assert (
            sha256(
                (ROOT / f"examples/generated/paired-rotary/decoder-s{s}.cu").read_bytes()
            ).hexdigest()
            == hit["cuda_sha256"]
        )
        assert (
            c["rotary_witness_pass"]
            and c["witness_pass"]
            and [r["seed"] for r in c["rotary_witness"]] == seeds
        )
        for row in c["rotary_witness"]:
            for k, h in [("query", 12), ("key", 2)]:
                r = row[k]
                assert (
                    r["finite"]
                    and r["unequal_bits"] == 0
                    and r["actual_stride"] == r["expected_stride"]
                    and r["shape"] == [2, h, s, 128]
                )
        rows = c["backends"]
        a, b = rows["split"], rows["rotary"]
        assert b["profile"]["count"] == a["profile"]["count"] - 9
        assert sum(v for k, v in b["profile"]["kernel_names"].items() if "rotary_pair(" in k) == 1
        assert [r["seed"] for r in b["samples"]] == seeds
        for x in b["samples"]:
            assert (
                x["hidden"]["unequal_bits"]
                == x["probabilities"]["unequal_bits"]
                == x["masked_nonzero"]
                == 0
            )
        eligible = {k for k, v in rows.items() if v["passed"]}
        assert all(set(w) == eligible for w in c["orderings"]) and "inductor" in rows
        for name, v in rows.items():
            if name not in eligible:
                assert "windows" not in v and "profile" not in v
                continue
            assert (
                len(v["windows"]) == 7
                and sum(v["profile"]["kernel_names"].values()) == v["profile"]["count"]
            )
            for metric in ("host", "device"):
                assert v[metric + "_us"] == median(
                    median(w[metric]["samples_us"]) for w in v["windows"]
                )
