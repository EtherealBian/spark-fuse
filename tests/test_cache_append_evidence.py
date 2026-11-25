"""Audit cache dataflow, real specialization choices, eligibility and raw timing distributions."""

from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from tools.verify_sources import verify_reports

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"


def test_append_raw_panels_and_width_witness():
    docs = []
    for file in ("cache-append-initial-gb10.json", "cache-append-panel-gb10.json"):
        p = R / file
        assert verify_reports([p])["verified_reports"] == 1
        d = json.loads(p.read_text())
        docs.append(d)
        assert (
            d["passed"] and len(d["cases"]) == 5 and len(set([d["export_seed"], *d["seeds"]])) == 5
        )
        for c in d["cases"]:
            for idx, r in enumerate(c["layouts"]):
                assert r["offsets"] == ([0] * 4 if idx == 0 else [1, 2, 3, 5])
                eligible = {k for k, ss in r["samples"].items() if all(s["passed"] for s in ss)}
                assert {"sparkfuse", "inductor", "eager"} <= eligible
                for k in eligible:
                    ss = r["samples"][k]
                    assert [s["seed"] for s in ss] == d["seeds"]
                    assert all(
                        s["new_storage"]
                        and s["unequal_bits"] == [0, 0]
                        and all(s["matching_strides"])
                        for s in ss
                    )
                assert len(r["windows"]) == 7 and all(
                    set(w["order"]) == eligible and len(w["order"]) == len(eligible)
                    for w in r["windows"]
                )
                for metric in ("device", "host"):
                    assert r["median_us"][metric] == {
                        k: median(median(w[metric][k]["samples_us"]) for w in r["windows"])
                        for k in eligible
                    }
    a, b = docs
    assert set(a["seeds"]).isdisjoint(b["seeds"])
    for c in b["cases"]:
        assert (
            c["control"]["schedule"]["alignment_policy"] == "element_fallback"
            and c["record"]["schedule"]["alignment_policy"] == "runtime_width"
        )
        assert all("fixed_width" in r["median_us"]["device"] for r in c["layouts"])
    (c,) = [c for c in b["cases"] if c["label"] == "carry-long"]
    assert len(c["packet_witness"]) == 4
    for r, o, width in zip(c["packet_witness"], (0, 1, 2, 4), (8, 1, 2, 4), strict=True):
        assert (
            r["offset_elements"] == o
            and r["input_address_mod_16"] == [2 * o] * 4
            and r["profile"]["count"] == 1
        )
        ((name, count),) = r["profile"]["kernel_names"].items()
        assert count == 1 and f"paired_append<{width}," in name


def test_real_decode_chains_and_samples():
    p = R / "qwen-decode-cache-gb10.json"
    d = json.loads(p.read_text())
    assert (
        d["passed"]
        and verify_reports([p])["verified_reports"] == 1
        and not d["verify_only"]
        and d["profile"]
    )
    assert len(set([d["export_seed"], *d["seeds"]])) == 6 and d["timing_seed"] == d["seeds"][-1]
    assert [c["initial_prefix"] for c in d["cases"]] == [31, 255, 1023]
    for c in d["cases"]:
        n = c["initial_prefix"]
        assert [s["seed"] for s in c["controls"]] == d["seeds"] and all(
            x["masked_prob_exact"]
            and x["masked_hidden_exact"]
            and x["erased_prob_exact"]
            and x["erased_hidden_bits"] > 0
            for x in c["controls"]
        )
        for i, g in enumerate(c["graphs"]):
            (hit,) = [
                r for r in g["partitions"]["append"]["records"] if r.get("kind") == "paired_append"
            ]
            assert hit["logical_shape"] == [2, 2, n + i, 1, 128] and hit["physical_order"] == (
                ["BSHD", "BHSD"] * 2 if i == 0 else ["BHSD"] * 4
            )
            assert hit["schedule"]["alignment_policy"] == "runtime_width"
            assert (
                sha256(
                    (ROOT / f"examples/generated/cache-append/p{n}-step{i + 1}.cu").read_bytes()
                ).hexdigest()
                == hit["cuda_sha256"]
            )
            assert not any(
                r.get("kind") == "paired_append" for r in g["partitions"]["rotary"]["records"]
            )
        rows = c["backends"]
        eligible = {k for k, v in rows.items() if v["passed"]}
        assert {"eager", "exported_aten", "rotary", "append"} <= eligible and "inductor" in rows
        for k, v in rows.items():
            assert [t["seed"] for t in v["trajectories"]] == d["seeds"] and v["passed"] == all(
                t["passed"] for t in v["trajectories"]
            )
            if not v["passed"]:
                assert "timing" not in v
                continue
            for t in v["trajectories"]:
                assert len(t["steps"]) == 2
                for i, x in enumerate(t["steps"]):
                    assert (
                        x["passed"]
                        and x["prefix_exact"]
                        and x["input_unchanged"]
                        and x["new_storage"]
                        and x["masked_nonzero"] == 0
                    )
                    assert (
                        x["outputs"]["key"]["shape"]
                        == x["outputs"]["value"]["shape"]
                        == [2, 2, n + i + 1, 128]
                    )
                    if k != "inductor":
                        assert x["output_strides"] == x["expected_strides"] and all(
                            y["unequal_bits"] == 0 and y["finite"] for y in x["outputs"].values()
                        )
                    if i:
                        assert x["cache_input_strides"] == t["steps"][i - 1]["output_strides"][2:]
            for i, t in enumerate(v["timing"]):
                assert len(t["windows"]) == len(c["orderings"][i]) == 7 and all(
                    len(o) == len(eligible) and set(o) == eligible for o in c["orderings"][i]
                )
                assert sum(t["profile"]["kernel_names"].values()) == t["profile"]["count"]
                for metric in ("device", "host"):
                    assert t[metric + "_us"] == median(
                        median(w[metric]["samples_us"]) for w in t["windows"]
                    )
        for a, b in zip(rows["rotary"]["timing"], rows["append"]["timing"], strict=True):
            assert (
                b["profile"]["count"] == a["profile"]["count"] - 1
                and sum(v for k, v in b["profile"]["kernel_names"].items() if "paired_append<" in k)
                == 1
            )
