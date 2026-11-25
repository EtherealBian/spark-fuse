"""Verify the private-intermediate admission gate, weight lifetime witness and timing eligibility."""

import json
from pathlib import Path
from statistics import median

from tools.verify_sources import verify_reports

ROOT = Path(__file__).resolve().parents[1]


def test_mlp_packing_admission_report():
    path = ROOT / "reports/qwen-mlp-projection-admission-gb10.json"
    assert verify_reports([path])["verified_reports"] == 1
    d = json.loads(path.read_text())
    panels = d["panels"]
    assert (
        len(panels) == 2
        and len({p["weight_seed"] for p in panels}) == 2
        and len({tuple(p["weight_sha256"]) for p in panels}) == 2
    )
    for p in panels:
        assert len(p["weight_sha256"]) == 4 and all(
            len(bytes.fromhex(x)) == 32 for x in p["weight_sha256"]
        )
        assert p["extra_packed_bytes"] == 2 * 8960 * 1536 * 2 and p["packing"][
            "median_us"
        ] == median(p["packing"]["samples_us"])
        ctrl = p["replacement_control"]
        assert (
            ctrl["snapshot_gate"]["unequal_bits"] > 0
            and ctrl["repacked_gate"]["unequal_bits"] == 0
            and ctrl["snapshot_gate"]["reference_bf16"] == 0
        )
        assert [(c["prefix"], c["stage"]) for c in p["cases"]] == [
            (n, step) for n in (31, 127) for step in ("prefill", "first", "second")
        ]
        for c in p["cases"]:
            samples = c["samples"]
            assert (
                len(samples) == 3
                and len({s["seed"] for s in samples}) == 3
                and samples[-1]["seed"] == c["timing_seed"]
            )
            assert [s["seed"] for s in c["inductor_down"]] == [s["seed"] for s in samples]
            assert len({s["input_sha256"] for s in samples}) == 3
            for s in samples:
                seq = c["prefix"] if c["stage"] == "prefill" else 1
                assert s["input"]["shape"] == [2, seq, 1536] and [
                    t["shape"] for t in s["reference"]
                ] == [[2, seq, 8960]] * 3 + [[2, seq, 1536]]
                for name, res in s["methods"].items():
                    t = res["tensors"]
                    assert len(t) == 4
                    assert res["private_intermediates_exact"] == (
                        not any(a["unequal_bits"] for a in t)
                        and all(a["stride_match"] for a in t[-2:])
                    )
                    assert res["private_intermediates_exact"]
                    views = name in ("packed_views", "pack_per_call")
                    assert res["shared_gate_up_storage"] == views and all(
                        a["stride_match"] != views for a in t[:2]
                    )
                    assert all(
                        a["stride_match"] == (a["layout"]["stride"] == b["stride"])
                        for a, b in zip(t, s["reference"], strict=True)
                    )
            eligible = {"hf"} | {
                n
                for n in samples[0]["methods"]
                if all(s["methods"][n]["private_intermediates_exact"] for s in samples)
            }
            if all(v.get("unequal_bits") == 0 and v["stride_match"] for v in c["inductor_down"]):
                eligible.add("inductor")
            for t in c["inductor_down"]:
                if t.get("unequal_bits", 0) > 0:
                    assert t["actual_bf16"] != t["reference_bf16"] and len(t["first_index"]) == 3
            assert (
                set(c["profiles"]) == eligible
                and len(c["windows"]) == 7
                and all(
                    len(w["order"]) == len(eligible) and set(w["order"]) == eligible
                    for w in c["windows"]
                )
            )
            assert (
                c["profiles"]["two"]["count"] == 5 and c["profiles"]["packed_views"]["count"] == 4
            )
            for key in ("device", "host"):
                assert c["median_us"][key] == {
                    n: median(median(w[key][n]["samples_us"]) for w in c["windows"])
                    for n in eligible
                }
