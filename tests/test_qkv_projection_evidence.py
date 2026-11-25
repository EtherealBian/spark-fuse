"""Private/visible QKV gates, HF cache checks, rejected counterexamples and raw timings."""

import json
from pathlib import Path
from statistics import median
from tools.verify_sources import verify_reports

ROOT = Path(__file__).resolve().parents[1]


def test_qkv_projection_evidence():
    path = ROOT / "reports/qwen-qkv-projection-admission-gb10.json"
    assert verify_reports([path])["verified_reports"] == 1
    panels = json.loads(path.read_text())["panels"]
    assert len(panels) == 2 and all(
        len({p[key] for p in panels}) == 2
        for key in ("weight_seed", "packed_weight_sha256", "packed_bias_sha256")
    )
    for p in panels:
        assert (
            p["sizes"] == p["nonzero_bias"] == [1536, 256, 256]
            and len(p["bias_sha256"]) == len(p["weight_sha256"]) == 3
        )
        assert len(p["seeds"]) == len(set(p["seeds"])) == 3
        assert [(c["prefix"], c["stage"]) for c in p["cases"]] == [
            (i, s) for i in (31, 127, 255) for s in ("prefill", "decode1", "decode2")
        ]
        for c in p["cases"]:
            samples = c["samples"]
            assert (
                [s["seed"] for s in samples] == p["seeds"]
                and c["timing_seed"] == p["seeds"][-1]
                and len({s["input_sha256"] for s in samples}) == 3
            )
            names = {"three", "packed_views", "packed_copies", "q_plus_kv", "late_bias", "inductor"}
            for s in samples:
                assert s["input"]["shape"] == [
                    2,
                    c["prefix"] if c["stage"] == "prefill" else 1,
                    1536,
                ]
                assert (
                    len(s["reference_sha256"]) == 3
                    and len(s["actual_cache"]) == 2
                    and all(x["unequal_bits"] == 0 for x in s["actual_cache"])
                )
                assert set(s["methods"]) == names
                for n, v in s["methods"].items():
                    if "error" in v:
                        assert n == "inductor" and not v["private_exact"] and not v["public_exact"]
                        continue
                    assert len(v["projections"]) == len(v["rotated_qkv"]) == 3
                    assert v["private_exact"] == all(
                        not a["unequal_bits"] for a in v["projections"] + v["rotated_qkv"]
                    )
                    assert v["public_exact"] == (
                        v["private_exact"]
                        and v["independent_storage"]
                        and all(a["stride_match"] for a in v["projections"])
                    )
                    for a in v["projections"] + v["rotated_qkv"]:
                        if a["unequal_bits"]:
                            assert a["actual_bf16"] != a["reference_bf16"] and len(
                                a["first_index"]
                            ) == len(a["layout"]["shape"])
                m = s["methods"]
                assert (
                    m["three"]["public_exact"]
                    and not m["packed_views"]["independent_storage"]
                    and m["packed_copies"]["independent_storage"]
                )
                assert not m["late_bias"]["private_exact"]
                assert m["packed_copies"]["public_exact"] == m["packed_views"]["private_exact"]
                assert m["packed_views"]["private_exact"] != (
                    c["prefix"] == 127 and c["stage"] == "prefill"
                )
                assert m["q_plus_kv"]["private_exact"] != (
                    c["prefix"] in (127, 255) and c["stage"] == "prefill"
                )
            admitted = {n for n in names if all(s["methods"][n]["private_exact"] for s in samples)}
            assert set(c["profiles"]) == admitted and len(c["windows"]) == 7
            assert all(c["profiles"][n] == c["all_kernel_signatures"][n] for n in admitted)
            assert set(c["all_kernel_signatures"]) == {
                n for n in names if all("error" not in s["methods"][n] for s in samples)
            }
            for w in c["windows"]:
                assert len(w["order"]) == len(admitted) and set(w["order"]) == admitted
                for k, z in [("host", 15), ("device", 7)]:
                    assert set(w[k]) == admitted and all(
                        len(w[k][n]["samples_us"]) == z for n in admitted
                    )
            for k in ("host", "device"):
                assert c["median_us"][k] == {
                    n: median(median(w[k][n]["samples_us"]) for w in c["windows"]) for n in admitted
                }
