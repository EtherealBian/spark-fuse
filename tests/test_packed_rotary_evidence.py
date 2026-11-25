"""Verify the published same-graph ablation, numerical admission, traces and source hashes."""

from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from tools.verify_sources import verify_reports

ROOT = Path(__file__).resolve().parents[1]


def timings(ws, profile, med, allowed):
    assert set(profile) == allowed and len(ws) == 7
    for w in ws:
        assert len(w["order"]) == len(allowed) and set(w["order"]) == allowed
        for k, n in [("device", 7), ("host", 15)]:
            assert set(w[k]) == allowed and all(len(w[k][v]["samples_us"]) == n for v in allowed)
    for k in ("device", "host"):
        assert med[k] == {n: median(median(w[k][n]["samples_us"]) for w in ws) for n in allowed}


def graphs(gs):
    for name in ("two_regions", "joint"):
        rs = gs[name]["records"]
        kinds = [r.get("kind") for r in rs]
        for k in ("split_copy3", "rotary_pair"):
            assert kinds.count(k) == int(name == "two_regions")
        assert kinds.count("packed_rotary_copy") == int(name == "joint")
    (hit,) = [r for r in gs["joint"]["records"] if r.get("kind") == "packed_rotary_copy"]
    key = hit["cuda_sha256"]
    assert (
        key
        == sha256(
            (ROOT / "examples/generated/packed-rotary" / (key + ".cu")).read_bytes()
        ).hexdigest()
    )
    assert hit["widths"] == [1536, 256, 256] and hit["head_dim"] == 128
    return key


def test_packed_rotary_evidence():
    paths = [ROOT / "reports" / ("qwen-packed-rotary" + s + "-gb10.json") for s in ("", "-heldout")]
    assert verify_reports(paths)["verified_reports"] == 2
    rep = [json.loads(p.read_text()) for p in paths]
    assert (
        len({r["weight_seed"] for r in rep})
        == len({tuple(r["checkpoint"]["packed_sha256"]) for r in rep})
        == 2
    )
    for r in rep:
        seeds = r["layer_seeds"]
        assert (
            len(set(seeds)) == 5
            and r["export_seed"] not in seeds
            and r["passed"]
            and not r["verify_only"]
        )
        assert [c["prefix"] for c in r["cases"]] == [127, 1024]
        for c in r["cases"]:
            iso = c["isolated"]
            assert [i["stage"] for i in iso] == ["prefill", "decode1", "decode2"]
            hashes = []
            for i in iso:
                hashes.append(graphs(i["graphs"]))
                assert len(i["rows"]) == len({x["input_phase_sha256"][0] for x in i["rows"]}) == 3
                for row in i["rows"]:
                    assert (
                        set(row["methods"]) == {"eager", "two_regions", "joint", "inductor"}
                        and row["seed"] != r["export_seed"]
                    )
                    assert len(row["upstream_wide_projection_against_hf"]) == 3
                    for n, m in row["methods"].items():
                        if "error" in m:
                            assert n == "inductor" and not m["passed"]
                            continue
                        assert len(m["tensors"]) == 3 and m["passed"] == (
                            m["fresh"]
                            and m["inputs_unchanged"]
                            and all(
                                d["unequal_bits"] == 0
                                and d["stride_match"]
                                and d["layout"]["storage_offset"] == 0
                                for d in m["tensors"]
                            )
                        )
                        assert n == "inductor" or m["passed"]
                if i["stage"] == "decode2":
                    assert i["timing_seed"] is None and "windows" not in i
                    continue
                allowed = {
                    n for n in row["methods"] if all(x["methods"][n]["passed"] for x in i["rows"])
                }
                assert i["timing_seed"] == i["rows"][-1]["seed"]
                timings(i["windows"], i["profiles"], i["median_us"], allowed)
                assert [i["profiles"][n]["count"] for n in ("two_regions", "joint")] == [2, 1]
            if c["prefix"] == 127:
                assert any(
                    d["unequal_bits"]
                    for d in iso[0]["rows"][0]["upstream_wide_projection_against_hf"]
                )
            b = c["backends"]
            assert set(b) == {"eager", "direct", "two_regions", "joint", "inductor"}
            for n, p in b.items():
                assert [t["seed"] for t in p["trajectories"]] == seeds
                for traj in p["trajectories"]:
                    for st in traj["steps"]:
                        if "error" in st:
                            assert n == "inductor" and not st["passed"]
                            continue
                        outs = st["outputs"]
                        assert set(outs) == {"hidden", "probabilities", "key", "value"}
                        assert st["passed"] == (
                            st["new_storage"]
                            and st["input_unchanged"]
                            and st["prefix_exact"]
                            and st["masked_nonzero"] == 0
                            and st["max_row_sum_error"] <= 0.004
                            and (n == "inductor" or st["output_strides"] == st["expected_strides"])
                            and all(
                                o["tolerance_pass"] and (n == "inductor" or o["unequal_bits"] == 0)
                                for o in outs.values()
                            )
                        )
                    assert traj["passed"] == (
                        len(traj["steps"]) == 2 and all(st["passed"] for st in traj["steps"])
                    )
                assert p["passed"] == all(t["passed"] for t in p["trajectories"])
                assert n == "inductor" or p["passed"]
            allowed = {n for n, p in b.items() if p["passed"]}
            assert len(c["graphs"]) == 2
            for k, g in enumerate(c["graphs"]):
                assert graphs(g["variants"]) == hashes[k + 1]
                assert "aten.linear" in g["packed"]
                ws, pr, me = c["windows"][k], c["profiles"][k], c["median_us"][k]
                timings(ws, pr, me, allowed)
                assert pr["two_regions"]["count"] - pr["joint"]["count"] == 1
