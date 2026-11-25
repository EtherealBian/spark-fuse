"""Verify published score policy decisions, numerical counterexamples and raw window medians."""

from hashlib import sha256
import json
from pathlib import Path
from statistics import median

from tools.verify_sources import verify_reports

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    path = ROOT / "reports" / (name + "-gb10.json")
    assert verify_reports([path])["verified_reports"] == 1
    return json.loads(path.read_text())


def test_half_input_origin_and_negative_control():
    initial = read("ordered-score-slots-initial")
    fixed = read("ordered-score-slots")
    origin = read("ordered-score-half-origin")
    assert not initial["passed"] and fixed["passed"] and origin["passed"]
    assert initial["seeds"] == fixed["seeds"] and set(initial["seeds"]) < set(origin["seeds"])
    assert initial["export_seed"] == fixed["export_seed"] == origin["export_seed"]
    assert set(origin["seeds"]).isdisjoint([origin["export_seed"]])
    prior = next(c for c in initial["cases"] if c["label"] == "edge")
    assert prior["width"] == 2048
    prior_counts = {}
    for layout in prior["layouts"]:
        assert not layout["passed"] and set(layout["median_us"]["device"]) == {"eager"}
        for name in ("slots", "loop"):
            for s in layout["samples"][name]:
                low, masked, fp32 = s["unequal_bits"]
                assert masked == 0 and fp32 > 0 and not s["passed"]
                prior_counts[name, tuple(layout["offset_elements"]), s["seed"]] = (low, fp32)
    assert [c["width"] for c in origin["cases"]] == [1023, 1024, 1025, 1537, 2047, 2048]
    for c in origin["cases"]:
        w = c["width"]
        isp = c["profiles"]["implicit_half_to_float"]
        esp = c["profiles"]["on_explicit_float_input"]
        assert isp["count"] == esp["count"] == 1 and "softmax_warp_forward" in next(
            iter(esp["kernel_names"])
        )
        assert ("cunn_SoftMaxForward" in next(iter(isp["kernel_names"]))) == (w > 1024)
        assert {(s["seed"], tuple(s["offset_elements"])) for s in c["samples"]} == {
            (sd, o) for sd in origin["seeds"] for o in ((0, 0), (1, 3))
        }
        for s in c["samples"]:
            n = s["implicit_vs_explicit_fp32_bits"]
            assert s["fused_vs_explicit_bits"] == [0] * 4 and s["fused_vs_implicit_fp32_bits"] == n
            assert (n > 0) == (w > 1024) and len(s["examples"]) == min(4, n)
            for ex in s["examples"]:
                row, col = ex["index"]
                assert 0 <= row < 24 and 0 <= col < w
                assert int(ex["implicit_fp32_hex"], 16) != int(ex["explicit_fp32_hex"], 16)
            if w == 2048 and s["seed"] in initial["seeds"]:
                for name in ("slots", "loop"):
                    assert (s["implicit_vs_explicit_half_bits"], n) == prior_counts[
                        name, tuple(s["offset_elements"]), s["seed"]
                    ]


def test_ordered_slot_panel_resources_and_admission():
    d = read("ordered-score-slots")
    assert len(d["seeds"]) == 5 and d["timing_seed"] == d["seeds"][-1]
    assert {c["label"] for c in d["cases"]} == {
        "decode",
        "visible",
        "edge",
        "many",
        "float",
        "explicit",
    }
    for c in d["cases"]:
        edge = c["label"] == "edge"
        builds = c["builds"]
        for name, b in builds.items():
            if "rejected" in b:
                assert edge and name in ("slots", "loop") and "keep ATen" in b["rejected"]
                continue
            src = ROOT / f"examples/generated/ordered-score-panel/{c['label']}-{name}.cu"
            assert (
                sha256(src.read_bytes()).hexdigest() == b["source_sha256"]
                and len(b["binary_sha256"]) == 64
            )
        if edge:
            assert any(
                r.get("half_to_float") and r["reason"] == "outside_ordered_score_domain"
                for r in c["decisions"]
            )
        else:
            assert (
                builds["slots"]["resources"]["stack_bytes"]
                == 0
                < builds["loop"]["resources"]["stack_bytes"]
            )
            assert (
                builds["slots"]["resources"]["registers"] > builds["loop"]["resources"]["registers"]
            )
            for name in ("loop", "slots"):
                assert builds[name]["static_instruction_sites"]["SHFL.BFLY"] > 0
        assert [item["offset_elements"] for item in c["layouts"]] == [[0, 0], [1, 3]]
        for lay in c["layouts"]:
            for ss in lay["samples"].values():
                assert [s["seed"] for s in ss] == d["seeds"]
                for s in ss:
                    if "error" not in s:
                        assert s["passed"] == (
                            not any(s["unequal_bits"]) and all(s["matching_strides"])
                        )
            eligible = {k for k, ss in lay["samples"].items() if all(s["passed"] for s in ss)}
            assert (
                lay["passed"]
                and set(c["required"]) <= eligible
                and set(lay["profiles"]) == eligible
            )
            assert len(lay["windows"]) == 7 and all(
                len(w["order"]) == len(eligible) and set(w["order"]) == eligible
                for w in lay["windows"]
            )
            for metric in ("host", "device"):
                assert lay["median_us"][metric] == {
                    k: median(median(w[metric][k]["samples_us"]) for w in lay["windows"])
                    for k in eligible
                }
    for r in d["order_witness"]:
        assert [s["seed"] for s in r["samples"]] == d["seeds"]
        assert all((s["fp32_unequal_bits"] > 0) == (r["width"] > 2048) for s in r["samples"])
        assert (
            "softmax_warp_forward" in next(iter(r["framework_softmax_profile"]["kernel_names"]))
        ) == (r["width"] <= 2048)


def test_qwen_ordered_score_sources_and_stateful_replay():
    main = read("qwen-decode-ordered-scores-qualified")
    replay = read("qwen-ordered-score-replay")
    old = read("qwen-decode-head-repeat-initial")
    assert (
        not old["passed"]
        and replay["weight_seed"] == old["weight_seed"]
        and replay["seeds"] == old["seeds"]
    )
    assert (
        set(main["seeds"]).isdisjoint(replay["seeds"])
        and main["weight_seed"] != replay["weight_seed"]
    )
    for doc, prefixes in [(main, [1024, 1535, 2047]), (replay, [1024, 2046])]:
        assert (
            doc["score_panel"]
            and doc["passed"]
            and [c["initial_prefix"] for c in doc["cases"]] == prefixes
        )
        for c in doc["cases"]:
            for i, g in enumerate(c["graphs"]):
                width = c["initial_prefix"] + i + 1
                for name in ("framework", "loop", "slots"):
                    rs = g["partitions"][name]["records"]
                    hit = [r for r in rs if r.get("module") and "softmax" in r.get("ops", [])]
                    assert any(r.get("kind") == "paired_head_repeat" for r in rs) and any(
                        r.get("kind") == "paired_append" for r in rs
                    )
                    if name == "framework" or width > 2048:
                        reason = (
                            "framework_softmax"
                            if name == "framework"
                            else "outside_ordered_score_domain"
                        )
                        assert not hit and sum(r.get("reason") == reason for r in rs) == 1
                    else:
                        (rec,) = hit
                        assert rec["schedule"]["layout"] == (
                            "warp_slots" if name == "slots" else "warp_xor"
                        )
                        if doc is main:
                            assert (
                                sha256(
                                    (
                                        ROOT
                                        / f"examples/generated/ordered-scores/p{c['initial_prefix']}-step{i + 1}-{name}.cu"
                                    ).read_bytes()
                                ).hexdigest()
                                == rec["cuda_sha256"]
                            )
            for name, b in c["backends"].items():
                if name == "inductor":
                    assert not b["passed"] and "timing" not in b
                    continue
                assert b["passed"] and [t["seed"] for t in b["trajectories"]] == doc["seeds"]
                for t in b["trajectories"]:
                    assert len(t["steps"]) == 2
                    for s in t["steps"]:
                        assert (
                            s["passed"]
                            and s["new_storage"]
                            and s["prefix_exact"]
                            and s["input_unchanged"]
                            and s["masked_nonzero"] == 0
                        )
                        assert (
                            s["output_strides"] == s["expected_strides"]
                            and set(s["outputs"]) == {"hidden", "probabilities", "key", "value"}
                            and all(o["unequal_bits"] == 0 for o in s["outputs"].values())
                        )
                if doc is main:
                    for i, tm in enumerate(b["timing"]):
                        assert len(tm["windows"]) == 7 and all(name in o for o in c["orderings"][i])
                        for metric in ("host", "device"):
                            assert tm[metric + "_us"] == median(
                                median(w[metric]["samples_us"]) for w in tm["windows"]
                            )
                        if name in ("slots", "loop", "framework"):
                            assert tm["profile"]["count"] == (
                                20
                                if name == "framework" or c["initial_prefix"] + i + 1 > 2048
                                else 17
                            )
