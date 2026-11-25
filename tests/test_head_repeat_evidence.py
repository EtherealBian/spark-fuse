"""Replay the numerical admission decisions, launch counts and timing summaries offline."""

from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from tools.verify_sources import verify_reports

ROOT = Path(__file__).resolve().parents[1]


def load(stem):
    p = ROOT / "reports" / (stem + "-gb10.json")
    assert verify_reports([p])["verified_reports"] == 1
    return json.loads(p.read_text())


def test_group_as_row_numerical_rejection():
    d = load("grouped-kv-admission")
    cases = d["cases"]
    weights = {x["weight_seed"] for x in cases}
    prefixes = {31, 32, 62, 63, 254, 255, 256, 1023, 1024}
    assert len(weights) == 2 and len(cases) == 18
    cohorts = []
    ops = []
    for w in weights:
        chosen = [x for x in cases if x["weight_seed"] == w]
        assert {x["prefix"] for x in chosen} == prefixes
        seeds = {s["seed"] for s in chosen[0]["samples"]}
        assert len(seeds) == 3
        cohorts.append(seeds)
        for x in chosen:
            assert (
                x["keys"] == x["prefix"] + 1
                and x["export_seed"] not in seeds
                and {s["seed"] for s in x["samples"]} == seeds
            )
            for s in x["samples"]:
                assert all(s["controls"].values())
                for k in ("QK", "PV"):
                    op = s[k]
                    ops.append(op)
                    assert op["finite"] and len(op["examples"]) == min(4, op["unequal_bits"])
                    for ex in op["examples"]:
                        assert int(ex["eager_hex"], 16) != int(ex["grouped_hex"], 16) and all(
                            0 <= i < n for i, n in zip(ex["index"], op["shape"], strict=True)
                        )
    assert cohorts[0].isdisjoint(cohorts[1])
    summary = d["summary"]
    assert summary == dict(
        tensors=len(ops),
        tensors_with_bit_mismatch=sum(op["unequal_bits"] > 0 for op in ops),
        unequal_elements=sum(op["unequal_bits"] for op in ops),
        tolerance_failures=sum(not op["tolerance_pass"] for op in ops),
        max_abs=max(op["max_abs"] for op in ops),
        sampled_bit_exact=False,
    )
    assert (
        d["decision"] == "rejected_bit_exact"
        and summary["tolerance_failures"] == 0
        and summary["unequal_elements"] > 0
    )


def test_head_repeat_panel_is_raw_copy_and_not_zero_copy():
    d = load("head-repeat-panel")
    assert d["passed"] and len(d["cases"]) == 4
    for c in d["cases"]:
        g = c["record"]["logical_shape"][2]
        assert 1 < g <= 32 and c["materialized_output_bytes"] == g * c["input_bytes"]
        for r in c["layouts"]:
            keys = set(r["samples"])
            assert keys == {"eager", "inductor", "sparkfuse"}
            for ss in r["samples"].values():
                assert [s["seed"] for s in ss] == d["seeds"] and all(
                    s["new_storage"] and all(s["matching_strides"]) and s["unequal_bits"] == [0, 0]
                    for s in ss
                )
            assert len(r["windows"]) == 7 and all(
                len(w["order"]) == len(keys) and set(w["order"]) == keys for w in r["windows"]
            )
            for metric in ("device", "host"):
                assert r["median_us"][metric] == {
                    k: median(median(w[metric][k]["samples_us"]) for w in r["windows"])
                    for k in keys
                }
            prof = r["profiles"]
            assert (prof["eager"]["count"], prof["sparkfuse"]["count"]) == (2, 1)
            width = c["record"]["schedule"]["max_packet_elements"] if r["offsets"] == [0, 0] else 1
            assert f"paired_head_repeat<{width}," in next(iter(prof["sparkfuse"]["kernel_names"]))


def test_head_repeat_long_decode_boundary_recovers_original_counterexample():
    old = load("qwen-decode-head-repeat-initial")
    new = load("qwen-decode-head-repeat-boundary")
    assert not old["passed"] and new["passed"] and new["framework_softmax"]
    assert (old["weight_seed"], old["export_seed"], old["seeds"]) == (
        new["weight_seed"],
        new["export_seed"],
        new["seeds"],
    )
    assert (
        [x["initial_prefix"] for x in old["cases"]]
        == [31, 254, 1024]
        == [x["initial_prefix"] for x in new["cases"]]
    )
    last = old["cases"][-1]
    for name in ("rotary", "append", "repeat"):
        v = last["backends"][name]
        assert not v["passed"] and "timing" not in v
        assert any(
            s["outputs"]["probabilities"]["unequal_bits"]
            for t in v["trajectories"]
            for s in t["steps"]
        )
    for c in new["cases"]:
        assert c["passed"] and all(
            x["erased_hidden_bits"] > 0
            and x["masked_prob_exact"]
            and x["masked_hidden_exact"]
            and x["erased_prob_exact"]
            for x in c["controls"]
        )
        assert not c["backends"]["inductor"]["passed"] and "timing" not in c["backends"]["inductor"]
        for i, graph in enumerate(c["graphs"]):
            records = graph["partitions"]["repeat"]["records"]
            (hit,) = [r for r in records if r.get("kind") == "paired_head_repeat"]
            assert len([r for r in records if r.get("reason") == "framework_softmax"]) == 1
            assert hit["logical_shape"] == [2, 2, 6, c["initial_prefix"] + 1 + i, 128]
            assert (
                hit["cuda_sha256"]
                == sha256(
                    (
                        ROOT
                        / f"examples/generated/head-repeat/p{c['initial_prefix']}-step{i + 1}.cu"
                    ).read_bytes()
                ).hexdigest()
            )
        for k in ("rotary", "append", "repeat", "exported_aten", "eager"):
            rr = c["backends"][k]
            assert rr["passed"] and [t["seed"] for t in rr["trajectories"]] == new["seeds"]
            for t in rr["trajectories"]:
                assert t["passed"] and len(t["steps"]) == 2
                for st in t["steps"]:
                    assert (
                        st["passed"]
                        and st["prefix_exact"]
                        and st["new_storage"]
                        and st["input_unchanged"]
                        and st["masked_nonzero"] == 0
                        and st["output_strides"] == st["expected_strides"]
                        and all(m["unequal_bits"] == 0 for m in st["outputs"].values())
                    )
            for i, t in enumerate(rr["timing"]):
                assert len(t["windows"]) == 7 and all(k in o for o in c["orderings"][i])
                assert all(
                    t[m + "_us"] == median(median(w[m]["samples_us"]) for w in t["windows"])
                    for m in ("host", "device")
                )
        for a, b in zip(
            c["backends"]["append"]["timing"], c["backends"]["repeat"]["timing"], strict=True
        ):
            assert (
                b["profile"]["count"] == a["profile"]["count"] - 1
                and sum(
                    v for n, v in b["profile"]["kernel_names"].items() if "paired_head_repeat<" in n
                )
                == 1
            )
