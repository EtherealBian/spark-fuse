"""Offline evidence: compare actual graph records, not just the benchmark's pass flags."""

import json
from hashlib import sha256
from pathlib import Path

from test_packed_mean_evidence import device_windows
from test_phi3_native_evidence import samples
from tools.verify_sources import verify_reports

ROOT = Path(__file__).resolve().parents[1]
NATIVE = {"default", "previous", "packed"}
ISO = NATIVE | {"eager", "inductor", "inductor_casts"}
WHOLE = ISO | {"aten"}


def exact(values):
    return all(
        z["unequal_bits"] == 0 and z["finite"] and z["tolerance_pass"] for z in values.values()
    )


def test_real_layer_selection_and_independent_replay():
    paths = [ROOT / "reports" / f"qwen-packed-mean-{n}-gb10.json" for n in ("discovery", "heldout")]
    assert verify_reports(paths)["distinct_archives"] == 1
    a, b = [json.loads(p.read_bytes()) for p in paths]
    assert (
        a["selection_sha256"] is None
        and b["selection_sha256"] == sha256(paths[0].read_bytes()).hexdigest()
    )
    assert (a["seed_offset"], b["seed_offset"]) == (0, 98627) and a["sequences"] == b[
        "sequences"
    ] == [33, 65, 129]
    assert not a["verify_only"] and not b["verify_only"]
    main_inputs = set()
    live_inputs = set()
    all_seeds = set()
    for orig, rep in zip(a["cases"], b["cases"], strict=True):
        for k in (
            "export_sha256",
            "region",
            "ir",
            "selected",
            "kernels",
            "partition_records",
            "unchanged_records_sha256",
            "variants",
            "mixed_fx",
            "aten_graph",
        ):
            assert orig[k] == rep[k], k
        assert (
            orig["weight_seed"] != rep["weight_seed"]
            and orig["weight_sha256"].keys() == rep["weight_sha256"].keys()
            and all(v != rep["weight_sha256"][k] for k, v in orig["weight_sha256"].items())
        )
        assert len(orig["weight_sha256"]) >= 11
        cal = orig["calibration"]
        assert rep["calibration"] is None
        med = cal["medians"]
        tile = {k for k in med if k.startswith("mean_packed_tile-")}
        old = set(med) - tile
        assert (
            len(old) == 10
            and len(tile) == 9
            and set(cal["values"]) == set(med)
            and all(exact(v) for v in cal["values"].values())
        )
        device_windows(cal["windows"], 3, old | tile, med)
        assert orig["selected"] == {
            name: min(sorted(pool), key=med.__getitem__)
            for name, pool in [("previous", old), ("packed", tile)]
        }
        assert cal["seed"] not in all_seeds
        all_seeds.add(cal["seed"])
        for c in (orig, rep):
            hit = c["region"]
            target = hit["module"]
            assert (
                hit["split"]["reason"] == "fallback_barrier"
                and hit["live_ins"] == ["layer_post_attention_layernorm_weight", "x", "linear_3"]
                and hit["live_outs"] == ["add_4", "mul_8"]
            )
            assert c["ir"]["domain"] == [2, c["sequence"], 1536] and len(c["ir"]["output_ids"]) == 2
            assert sha256(c["aten_graph"].encode()).hexdigest() == c["export_sha256"]
            for name, variant in c["variants"].items():
                assert sha256(c["mixed_fx"].encode()).hexdigest() == variant["fx_sha256"]
                others = []
                for left, right in zip(c["partition_records"], variant["records"], strict=True):
                    if left.get("module") == target:
                        assert dict(left, schedule=c["kernels"][name]["schedule"]) == right
                        others.append({k: v for k, v in left.items() if k != "schedule"})
                    else:
                        assert left == right
                        others.append(left)
                assert (
                    sha256(json.dumps(others, sort_keys=True).encode()).hexdigest()
                    == c["unchanged_records_sha256"][name]
                )
            assert len(c["trials"]) == 3
            for t in c["trials"]:
                assert t["seed"] != c["export_seed"] and t["seed"] not in all_seeds
                all_seeds.add(t["seed"])
                # cos/sin and masks repeat by design; x must change. Also require a new physical o_proj output.
                x = t["model_input_sha256"][0]
                o = t["live_input_sha256"][2]
                assert (
                    x == t["live_input_sha256"][1] and x not in main_inputs and o not in live_inputs
                )
                assert (
                    t["live_input_sha256"][0]
                    == c["weight_sha256"]["layer.post_attention_layernorm.weight"]
                )
                main_inputs.add(x)
                live_inputs.add(o)
                assert (
                    set(t["isolated"]) == ISO
                    and set(t["whole"]) == WHOLE
                    and set(t["wires"]) == NATIVE
                )
                assert all(
                    set(v) == {"residual", "normalized"}
                    for v in [*t["isolated"].values(), *t["wires"].values()]
                )
                assert all(exact(v) for v in t["wires"].values())
                for result in t["whole"].values():
                    v = result["values"]
                    assert (
                        set(v) == {"hidden", "probabilities"}
                        and v["hidden"]["shape"] == [2, c["sequence"], 1536]
                        and v["probabilities"]["shape"] == [2, 12, c["sequence"], c["sequence"]]
                    )
                    assert result["passed"] == (
                        exact(v)
                        and result["masked_nonzero"] == 0
                        and result["max_row_sum_error"] <= 0.004
                    )
            for label, methods in [("whole", WHOLE), ("isolated", ISO)]:
                admitted = {
                    k
                    for k in methods
                    if all(
                        t[label][k]["passed"] if label == "whole" else exact(t[label][k])
                        for t in c["trials"]
                    )
                }
                assert NATIVE | {"eager"} <= admitted and set(c["allowed"][label]) == admitted
                samples(c[label], admitted)
                prof = c[label]["profiles"]
                assert prof["previous"]["count"] == prof["packed"]["count"]
                if label == "isolated":
                    assert prof["packed"]["count"] == 1
                else:
                    assert prof["packed"]["count"] > 1
    assert len(main_inputs) == len(live_inputs) == 18
