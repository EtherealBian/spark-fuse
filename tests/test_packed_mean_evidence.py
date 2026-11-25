"""Recreate exact admission, frozen selection and all reported medians from archived samples."""

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from statistics import median
from test_phi3_native_evidence import samples
from tools.verify_sources import verify_reports

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SPARK = {"previous_best", "packed_best", "default"}
METHODS = SPARK | {"eager", "inductor", "inductor_casts"}


def device_windows(ws, count, expected, m):
    assert len(ws) == count and set(m) == expected
    for w in ws:
        assert len(w["order"]) == len(expected) and set(w["order"]) == expected == set(w["device"])
        for t in w["device"].values():
            z = t["samples_us"]
            assert (
                len(z) == 7
                and all(isfinite(v) and v > 0 for v in z)
                and t["median_us"] == median(z)
            )
    assert m == {k: median(median(w["device"][k]["samples_us"]) for w in ws) for k in expected}


def test_primary_frozen_replay_and_gates():
    p, q = [REPORTS / f"packed-mean-{label}-gb10.json" for label in ("discovery", "heldout")]
    assert verify_reports([p, q])["distinct_archives"] == 1
    first, second = map(json.loads, (p.read_bytes(), q.read_bytes()))
    assert (
        first["selection_sha256"] is None
        and second["selection_sha256"] == sha256(p.read_bytes()).hexdigest()
        and second["seed_offset"] == 21061
    )
    assert first["case_indices"] == second["case_indices"] == list(range(9))
    assert not first["verify_only"] and not second["verify_only"]
    fingerprints = set()
    for orig, repeat in zip(first["cases"], second["cases"], strict=True):
        assert (
            orig["id"] == repeat["id"]
            and orig["export_sha256"] == repeat["export_sha256"]
            and len(orig["alignments"]) == 2
        )
        assert all(orig["kernels"][key] == value for key, value in repeat["kernels"].items())
        prior = {
            k
            for k, v in orig["kernels"].items()
            if v["schedule"]["layout"] in ("mean_cohort", "mean_tree")
        }
        tile = set(orig["kernels"]) - prior
        n, h = orig["shape"]
        assert len(prior) == 10 and len(tile) == 3 * sum(h % v == 0 for v in (2, 4, 8))
        for i, (a, b) in enumerate(zip(orig["alignments"], repeat["alignments"], strict=True)):
            assert a["offsets"] == b["offsets"] == ([1, 2, 3] if i else [0, 0, 0])
            assert a["selected"] == b["selected"]
            assert not b["calibration_windows"] and b["calibration_seed"] is None
            assert set(a["calibration_checks"]) == prior | tile and all(
                d["bit_mismatches"] == 0
                for checks in a["calibration_checks"].values()
                for d in checks
            )
            med = a["calibration_medians"]
            device_windows(a["calibration_windows"], 3, prior | tile, med)
            assert a["selected"] == {
                k: min(sorted(group), key=med.__getitem__)
                for k, group in [("previous_best", prior), ("packed_best", tile)]
            }
            for z in (a, b):
                assert len(z["holdout"]) == 3
                for row in z["holdout"]:
                    assert row["seed"] not in (a["calibration_seed"], orig["export_seed"])
                    assert set(row["checks"]) == METHODS
                    f = tuple(row["input_sha256"])
                    assert len(f) == 3 and len(set(f)) == 3 and f not in fingerprints
                    fingerprints.add(f)
                    assert len(row["output_sha256"]) == 1 + orig["visible"]
                    for cs in row["checks"].values():
                        assert len(cs) == 1 + orig["visible"] and all(
                            0
                            <= x["outside_tolerance"]
                            <= x["bit_mismatches"]
                            <= x["elements"]
                            == n * h
                            for x in cs
                        )
                admitted = {
                    k
                    for k in METHODS
                    if all(c["bit_mismatches"] == 0 for x in z["holdout"] for c in x["checks"][k])
                }
                assert admitted == set(z["admitted"]) and SPARK | {"eager"} <= admitted
                samples(z["fresh"], admitted)
                assert all(z["fresh"]["profiles"][k]["count"] == 1 for k in SPARK)
                device_windows(z["prepared_windows"], 7, SPARK, z["prepared_medians"])
    assert len(fingerprints) == 108


def test_binary_resources_and_emitted_sources():
    p = REPORTS / "packed-mean-discovery-gb10.json"
    d = json.loads(p.read_bytes())
    s = json.loads((REPORTS / "packed-mean-sass-gb10.json").read_text())
    assert s["panel_sha256"] == sha256(p.read_bytes()).hexdigest()
    by = {(v["id"], tuple(v["offsets"])): v for v in s["cases"]}
    assert len(by) == len(s["cases"]) == 18
    for i, case in enumerate(d["cases"]):
        for z in case["alignments"]:
            rec = by[case["id"], tuple(z["offsets"])]
            for role, name in z["selected"].items():
                spec = case["kernels"][name]
                key = rec["selected"][role]
                assert key == spec["cache_id"] and re.fullmatch("[a-f0-9]{64}", key)
                binary = s["binaries"][key]
                assert all(binary[k] == v for k, v in spec.items())
                assert binary["resources"].keys() == binary["static_instruction_sites"].keys()
                if role == "packed_best":
                    h = case["shape"][1]
                    assert len(binary["resources"]) == 2
                    assert all(
                        # Frozen sm_121 code objects report 1024 bytes above the two C++ arrays.
                        r["static_shared_bytes"] == 2 * h + 128 + 1024 and r["registers"] > 0
                        for r in binary["resources"].values()
                    )
                    assert all(binary["static_instruction_sites"].values())
                    if i in (0, 5):
                        path = (
                            ROOT / "examples/generated/packed-mean" / (spec["cuda_sha256"] + ".cu")
                        )
                        assert sha256(path.read_bytes()).hexdigest() == spec["cuda_sha256"]
