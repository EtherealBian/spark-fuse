"""Independently reconstruct original gates, counterfactual witnesses and shuffled medians."""

from collections import Counter
import gzip
from hashlib import sha256
import json
from math import isfinite, prod
from pathlib import Path
import re

import pytest
from test_phi3_native_evidence import samples
from tools.freeze_cast_cases import freeze
from tools.verify_sources import verify_archive, verify_reports

REPORTS = Path(__file__).resolve().parents[1] / "reports"
NAMES = {"eager", "inductor", "inductor_casts", "current", "selected"}
MODELS = {"rounded_rounded", "rounded_wide", "wide_rounded", "wide_wide"}


def test_fixed_catalog_exposes_missing_workers_and_reproduces_gates():
    path = REPORTS / "decomp-cast-catalog.json.gz"
    raw = path.read_bytes()
    catalog = json.loads(gzip.decompress(raw))
    z = catalog["cases"]
    assert catalog["index_lines"] == 773
    assert Counter(x["cohort"] for x in z) == {"retained_rejection": 17, "control": 3}
    assert len({x["id"] for x in z}) == len({x["line"] for x in z}) == len(z)
    assert {q["id"]: q["reason"] for q in catalog["excluded"]} == {
        "decomp-512x1536-bfloat16": "no retained worker",
        "decomp-48x8960-bfloat16": "discovery-error",
    }
    refs = {x["original_source"]["archive"]: x["original_source"]["source_tree_sha256"] for x in z}
    for arc, key in refs.items():
        assert arc == "sources/" + key + ".tar.gz" and verify_archive(REPORTS / arc, key) > 0
    paths = [REPORTS / ("decomp-cast-replay" + s + "-gb10.json") for s in ("", "-heldout")]
    assert verify_reports(paths)["distinct_archives"] == 1
    primary, heldout = (json.loads(p.read_bytes()) for p in paths)
    for a, b in zip(primary["cases"], heldout["cases"], strict=True):
        assert (
            a["id"] == b["id"]
            and a["cuda_sha256"] == b["cuda_sha256"]
            and a["seeds"][0] == b["seeds"][0]
        )
        assert len({tuple(x["inputs_sha256"]) for x in a["seeds"] + b["seeds"]}) == 5
    for report, offset in ((primary, 0), (heldout, 19609)):
        assert (
            report["seed_offset"],
            report["rtol"],
            report["atol"],
            report["catalog_sha256"],
        ) == (offset, 0.012, 0.0002, sha256(raw).hexdigest())
        assert (
            report["passed"]
            and not report["default_emulate_precision_casts"]
            and report["comparator_options"]
            == {"inductor": {}, "inductor_casts": {"emulate_precision_casts": True}}
        )
        for orig, case in zip(z, report["cases"], strict=True):
            cmp = orig["comparison"]
            assert orig["id"] == case["id"] and orig["cohort"] == case["cohort"]
            assert (
                re.fullmatch("[a-f0-9]{64}", orig["worker_sha256"])
                and case["original_worker_sha256"] == orig["worker_sha256"]
            )
            assert (
                case["selected_schedule"] == cmp["schedule"]
                and case["export_sha256"] == sha256(cmp["torch_export_graph"].encode()).hexdigest()
            )
            seed = cmp["seed"]
            assert seed == orig["job"]["seed"] + 2000 and [s["seed"] for s in case["seeds"]] == [
                seed,
                seed + 104729 + offset,
                seed + 130363 + offset,
            ]
            first = case["seeds"][0]["checks"]
            expected = {
                k: first[k]["passed"] == (cmp["backends"][old]["parity"] == "pass")
                for k, old in [
                    ("eager", "eager"),
                    ("selected", "sparkfuse"),
                    ("inductor", "inductor"),
                ]
            }
            assert case["original_admission_reproduced"] == expected and all(expected.values())
            bad = re.search(
                r"Mismatched elements: (\d+)", cmp["backends"]["inductor"].get("diagnostic", "")
            )
            assert first["inductor"]["outside_tolerance"] == (int(bad[1]) if bad else 0)
            for s in case["seeds"]:
                assert set(s["checks"]) == NAMES and len(set(s["inputs_sha256"])) == 3
                for v in s["checks"].values():
                    assert (
                        set(v["model_bit_mismatches"]) == MODELS
                        and v["bit_mismatches"] == v["model_bit_mismatches"]["rounded_rounded"]
                    )
                    assert (
                        0
                        <= v["outside_tolerance"]
                        <= v["bit_mismatches"]
                        <= v["elements"]
                        == prod(cmp["shape"])
                    )
                    assert (
                        v["passed"] == (v["outside_tolerance"] == 0) == ("first_outside" not in v)
                    )
                    if "first_outside" in v:
                        w = v["first_outside"]
                        assert all(
                            0 <= a < b for a, b in zip(w["index"], cmp["shape"], strict=True)
                        )
                        assert (
                            w["actual"]
                            != w["expected"]
                            == w["model_values"]["rounded_rounded"]["output"]
                        )
                        assert set(w["model_values"]) == MODELS and all(
                            isfinite(x)
                            for model in w["model_values"].values()
                            for x in model.values()
                        )
                assert s["checks"]["eager"]["bit_mismatches"] == 0
            allowed = {n for n in NAMES if all(s["checks"][n]["passed"] for s in case["seeds"])}
            assert (
                set(case["admitted"]) == allowed
                and {"current", "eager"} <= allowed
                and case["timing_seed"] == case["seeds"][-1]["seed"]
            )
            samples(case["timing"], allowed)
            for p in case["timing"]["profiles"].values():
                assert p["calls"] == 1 and p["count"] == sum(p["kernel_names"].values()) > 0


def test_freeze_never_invents_a_comparator_for_missing_evidence(tmp_path):
    key = "decomp-1x1-bfloat16"
    line = (json.dumps(dict(id=key, status="error", phase="broad-search")) + "\n").encode()
    (tmp_path / "index.jsonl").write_bytes(line)
    data = freeze(tmp_path, 1)
    assert data["cases"] == [] and data["excluded"] == [
        dict(id=key, line=1, reason="no retained worker")
    ]
    assert data["index_prefix_sha256"] == sha256(line).hexdigest()
    with pytest.raises(ValueError, match="incomplete"):
        freeze(tmp_path, 2)
    (tmp_path / "index.jsonl").write_bytes(line.rstrip())
    with pytest.raises(ValueError, match="incomplete"):
        freeze(tmp_path, 1)
