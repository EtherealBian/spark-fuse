"""The forensics and selection comparisons are reproducible without a CUDA driver."""

import hashlib
import json
from pathlib import Path
from statistics import median
from tools.verify_sources import verify_reports

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"


def test_shared_tree_panel():
    path = R / "mean-cohort-panel-gb10.json"
    d = json.loads(path.read_text())
    assert verify_reports([path])["verified_reports"] == 1
    assert len(d["cases"]) == 5
    for c in d["cases"]:
        trials = c["trials"]
        frozen = c["schedules"]
        assert len(trials) == 10
        for family in ("tree", "cohort"):
            subset = [t for t in trials if t["schedule"]["layout"] == "mean_" + family]
            assert len(subset) == (6 if family == "tree" else 4)
            best = min(median(t["device"]["samples_us"]) for t in subset)
            assert any(
                t["schedule"] == frozen[family + "_fastest"]
                and median(t["device"]["samples_us"]) == best
                for t in subset
            )
        assert len(c["heldout"]) == 2
        for i, item in enumerate(c["heldout"]):
            assert item["offsets"] == ([1, 2, 3] if i else [0, 0, 0])
            assert item["seed"] == c["seed"] + 101
            assert set(item["tapped"]) == set(frozen)
            assert all(
                len(taps) == 6 and all(q["bit_mismatches"] == 0 for q in taps)
                for taps in item["tapped"].values()
            )
            ok = {k for k, v in item["checks"].items() if v["passed"]}
            assert (
                len(item["windows"]) == 7 and set(item["host_sync"]) == set(item["median_us"]) == ok
            )
            for w in item["windows"]:
                assert len(w["order"]) == len(ok) and set(w["order"]) == set(w["device"]) == ok
            assert {
                k: median(median(w["device"][k]["samples_us"]) for w in item["windows"]) for k in ok
            } == item["median_us"]
            assert all(item["checks"][k]["bits"][0]["bit_mismatches"] == 0 for k in frozen)
    binaries = json.loads((R / "mean-cohort-binaries-gb10.json").read_text())
    assert binaries["panel_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    for key, v in binaries["layouts"].items():
        c = d["cases"][0]
        assert any(
            x["kernel"] == v["kernel"] and x["schedule"] == c["schedules"][key] == v["schedule"]
            for x in c["trials"]
        )
        src = ROOT / "examples/generated/mean-cohort" / (v["schedule"]["layout"] + ".cu")
        assert (
            src.relative_to(ROOT).as_posix() == v["source"]
            and hashlib.sha256(src.read_bytes()).hexdigest() == v["source_sha256"]
        )
        assert "sm_121" in v["resource_dump"] and "REG:" in v["resource_dump"]


def test_independent_cohort_worker():
    import gzip

    p = R / "mean-cohort-replay.json.gz"
    assert verify_reports([p])["verified_reports"] == 1
    record = json.loads(gzip.decompress(p.read_bytes()))
    q = record["tuning"]
    c = record["comparison"]
    assert record["status"] == "ok" and q["seed"] == 64333 and c["seed"] == 66333
    assert len(q["trials"]) == 10 and not q["invalid"] and q["heldout_errors"][0]["unequal"] == 0
    assert (
        all(b["parity"] == "pass" for b in c["backends"].values())
        and c["backends"]["sparkfuse"]["errors"][0]["unequal"] == 0
    )
    assert c["schedule"] == q["selected"]
