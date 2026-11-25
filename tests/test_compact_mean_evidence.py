"""Recompute the published experiment from its samples without importing torch."""

import gzip
import json
from pathlib import Path
from statistics import median

import pytest
from tools.campaign_snapshot import check_tuning, checked_median
from tools.verify_sources import verify_reports


REPORTS = Path(__file__).resolve().parents[1] / "reports"
PANEL = REPORTS / "compact-mean-panel.json"
REPLAY = REPORTS / "compact-mean-replay.json.gz"
FAMILIES = ("block", "compact_mean")


def test_compact_report_sources_and_replay():
    assert verify_reports([PANEL, REPLAY])["verified_reports"] == 2
    replay = json.loads(gzip.decompress(REPLAY.read_bytes()))
    job, result, tune = (replay[k] for k in ("job", "comparison", "tuning"))
    assert replay["status"] == "ok" and job["case"] == result["case"] == tune["case"] == "mean"
    assert tune["compiler"] == json.loads(PANEL.read_bytes())["compiler"]
    assert (
        result["shape"] == [job["rows"], job["width"]]
        and result["dtype"] == "torch." + job["dtype"]
    )
    assert result["seed"] == job["seed"] + 2000 and tune["seed"] == job["seed"]
    check_tuning(tune, job)
    assert (
        result["schedule"] == tune["selected"]
        and len(tune["trials"]) + len(tune["invalid"]) == job["budget"]
    )
    assert set(result["backends"]) == {"eager", "inductor", "sparkfuse"}
    for backend in result["backends"].values():
        assert backend["parity"] == "pass" and backend["errors"][0]["finite_pairs"] == job["rows"]
        for clock in ("device", "host_sync"):
            checked_median(backend[clock])


@pytest.mark.parametrize("idx", range(4))
def test_compact_paired_panel(idx):
    cases = json.loads(PANEL.read_bytes())["cases"]
    assert len(cases) == len({(c["rows"], c["width"], c["dtype"]) for c in cases}) == 4
    c = cases[idx]
    grids = []
    for family in FAMILIES:
        trials = [t for t in c["trials"] if t["schedule"]["layout"] == family]
        grids.append({(t["schedule"]["threads"], t["schedule"]["vec"]) for t in trials})
        assert len(grids[-1]) == len(trials) == 8
        assert all(t["errors"][0]["finite_pairs"] == c["rows"] for t in trials)
        winner = min(trials, key=lambda t: checked_median(t["discovery"]))
        assert winner["schedule"] == c["selected"][family]
    assert grids[0] == grids[1] and {n for n, _ in grids[0]} == {128, 256, 512, 1024}
    assert [w["offset"] for w in c["heldout"]] == [0, 1]
    assert len({c["train_seed"], *(w["seed"] for w in c["heldout"])}) == 3
    for w in c["heldout"]:
        assert (
            set(w["errors"]) == set(w["ordinary_host_sync_us"]) == {*FAMILIES, "eager", "inductor"}
            and not w["rejected"]
        )
        assert all(err[0]["finite_pairs"] == c["rows"] for err in w["errors"].values())
        times = {k: [] for k in w["errors"]}
        assert len(w["device_windows"]) == 7
        for round_ in w["device_windows"]:
            assert sorted(round_["order"]) == sorted(round_["samples"]) == sorted(times)
            for k, d in round_["samples"].items():
                assert len(d["samples_us"]) == 7
                times[k].append(checked_median(d))
        assert w["device_us"] == {k: median(v) for k, v in times.items()}
        ratios = [a / b for a, b in zip(times["block"], times["compact_mean"])]
        assert (
            median(ratios) == w["paired_block_over_compact"]
            and sum(v > 1 for v in ratios) == w["compact_wins"]
        )
        for d in w["ordinary_host_sync_us"].values():
            assert len(d["samples_us"]) == 31
            checked_median(d)


def test_static_disassembly_is_bound_to_the_selected_sources():
    from hashlib import sha256
    from tools.inspect_reduction_sass import op_counts

    assert op_counts(
        "Function : kernel\n /*0010*/ @!P0 SHFL.DOWN R0, R1;\n /*0020*/ BAR.SYNC.DEFER_BLOCKING 0;\n"
    ) == {"kernel": {"SHFL.DOWN": 1, "BAR.SYNC.DEFER_BLOCKING": 1}}
    with pytest.raises(ValueError):
        op_counts("cuobjdump: no disassembly present")
    report = json.loads((REPORTS / "compact-mean-sass.json").read_bytes())
    assert report["panel_sha256"] == sha256(PANEL.read_bytes()).hexdigest()
    case = json.loads(PANEL.read_bytes())["cases"][0]
    assert all(report[k] == case[k] for k in ("rows", "width", "dtype"))
    for name, record in report["families"].items():
        source = REPORTS.parent / "examples/generated/compact-mean" / (name + ".cu")
        assert sha256(source.read_bytes()).hexdigest() == record["source_sha256"]
        chosen = [v for v in case["trials"] if v["schedule"] == case["selected"][name]]
        assert len(chosen) == 1 and record["kernel_cache_id"] == chosen[0]["kernel_id"]
        assert record["schedule"] == case["selected"][name]
        for function in record["static_instructions"].values():
            assert all(type(n) is int and n > 0 for n in function.values())
            assert function["BAR.SYNC.DEFER_BLOCKING"] == (2 if name == "block" else 1)
