"""Offline consistency checks for the reduction-order incident; no CUDA context needed."""

import gzip
import json
from pathlib import Path
from statistics import median
from tools.verify_sources import verify_reports

R = Path(__file__).resolve().parents[1] / "reports"


def test_mean_tree_provenance_and_transaction():
    files = [
        R / name
        for name in (
            "mean-tree-history.json.gz",
            "mean-tree-replay.json.gz",
            "mean-tree-boundaries-gb10.json",
        )
    ]
    assert verify_reports(files)["verified_reports"] == 3
    history, replay = [json.loads(gzip.decompress(f.read_bytes())) for f in files[:2]]
    assert history["job"] == replay["job"] and history["job"]["seed"] == 1752
    assert history["status"] == "discovery-error" and "(25, 3533)" in history["diagnostic"]
    t = replay["tuning"]
    assert len(t["trials"]) == 6 and not t["invalid"]
    assert {(s["schedule"]["threads"], s["schedule"]["cache"]) for s in t["trials"]} == {
        (n, c) for n in (128, 256, 512) for c in (True, False)
    }
    assert all(s["schedule"]["layout"] == "mean_tree" for s in t["trials"])
    assert all(e["unequal"] == 0 for e in t["heldout_errors"])
    assert replay["status"] == "parity-failure" and replay["failed_backends"] == ["inductor"]
    comp = replay["comparison"]
    assert comp["seed"] == 3752
    assert comp["backends"]["sparkfuse"]["errors"][0]["unequal"] == 0
    assert (
        "device" not in comp["backends"]["inductor"]
        and "(43, 4916)" in comp["backends"]["inductor"]["diagnostic"]
    )


def test_mean_tree_profile_bits_and_times():
    d = json.loads((R / "mean-tree-boundaries-gb10.json").read_text())
    assert len(d["boundaries"]) == 6
    assert [c["width"] for c in d["cases"]] == [8960, 8160, 8164, 1536]
    for c, grid in zip(d["cases"], (48, 2, 17, 2), strict=True):
        (kernel,) = c["eager_mean_kernels"]
        assert (
            kernel["grid"] == [grid, 1, 1]
            and kernel["block"] == [32, 16, 1]
            and "reduce_kernel" in kernel["name"]
        )
        seeds = [r["seed"] for r in c["requests"]]
        assert len(set(seeds)) == 6 and c["timing_seed"] not in seeds
        for i, req in enumerate(c["requests"]):
            stats = req["boundaries"]["ordered"]
            assert len(stats) == 6 and all(
                s["elements"] and s["bit_mismatches"] == 0 for s in stats
            )
            assert req["storage_offsets"] == ([0, 0, 0] if i < 3 else [1, 2, 3])
        passed = {k for k, v in c["checks"].items() if v["passed"]}
        assert set(c["times"]) == passed and len(c["orderings"]) == 7
        assert all(set(order) == passed and len(order) == len(passed) for order in c["orderings"])
        for k, series in c["times"].items():
            assert len(series) == 7
            for clock in ("device", "host"):
                medians = [median(s[clock]["samples_us"]) for s in series]
                assert medians == [s[clock]["median_us"] for s in series]
                assert c["median_us"][k][clock] == median(medians)
    witness = d["cases"][0]["requests"][1]["witness"]
    assert witness["index"] == [25, 3533] and witness["eager"] == witness["ordered"]
    assert witness["old"][4] == -2.578125 and witness["ordered"][4] == -2.546875
