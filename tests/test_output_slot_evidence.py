"""Offline replay of allocation, arithmetic, profiler and timing evidence for explicit output reuse."""

import json
from math import isfinite
from pathlib import Path
from random import Random
from statistics import median

ROOT = Path(__file__).resolve().parents[1]


def test_output_slot_machine_evidence():
    p = json.loads((ROOT / "reports/output-slot-gb10.json").read_bytes())
    oracle = json.loads((ROOT / "reports/projection-affine-gb10.json").read_bytes())
    assert not p["smoke"] and (ROOT / "reports" / p["provenance"]["archive"]).is_file()
    seen = []
    for c in p["cases"]:
        seed, label, m, n = c["seed"], c["label"], c["m"], c["outputs"]
        seen.append((seed, label))
        assert n == (21 if label == "visible" else 3)
        o = next(q for q in oracle["cases"] if q["seed"] == seed and q["label"] == label)
        assert (
            c["oracle_source_tree"] == oracle["provenance"]["source_tree_sha256"]
            and c["ref_sha256"] == o["ref_sha256"]
            and len(o["oracle"]) == 8
        )
        assert c["independent_ring_slots"] and c["explicit_reuse_overwrites"]
        mem = c["allocator_8"]
        assert (
            mem["fresh"]["new_cuda_allocations"] == 8 * n
            and mem["fresh"]["retained_cuda_bytes"]
            == c["ring_setup"]["retained_allocator_delta_bytes"]
            >= c["ring_logical_bytes"]
            > 0
        )
        assert (
            mem["run_into"] == {"new_cuda_allocations": 0, "retained_cuda_bytes": 0}
            and c["ring_setup"]["wall_ms"] > 0
        )
        for name, rows in c["gates"].items():
            assert len(rows) == 8 and all(
                len(row) == n
                and all(r["geometry_match"] and r["different_bytes"] == 0 for r in row)
                for row in rows
            )
            prof = c["profile"][name]
            assert (
                prof["cuda_runtime"]["cudaGraphLaunch"] == 1
                and prof["cuda_runtime"]["cudaMemcpyAsync"] == n + 2
                and sum(prof["gpu_memcpy"].values()) == n + 2
            )
            assert (
                prof["cpu_op"].get("aten::clone", 0) == (n if name == "fresh" else 0)
                and sum(prof["kernel"].values()) == 3
            )
        assert c["profile"]["fresh"]["kernel"] == c["profile"]["run_into"]["kernel"]
        clock = c["clocks"]
        assert clock["seed"] == seed + m + 731 and len(clock["windows"]) == 7
        r = Random(clock["seed"])
        for w in clock["windows"]:
            order = ["fresh", "run_into"]
            r.shuffle(order)
            assert w["order"] == order
        for name, metrics in clock["median_us"].items():
            assert set(metrics) == {
                "rotating_sync_us",
                "rotating_submit_us",
                "fixed_sync_us",
                "fixed_submit_us",
                "queued8_us_per_forward",
            }
            for metric, v in metrics.items():
                samples = [w["samples"][name][metric] for w in clock["windows"]]
                assert all(
                    len(s) == (5 if metric.startswith("queued") else 16)
                    and all(isfinite(x) and x > 0 for x in s)
                    for s in samples
                ) and v == median(map(median, samples))
    assert sorted(seen) == sorted(
        (s, label) for s in (1661987, 1674131) for label in ("short", "visible", "prefill")
    )
