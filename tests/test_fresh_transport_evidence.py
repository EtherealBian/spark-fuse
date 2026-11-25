"""Offline replay of clocks, ownership accounting, trace topology and previous CPU oracle bindings."""

from pathlib import Path
from statistics import median
import json


def test_fresh_transport_panel():
    root = Path(__file__).resolve().parents[1] / "reports"
    d = json.loads((root / "fresh-transport-gb10.json").read_bytes())
    old = json.loads((root / "projection-affine-gb10.json").read_bytes())
    assert (
        not d["smoke"]
        and len(d["cases"]) == 12
        and len(d["controls"]) == 6
        and d["oracle_source_tree"] == old["provenance"]["source_tree_sha256"]
    )
    assert (root / d["provenance"]["archive"]).is_file()
    keys = set()
    for c in d["cases"] + d["controls"]:
        if c["label"] != "single_relu":
            k = (c["seed"], c["label"], c["stream"])
            assert k not in keys
            keys.add(k)
            ref = next(v for v in old["cases"] if (v["seed"], v["label"]) == k[:2])
            assert c["ref_sha256"] == ref["ref_sha256"] and c["input_sha256"] == ref["input_sha256"]
        n = c["outputs"]
        arms = {"aten", "cuda", "explicit_allocate"}
        assert (
            c["retained_after_close"]
            and set(c["gates"]) == arms
            and all(v == [n] * 8 for v in c["gates"].values())
        )
        assert all(v == c["allocations_8"]["aten"] for v in c["allocations_8"].values())
        assert c["allocations_8"]["aten"]["new_cuda_allocations"] == n * 8
        assert (
            "fresh_copy" not in c["metadata"]["aten"] and "slot_copy" not in c["metadata"]["cuda"]
        )
        t = c["clocks"]
        w = t["windows"]
        assert len(w) == 7
        for arm in arms:
            trace = c["profile"][arm]
            inputs = 1 if c["label"] == "single_relu" else 2
            assert sum(trace["gpu_memcpy"].values()) == inputs + (n if arm == "aten" else 0)
            assert sum(v for k, v in trace["kernel"].items() if "sparkfuse_slot_copy" in k) == int(
                arm != "aten"
            )
            for metric, reported in t["median_us"][arm].items():
                assert all(
                    set(row["order"]) == arms
                    and len(row["samples"][arm][metric])
                    == (5 if metric == "queued8_us_per_forward" else 16)
                    for row in w
                )
                assert reported == median(median(row["samples"][arm][metric]) for row in w)
    assert len(keys) == 12
