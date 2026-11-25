"""Recompute transport clocks and distinguish batched API calls from physical copy activities."""

import hashlib
import json
from math import isfinite
from pathlib import Path
from random import Random
from statistics import median

from sparkfuse.slot_copy import generate

ROOT = Path(__file__).resolve().parents[1]


def check_clock(clock, seed, names):
    assert clock["seed"] == seed and len(clock["windows"]) == 7 and set(clock["median_us"]) == names
    rng = Random(seed)
    for w in clock["windows"]:
        order = sorted(names)
        rng.shuffle(order)
        assert w["order"] == order and set(w["samples"]) == names
    metrics = {
        "rotating_sync_us",
        "rotating_submit_us",
        "fixed_sync_us",
        "fixed_submit_us",
        "queued8_us_per_forward",
    }
    for n, m in clock["median_us"].items():
        assert set(m) == metrics
        for k, t in m.items():
            ws = [w["samples"][n][k] for w in clock["windows"]]
            assert all(
                len(row) == (5 if k.startswith("queued") else 16)
                and all(isfinite(x) and x > 0 for x in row)
                for row in ws
            )
            assert median(map(median, ws)) == t


def test_fused_slot_transport_evidence():
    d = json.loads((ROOT / "reports/fused-slot-gb10.json").read_bytes())
    o = json.loads((ROOT / "reports/projection-affine-gb10.json").read_bytes())
    assert (
        not d["smoke"]
        and d["oracle_source_tree"] == o["provenance"]["source_tree_sha256"]
        and (ROOT / "reports" / d["provenance"]["archive"]).is_file()
    )
    keys = []
    for c in d["cases"]:
        seed, label, stream, m, n = [c[k] for k in ("seed", "label", "stream", "m", "outputs")]
        keys.append((seed, label, stream))
        assert n == (21 if label == "visible" else 3)
        orig = next(x for x in o["cases"] if x["seed"] == seed and x["label"] == label)
        assert (
            c["ref_sha256"] == orig["ref_sha256"]
            and c["input_sha256"] == orig["input_sha256"]
            and len(orig["oracle"]) == 8
            and all(len(row) == n for row in c["ref_sha256"])
        )
        assert c["versions_incremented_once"] and c["retained_fresh_result_checked"]
        names = {"aten", "cuda"} | ({"raw", "batch"} if stream == "explicit" else set())
        assert (
            set(c["capture_order"])
            == names
            == set(c["setup"])
            == set(c["gates"])
            == set(c["profile"])
        )
        assert c["ring_setup"]["retained_allocator_delta_bytes"] > 0
        meta = c["metadata"]["cuda"]["slot_copy"]
        assert (
            hashlib.sha256(generate(meta["output_bytes"]).encode()).hexdigest()
            == meta["cuda_sha256"]
            and len(meta["output_bytes"]) == n
        )
        compute = c["profile"]["aten"]["kernel"]
        assert sum(compute.values()) == 3
        for name in names:
            gates = c["gates"][name]
            assert len(gates) == 8 and all(
                len(row) == n
                and all(t["geometry_match"] and t["different_bytes"] == 0 for t in row)
                for row in gates
            )
            assert c["allocator_8"][name] == dict(new_cuda_allocations=0, retained_cuda_bytes=0)
            p = c["profile"][name]
            api = p["cuda_runtime"]
            assert api["cudaGraphLaunch"] == 1 and api["cudaMemcpyAsync"] == (
                2 if name in ("batch", "cuda") else 2 + n
            )
            assert api.get("cudaMemcpyBatchAsync", 0) == int(name == "batch") and sum(
                p["gpu_memcpy"].values()
            ) == (2 if name == "cuda" else 2 + n)
            assert {
                k: v for k, v in p["kernel"].items() if "sparkfuse_slot_copy" not in k
            } == compute
            assert sum(v for k, v in p["kernel"].items() if "sparkfuse_slot_copy" in k) == int(
                name == "cuda"
            )
        check_clock(c["clocks"], seed + m + 953 + int(stream == "explicit"), names)
    assert sorted(keys) == sorted(
        (s, label, t)
        for s in (1661987, 1674131)
        for label in ("short", "visible", "prefill")
        for t in ("default", "explicit")
    )
    single = []
    for c in d["single_output_controls"]:
        seed, n = c["seed"], c["elements"]
        single.append((seed, n))
        assert c["bytes"] == n * 4 and c["stream"] == "default" and len(set(c["ref_sha256"])) == 8
        for rows in c["gates"].values():
            assert len(rows) == 8 and all(
                r["different_bytes"] == 0 and r["geometry_match"] for r in rows
            )
        check_clock(c["clocks"], seed + n + 101, {"aten", "cuda"})
    assert sorted(single) == sorted(
        (s, n) for s in (1869841, 1869943) for n in (67, 65537, 4194305)
    )
