"""Replay the phase arithmetic offline and reject ambiguous profiler evidence."""

import json
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from statistics import median

import pytest
from tools.phase_attribution import map_ordinary, phase_totals, replay_intervals, summarize
from tools.verify_sources import verify_reports

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def sample():
    labels = []
    launches = []
    ks = []
    for i, name in enumerate(("gate", "up")):
        labels.append(
            dict(
                cat="user_annotation", ph="X", name="phase/" + name, ts=5 * i, dur=4, tid=8, pid=42
            )
        )
        launches.append(
            dict(
                cat="cuda_runtime",
                name="cudaLaunchKernel",
                ts=5 * i + 1,
                dur=2,
                pid=42,
                tid=8,
                args={"correlation": i + 1},
            )
        )
        ks.append(
            dict(
                cat="kernel",
                name="same kernel",
                ts=100 + 3 * i,
                dur=4,
                args={
                    "correlation": i + 1,
                    "device": 0,
                    "context": 1,
                    "stream": 7,
                    "grid": [2, 1, 1],
                    "block": [128, 1, 1],
                    "registers per thread": 32,
                    "shared memory": 0,
                },
            )
        )
    meta = {n: dict(group="mlp_projection", target="aten.linear.default") for n in ("gate", "up")}
    return labels, launches, ks, meta


def test_identical_symbols_resolve_by_fx_owner_and_overlap_is_not_idle_time():
    labels, launches, ks, meta = sample()
    m = map_ordinary(labels + launches + ks, meta)
    assert [r["node"] for r in m] == ["gate", "up"]
    ev = []
    for j in range(2):
        ev.append(
            dict(cat="cuda_runtime", name="cudaGraphLaunch", ts=j, args={"correlation": 40 + j})
        )
        for i, k in enumerate(ks):
            e = deepcopy(k)
            e["ts"] += 20 * j
            e["args"].update({"graph id": 2, "graph node id": 11 + i, "correlation": 40 + j})
            ev.append(e)
    ids, samples = replay_intervals(ev, m, 2)
    assert ids == [11, 12] and samples == [[[0, 4000], [3000, 4000]]] * 2
    t = phase_totals(m, samples[0])
    assert (
        t["reported_overlap"] == 1000
        and t["inter_kernel_gap"] == 0
        and t["interval_union"] == t["first_to_last"] == 7000
        and t["sum_kernel_durations"] == t["mlp_projection"] == 8000
    )
    ev[-1]["args"]["graph node id"] = 11
    with pytest.raises(ValueError, match="graph-node order"):
        replay_intervals(ev, m, 2)


@pytest.mark.parametrize(
    "defect,match",
    [
        ("missing", "missing or reused"),
        ("duplicate", "ambiguous launch"),
        ("nested", "one FX owner"),
        ("thread", "one FX owner"),
        ("streams", "one GPU stream"),
        ("memset", "copy or memset"),
    ],
)
def test_ambiguous_discovery_is_rejected(defect, match):
    labels, ls, ks, meta = sample()
    if defect == "missing":
        ls.pop()
    if defect == "duplicate":
        ls.append(ls[0])
    if defect == "nested":
        labels.append(labels[0])
    if defect == "thread":
        ls[0]["tid"] = 9
    if defect == "streams":
        ks[0]["args"]["stream"] = 9
    if defect == "memset":
        ks.append(dict(cat="gpu_memset"))
    with pytest.raises(ValueError, match=match):
        map_ordinary(labels + ls + ks, meta)


def test_published_decode_phase_evidence():
    path = REPORTS / "qwen-decode-phase-attribution-gb10.json"
    assert verify_reports([path])["verified_reports"] == 1
    d = json.loads(path.read_text())
    prev = json.loads((REPORTS / "qwen-decode-cache-fanout-gb10.json").read_text())
    assert (
        d["passed"]
        and d["export_seed"] != d["timing_seed"]
        and all(
            d[k] == prev[k]
            for k in (
                "device",
                "torch",
                "transformers",
                "export_seed",
                "timing_seed",
                "weight_seed",
            )
        )
    )
    assert [(r["prefix"], r["step"]) for r in d["cases"]] == [
        (p, i) for p in (1024, 2047) for i in (1, 2)
    ]
    nreplays = 0
    for c in d["cases"]:
        step = c["step"]
        old = next(z for z in prev["cases"] if z["initial_prefix"] == c["prefix"])
        gr = old["graphs"][step - 1]
        assert c["export_graph_sha256"] == sha256(gr["aten"].encode()).hexdigest()
        assert Counter(tuple(o) for o in c["window_order"]) == {
            ("separate", "fanout"): 4,
            ("fanout", "separate"): 4,
        }
        assert (
            len(c["inputs"]) == 6
            and len(c["reference"]) == 4
            and all(len(bytes.fromhex(t["sha256"])) == 32 for t in c["inputs"] + c["reference"])
        )
        for name, b in c["backends"].items():
            m = b["mapping"]
            assert len(m) == old["backends"][name]["timing"][step - 1]["profile"]["count"] and [
                r["ordinal"] for r in m
            ] == list(range(len(m)))
            assert (
                len(set(b["graph_node_ids"])) == len(m)
                and b["fx_sha256"] == sha256(gr["partitions"][name]["fx"].encode()).hexdigest()
            )
            assert sorted(r["detail"] for r in m if r["group"] == "cache_copy") == (
                ["cache_fanout"] if name == "fanout" else ["paired_append", "paired_head_repeat"]
            )
            assert [r["detail"] for r in m if r["group"] == "mlp_projection"] == [
                "layer.mlp." + x + "_proj.weight" for x in ("gate", "up", "down")
            ]
            for key in ("ordinary", "replay"):
                v = b[key]
                assert (
                    v["passed"]
                    and v["prefix_exact"]
                    and v["input_unchanged"]
                    and v["new_storage"]
                    and v["masked_nonzero"] == 0
                )
                assert (
                    len(v["outputs"]) == 4
                    and all(t["unequal_bits"] == 0 for t in v["outputs"].values())
                    and v["output_strides"] == [t["stride"] for t in c["reference"]]
                )
            assert len(b["windows"]) == 8 and b["median_us"] == summarize(m, b["windows"])
            for w in b["windows"]:
                assert (
                    len(w["event_samples_us"]) == 7
                    and min(w["event_samples_us"]) > 0
                    and len(w["intervals_ns"]) == 6
                )
                nreplays += len(w["intervals_ns"])
                for replay in w["intervals_ns"]:
                    t = phase_totals(m, replay)
                    assert (
                        t["sum_kernel_durations"] - t["reported_overlap"]
                        == t["interval_union"]
                        == t["first_to_last"] - t["inter_kernel_gap"]
                    )
                    assert t["mlp_projection"] == sum(
                        duration
                        for (_, duration), r in zip(replay, m, strict=True)
                        if r["group"] == "mlp_projection"
                    )
            assert b["median_us"]["unprofiled_graph"] == median(
                median(w["event_samples_us"]) for w in b["windows"]
            )
    assert nreplays == 384
