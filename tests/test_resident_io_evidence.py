"""An ordinary-graph census is not a request latency: verify both and all raw host clocks."""

import json
from math import isfinite
from pathlib import Path
from random import Random
from statistics import median
from tools.verify_sources import verify_reports

R = Path(__file__).resolve().parents[1] / "reports/resident-io-gb10.json"


def test_native_resident_request_host_ablation():
    r = json.loads(R.read_bytes())
    assert not r["smoke"] and verify_reports([R])["distinct_archives"] == 1
    seen = set()
    inputs = set()
    wh = set()
    for c in r["cases"]:
        key = c["label"], c["seed"]
        assert key not in seen
        seen.add(key)
        m, k, ns = c["m"], c["k"], c["widths"]
        full = bool(ns)
        n = 9 if full else 1
        t = c["serving"]
        seed = c["seed"] + m + k + 463
        iq = [v for a in c["inputs_sha256"] for v in a]
        assert len(iq) == len(set(iq)) == 20 and inputs.isdisjoint(iq)
        inputs.update(iq)
        w = list(c["weights_sha256"].values())
        assert len(w) == (6 if full else 0) and wh.isdisjoint(w)
        wh.update(w)
        ref = c["reference_sha256"]
        assert len(ref) == 8 and all(len(y) == n for y in ref)
        assert c["cpu_int32_sha256"] == ([y[3::2] for y in ref] if full else [])
        profiles = c["ordinary_graph_profiles"]
        assert profiles["python"] == profiles["native"]
        p = profiles["python"]
        assert p["calls"] == 1 and sum(p["kernel_names"].values()) == p["count"] == (
            3 if full else 2
        )
        if full:
            for marker in ["packed_int32_fanout", "residual_token_quant"]:
                assert sum(v for name, v in p["kernel_names"].items() if marker in name) == 1
        md = c["metadata"]
        assert {j: v for j, v in md["python"].items() if j != "request_backend"} == {
            j: v for j, v in md["native"].items() if j != "request_backend"
        }
        for name in ("native", "python"):
            d = md[name]
            assert (
                d["request_backend"] == name
                and len(d["output_bindings"]) == n
                and all(
                    b["source"] == "copy" and b["base"] == i
                    for i, b in enumerate(d["output_bindings"])
                )
            )
            assert d["input_bytes"] == 4 * m * k
            assert c["setup"][name]["wall_ms"] > 0
        names = {"eager", "python", "native"}
        assert (
            set(c["gates"]) == set(t["median_us"]) == names
            and t["seed"] == seed
            and len(t["windows"]) == 7
        )
        for g in c["gates"].values():
            assert (
                g["fresh"]
                and g["container_match"]
                and len(g["outputs"]) == 8
                and all(
                    len(y) == n
                    and all(x["geometry_match"] and x["different_bytes"] == 0 for x in y)
                    for y in g["outputs"]
                )
            )
        rng = Random(seed)
        lens = {
            "fixed_sync_us": 16,
            "rotating_sync_us": 16,
            "fixed_submit_us": 16,
            "rotating_submit_us": 16,
            "queued8_us_per_forward": 5,
        }
        for win in t["windows"]:
            order = sorted(names)
            rng.shuffle(order)
            assert win["order"] == order and set(win["samples"]) == names
        for name, stats in t["median_us"].items():
            assert set(stats) == set(lens)
            for metric, num in lens.items():
                samples = [win["samples"][name][metric] for win in t["windows"]]
                assert all(len(y) == num and all(isfinite(x) and x > 0 for x in y) for y in samples)
                assert stats[metric] == median(map(median, samples))
    assert seen == {
        (label, s)
        for s in (1481849, 1495241)
        for label in ("qkv-short", "qkv-prefill", "one-output")
    }
