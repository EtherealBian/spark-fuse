"""Recompute admissions from every raw graph-event window. No CUDA or autotuner invocation."""

from hashlib import sha256
import json
import math
from pathlib import Path
from random import Random
from statistics import median
from test_phi3_native_evidence import samples
from tools.verify_sources import verify_reports

ROOT = Path(__file__).resolve().parents[1] / "reports"


def digest(obj):
    return sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def windows(ws, n, keys, seed):
    assert len(ws) == n
    rng = Random(seed)
    out = []
    for w in ws:
        expected = sorted(keys)
        rng.shuffle(expected)
        assert w["order"] == expected and set(w["samples_us"]) == set(keys)
        assert all(
            len(xs) == 7 and all(math.isfinite(x) and x > 0 for x in xs)
            for xs in w["samples_us"].values()
        )
        out.append({k: median(xs) for k, xs in w["samples_us"].items()})
    return out


def decision(c, expected, tallies):
    r = c["receipt"]
    assert (
        r["checksum"] == digest({k: v for k, v in r.items() if k != "checksum"})
        and r["min_gain"] == 1.02
        and not r["rejected"]
    )
    assert r["stamp"]["site"] == c["region"]["module"] and r["edges"] == [
        c["region"]["live_ins"],
        c["region"]["live_outs"],
    ]
    assert (
        "uuid" not in r["stamp"]
        and len(r["stamp"]["device_fingerprint"]) == 64
        and len(set(r["request_hashes"])) == 3
    )
    a = r["anchor"]

    def key(s):
        return f"{s['layout']}:{s['threads']}:{int(s['cache'])}:{s['vec']}"

    keys = {key(s) for s in r["candidates"]} | {a}
    assert len(keys) == 4 and a == key(c["region"]["schedule"])
    ws = windows(r["discovery"], 5, keys, r["seed"])
    b = min(keys, key=lambda k: (median(w[k] for w in ws), k != a, k))
    assert b == r["nominee"]
    if a == b:
        assert r["confirmation"] == []
        chosen = a
    else:
        ws = windows(r["confirmation"], 7, [a, b], r["seed"] + 1)
        ratios = [w[a] / w[b] for w in ws]
        wins = sum(x > 1 for x in ratios)
        m = median(ratios)
        chosen = b if wins >= 6 and m >= 1.02 else a
        assert r["decision"] == dict(
            ratios=ratios, wins=wins, median_ratio=m, min_gain=1.02, selected=chosen
        )
    assert chosen == r["decision"]["selected"] == key(r["selected"])
    tallies.append(chosen != a)
    for phase, group in [("train", keys), ("confirm", {a, b}), ("audit", {a, chosen})]:
        assert r["gates"][phase] == {k: expected for k in group}
    assert len(c["trials"]) == 2
    assert all(t["replay"] == expected for t in c["trials"])
    return r, a, b, chosen


def test_qwen_whole_context_receipts_and_timing():
    p = ROOT / "qwen-context-gb10.json"
    d = json.loads(p.read_bytes())
    assert verify_reports([p])["distinct_archives"] == 1
    source = (ROOT / "qwen-packed-mean-heldout-gb10.json").read_bytes()
    assert d["rms_report_sha256"] == sha256(source).hexdigest()
    rms = {c["sequence"]: c["kernels"]["packed"]["schedule"] for c in json.loads(source)["cases"]}
    assert [(c["sequence"], c["weight_seed"]) for c in d["cases"]] == [
        (s, w) for s in (33, 65, 129) for w in (533111, 581119)
    ]
    seen = set()
    tallies = []
    for c in d["cases"]:
        assert c["rms"] == rms[c["sequence"]]
        r, a, b, chosen = decision(c, {"live_ins": 2, "live_outs": 1, "whole": 2}, tallies)
        assert r["stamp"]["records"] == digest(c["records"])
        assert c["region"]["live_outs"] == ["mul_9"] and c["region"]["live_ins"] == [
            "linear_4",
            "linear_5",
        ]
        for t in c["trials"]:
            assert set(t["baselines"]) == {
                "eager",
                "inductor",
                "inductor_casts",
                *c["label"].values(),
            } and all(x == 1 for x in t["targets"].values())
            for z in t["baselines"].values():
                assert z["passed"] == (
                    all(
                        v["finite"] and v["tolerance_pass"] and v["unequal_bits"] == 0
                        for v in z["values"].values()
                    )
                    and z["masked_nonzero"] == 0
                    and z["max_row_sum_error"] <= 0.004
                )
        hashes = r["request_hashes"] + [digest(t["inputs"]) for t in c["trials"]]
        assert len(set(hashes)) == 5 and seen.isdisjoint(hashes)
        seen.update(hashes)
        allowed = {
            k
            for k in c["trials"][0]["baselines"]
            if all(t["baselines"][k]["passed"] for t in c["trials"])
        }
        assert allowed == set(c["admitted"])
        samples(c["whole"], allowed)
        samples(c["isolated"], set(c["label"].values()))
        assert all(
            c["whole"]["profiles"][n]["count"] == 19 and c["isolated"]["profiles"][n]["count"] == 1
            for n in c["label"].values()
        )
    assert len(tallies) == 6 and not all(tallies)


def test_projection_context_receipts_include_positive_deployments():
    p = ROOT / "context-projection-gb10.json"
    d = json.loads(p.read_bytes())
    assert verify_reports([p])["distinct_archives"] == 1
    assert [(c["shape"], c["offset"]) for c in d["cases"]] == [
        (shape, off) for shape in ([128, 2560], [63, 8160]) for off in (0, 1)
    ]
    tallies = []
    for c in d["cases"]:
        r, _, _, _ = decision(c, {"live_ins": 3, "live_outs": 2, "whole": 2}, tallies)
        assert [entry["offset"] for entry in r["stamp"]["inputs"]] == [c["offset"]] * 2
        hs = r["request_hashes"] + [digest(t["input_hashes"]) for t in c["trials"]]
        assert len(set(hs)) == 5
        for t in c["trials"]:
            assert set(t["values"]) == {
                "eager",
                "incumbent",
                "selected",
                "inductor",
                "inductor_casts",
            } and all(len(v) == 2 for v in t["values"].values())
        allowed = {
            k
            for k in c["trials"][0]["values"]
            if all(all(v["bit_mismatches"] == 0 for v in t["values"][k]) for t in c["trials"])
        }
        assert allowed == set(c["allowed"])
        samples(c["whole"], allowed)
        assert (
            c["whole"]["profiles"]["incumbent"]["count"]
            == c["whole"]["profiles"]["selected"]["count"]
        )
    assert any(tallies)
