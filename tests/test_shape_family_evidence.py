"""Verify logical storage, per-process allocator deltas, gates and mixed-shape raw windows offline."""

from hashlib import sha256
import json
import math
from pathlib import Path
from random import Random
from statistics import median
from tools.verify_sources import verify_reports

R = Path(__file__).resolve().parents[1] / "reports"


def test_shape_family_shared_state_and_alternating_requests():
    path = R / "shape-family-qwen-gb10.json"
    doc = json.loads(path.read_bytes())
    assert not doc["smoke"] and verify_reports([path])["distinct_archives"] == 1
    assert (
        doc["rms_report_sha256"]
        == sha256((R / "qwen-packed-mean-heldout-gb10.json").read_bytes()).hexdigest()
    )
    assert [c["weight_seed"] for c in doc["cases"]] == [903383, 911293]
    weights = []
    Rms = 93595648
    for c in doc["cases"]:
        sig = c["weight_sha256"]
        assert len(sig) == 12 and len(set(sig.values())) == 12
        weights.append(sig)
        assert c["sequences"] == [33, 65, 129] and c["request_shapes"] == c["sequences"] * 4
        assert (
            len(c["inputs"]) == len(c["references"]) == 12
            and len({h for pair in c["references"] for h in pair}) == 24
        )
        for j in range(3):
            assert all(len({a[i] for a in c["inputs"][j::3]}) == 4 for i in range(4))
        stage = sum(2 * (2 * s * 1536 + 4 * s * 128 + 2 * s * s) for s in c["sequences"])
        snap = {}
        for kind, n in [("separate", 3), ("shared", 1)]:
            m = c["memory"][kind]
            inv = m["inventory"]
            assert (
                m["variant"] == kind
                and m["weight_seed"] == c["weight_seed"]
                and m["weight_sha256"] == sig
                and inv == c["inventory"][kind]
            )
            assert inv == dict(
                named_state_per_shape=12,
                unique_weight_stores=12 * n,
                unique_weight_bytes=Rms * n,
                weight_bytes_counting_each_shape=Rms * 3,
                input_staging_bytes=stage,
                logical_static_bytes=stage + Rms * n,
            )
            assert (
                m["peak_during_build"] >= m["after_build"] > m["baseline"] == m["after_close"]
                and m["setup_ms"] > 0
            )
            assert m["retained_delta"] == m["after_build"] - m["baseline"]
            snap[kind] = m
        assert snap["separate"]["baseline"] == snap["shared"]["baseline"]
        for k in ("after_build", "peak_during_build", "retained_delta"):
            assert snap["separate"][k] - snap["shared"][k] == 2 * Rms
        required = {"separate_direct", "shared_direct", "shared_routed", "mixed", "eager"}
        passed = set()
        assert set(c["gates"]) == required | {"inductor", "inductor_casts"}
        for n, g in c["gates"].items():
            assert len(g["checks"]) == 12
            ok = g["distinct_outputs"]
            for s, q in zip(c["request_shapes"], g["checks"], strict=True):
                assert q["values"]["hidden"]["shape"] == [2, s, 1536] and q["values"][
                    "probabilities"
                ]["shape"] == [2, 12, s, s]
                bits = all(
                    v["finite"] and v["tolerance_pass"] and v["unequal_bits"] == 0
                    for v in q["values"].values()
                )
                assert q["passed"] == (
                    bits and q["masked_nonzero"] == 0 and q["max_row_sum_error"] <= 0.004
                )
                ok = ok and q["passed"] and q["public_layout"] and q["alias_match"]
            assert ok == g["passed"]
            if ok:
                passed.add(n)
        t = c["timing"]
        assert required <= passed and t["median_us"].keys() == passed and len(t["windows"]) == 7
        rng = Random(t["seed"])
        for w in t["windows"]:
            order = sorted(passed)
            rng.shuffle(order)
            assert w["order"] == order and w["samples"].keys() == passed
            for row in w["samples"].values():
                assert set(row) == {"sync_us", "submit_us", "burst_us_per_forward"}
                for k, xs in row.items():
                    assert len(xs) == (5 if k.startswith("burst") else 24) and all(
                        math.isfinite(x) and x > 0 for x in xs
                    )
                assert all(a <= b for a, b in zip(row["submit_us"], row["sync_us"], strict=True))
        for n, row in t["median_us"].items():
            for k, x in row.items():
                assert x == median(median(w["samples"][n][k]) for w in t["windows"])
    assert weights[0]["layer.mlp.up_proj.weight"] != weights[1]["layer.mlp.up_proj.weight"]
