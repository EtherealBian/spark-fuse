"""Offline validation of all numerical gates, source identity, launches and paired clocks."""

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from random import Random
from statistics import median
from tools.verify_sources import verify_reports

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/projection-affine-gb10.json"


def test_ordered_bias_narrow_projection_evidence():
    report = json.loads(REPORT.read_bytes())
    assert not report["smoke"] and verify_reports([REPORT])["distinct_archives"] == 1
    seen = set()
    inputs = set()
    for c in report["cases"]:
        m, k, ns = c["m"], c["k"], c["widths"]
        assert (m, k, ns) == {
            "short": (17, 1536, [128, 64, 64]),
            "visible": (33, 1536, [512, 256, 256]),
            "prefill": (129, 3072, [1536, 256, 256]),
        }[c["label"]]
        key = (c["label"], c["seed"])
        assert key not in seen
        seen.add(key)
        debug = c["debug_outputs"]
        assert debug == (c["label"] == "visible")
        n = 21 if debug else 3
        ix = [v for r in c["input_sha256"] for v in r]
        assert len(ix) == len(set(ix)) == 20 and inputs.isdisjoint(ix)
        inputs.update(ix)
        assert len(c["weight_sha256"]) == 9
        refs = c["ref_sha256"]
        assert len(refs) == len(c["oracle"]) == 8
        assert all(len(r) == n for r in refs)
        for r, q in zip(refs, c["oracle"], strict=True):
            assert (
                len(q["cpu_int32"]) == 3
                and len(set(q["cpu_int32"])) == 3
                and q["narrow"] == (r[8::6] if debug else r)
            )
            if debug:
                assert (q["codes"], q["scales"], q["cpu_int32"]) == (r[1], r[2], r[3::6])
        gates = c["gates"]
        assert set(gates) == {
            "eager",
            "separate",
            "grouped_affine",
            "packed_affine",
            "inductor",
            "cast_inductor",
        }
        for name, g in gates.items():
            checks = [g["ordinary"]] + ([g["resident"]] if name in c["metadata"] else [])
            assert g["admitted"] == all(x["admitted"] for x in checks)
            for a in checks:
                assert (
                    a["fresh"]
                    and a["container"]
                    and len(a["outputs"]) == 8
                    and all(len(x) == n for x in a["outputs"])
                )
                assert a["admitted"] == all(
                    v["geometry_match"] and v["different_bytes"] == 0
                    for row in a["outputs"]
                    for v in row
                )
        passed = [k for k, g in gates.items() if g["admitted"]]
        assert {"eager", "separate", "grouped_affine", "packed_affine"} <= set(passed)
        profiles = c["ordinary_profiles"]
        assert set(profiles) == set(passed)
        for name, expected in [
            ("eager", 28),
            ("separate", 10),
            ("grouped_affine", 5),
            ("packed_affine", 3),
        ]:
            p = profiles[name]
            assert p["calls"] == 1 and sum(p["kernel_names"].values()) == p["count"] == expected
            if name == "eager":
                continue
            md = c["metadata"][name]
            assert (
                md["request_backend"] == "native"
                and len(md["output_bindings"]) == n
                and all(b["source"] == "copy" for b in md["output_bindings"])
            )
            assert c["setup"][name]["wall_ms"] > 0
            if name == "separate":
                assert sum(r.get("kind") == "ordered_int32_epilogue" for r in md["regions"]) == 3
                continue
            kind = "grouped_int32_affine" if name == "grouped_affine" else "packed_int32_projection"
            (entry,) = (r for r in md["regions"] if r.get("kind") == kind)
            assert [b["shape"] for b in entry["branches"]] == [[m, n] for n in ns]
            roles = (
                (["acc"] if name == "packed_affine" else [])
                + ["cast", "row_scaled", "output", "biased", "narrow"]
                if debug
                else ["narrow"]
            )
            assert all(b["output_roles"] == roles for b in entry["branches"])
            assert [b["narrow_dtype"] for b in entry["branches"]] == [
                "f16" if dt == "torch.float16" else "bf16" for dt in c["narrow_dtypes"]
            ]
            if name == "packed_affine" and c["label"] == "short":
                assert (
                    sha256(
                        (
                            ROOT / "examples/generated/token-quant/affine_17x128_64_64.cu"
                        ).read_bytes()
                    ).hexdigest()
                    == entry["cuda_sha256"]
                )
        d = c["ordinary_timing"]
        rng = Random(c["seed"] + m + 309)
        assert len(d["windows"]) == 7
        for win in d["windows"]:
            order = passed.copy()
            rng.shuffle(order)
            assert win["order"] == order
        for clock, nt in [("device", 7), ("host", 15)]:
            for name in passed:
                samples = [w[clock][name]["samples_us"] for w in d["windows"]]
                assert all(len(v) == nt and all(isfinite(x) and x > 0 for x in v) for v in samples)
                assert d["median_us"][clock][name] == median(map(median, samples))
        s = c["serving"]
        rng = Random(c["seed"] + m + 557)
        assert len(s["windows"]) == 7 and s["seed"] == c["seed"] + m + 557
        for w in s["windows"]:
            order = sorted(passed)
            rng.shuffle(order)
            assert order == w["order"]
        for name in passed:
            for metric in (
                "fixed_sync_us",
                "rotating_sync_us",
                "fixed_submit_us",
                "rotating_submit_us",
                "queued8_us_per_forward",
            ):
                samples = [w["samples"][name][metric] for w in s["windows"]]
                assert all(
                    len(v) == (5 if metric.startswith("queued") else 16)
                    and all(isfinite(x) and x > 0 for x in v)
                    for v in samples
                )
                assert s["median_us"][name][metric] == median(map(median, samples))
    assert seen == {
        (label, seed) for label in ("short", "visible", "prefill") for seed in (1661987, 1674131)
    }
