"""Independent audit of the measured quantizer/epilogue ablation and skipped baselines."""

from hashlib import sha256
import json
from math import isfinite, prod
from pathlib import Path
from random import Random
from statistics import median
from tools.verify_sources import verify_reports

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports/projection-epilogue-gb10.json"


def test_projection_ablation_preregistered_paths_and_costs():
    d = json.loads(R.read_bytes())
    assert (
        not d["smoke"] and verify_reports([R])["distinct_archives"] == 1 and len(d["cases"]) == 12
    )
    seen = set()
    ids = set()
    wids = set()
    for c in d["cases"]:
        seed = c["seed"]
        label = c["label"]
        scope = c["scope"]
        key = seed, label, scope
        assert key not in seen
        seen.add(key)
        proj = scope == "projection"
        hs = {v for a in c["inputs_sha256"] for v in a}
        assert len(hs) == (8 if proj else 12) and not hs & ids
        ids |= hs
        assert len(c["geometry"]) == len(c["reference_sha256"]) == 4
        for off, g in enumerate(c["geometry"]):
            assert len(g) == (2 if proj else 3) and all(t[2] == off for t in g)
        if proj:
            assert c["cpu_accumulator_sha256"] == [out[3] for out in c["reference_sha256"]] and set(
                c["weights_sha256"]
            ) == {"weight", "channel_scale"}
            w = c["weights_sha256"]["weight"]
            assert w not in wids
            wids.add(w)
        else:
            assert not c["cpu_accumulator_sha256"] and not c["weights_sha256"]
        t = c["timing"]
        gates = c["gates"]
        expected = {"eager", "aten", "mixed", "tail", "inductor", "cast_inductor"} | (
            {"quant", "both"} if proj else set()
        )
        assert set(gates) == expected
        passed = []
        for name, g in gates.items():
            if "error" in g:
                assert name in ("inductor", "cast_inductor") and not g["admitted"]
                continue
            assert len(g["outputs"]) == 4 and all(
                len(v) == (5 if proj else 1) for v in g["outputs"]
            )
            assert g["admitted"] == (
                g["fresh"]
                and all(
                    v["geometry_match"] and v["different_bytes"] == 0
                    for row in g["outputs"]
                    for v in row
                )
            )
            if proj:
                for row, w in zip(g["outputs"], g["code_witnesses"], strict=True):
                    assert (w is None) == (row[1]["different_bytes"] == 0)
            if g["admitted"]:
                passed.append(name)
        assert expected - {"inductor", "cast_inductor"} <= set(passed) and set(
            t["profiles"]
        ) == set(passed)
        for mode, records in c["records"].items():
            q = [r for r in records if r.get("kind") == "residual_token_quant"]
            e = [r for r in records if r.get("kind") == "ordered_int32_epilogue"]
            assert len(q) == (mode in ("quant", "both")) and len(e) == (mode in ("tail", "both"))
            if e:
                assert e[0]["shape"] == [prod(c["shape"][:-1]), c["channels"]] and e[0][
                    "output_roles"
                ] == ["output"]
                if label == "b16":
                    assert (
                        e[0]["cuda_sha256"]
                        == sha256(
                            (ROOT / "examples/generated/token-quant/dequant_32x128.cu").read_bytes()
                        ).hexdigest()
                    )
            prof = t["profiles"][mode]
            counts = prof["kernel_names"]
            assert sum(k for s, k in counts.items() if "residual_token_quant" in s) == len(q)
            assert sum(k for s, k in counts.items() if "ordered_int32_epilogue" in s) == len(
                e
            ) and prof["count"] == sum(counts.values())
        rng = Random(seed + c["channels"] + (199 if proj else 797))
        assert len(t["windows"]) == 7
        for window in t["windows"]:
            order = passed.copy()
            rng.shuffle(order)
            assert window["order"] == order
        for metric, n in [("device", 7), ("host", 15)]:
            assert set(t["median_us"][metric]) == set(passed)
            for name in passed:
                samples = [w[metric][name]["samples_us"] for w in t["windows"]]
                assert all(len(v) == n and all(isfinite(x) and x > 0 for x in v) for v in samples)
                assert t["median_us"][metric][name] == median(map(median, samples))
    assert seen == {
        (s, n, k)
        for s in (903757, 928043)
        for n in ("b16", "h33", "b64")
        for k in ("isolated", "projection")
    }
