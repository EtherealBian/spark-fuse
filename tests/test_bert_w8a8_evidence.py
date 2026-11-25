"""Replay the measured real-layer gates, independent integer-oracle inventory and shuffled clocks."""

import json
from math import isfinite
from pathlib import Path
from random import Random
from statistics import median


def test_quantized_bert_layer_archived_evidence():
    root = Path(__file__).resolve().parents[1] / "reports"
    p = json.loads((root / "bert-w8a8-gb10.json").read_bytes())
    assert not p["smoke"] and (root / p["provenance"]["archive"]).is_file()
    seen = []
    for c in p["cases"]:
        m, seed, res = c["m"], c["seed"], c["residual"]
        seen.append((m, seed))
        assert (
            res == (m == 33)
            and len(c["oracle"]) == len(c["ref_sha256"]) == 8
            and len(c["inputs_sha256"]) == 10
        )
        assert len({tuple(v) for v in c["inputs_sha256"]}) == 10 and all(
            len(v) == (2 if res else 1) for v in c["inputs_sha256"]
        )
        assert any("attention.self.query.rhs" in n for n in c["state_sha256"]) and any(
            "intermediate.dense.weight" in n for n in c["state_sha256"]
        )
        assert c["exported_fx"].count("target=torch.ops.aten._int_mm.default") == 3
        for record in c["oracle"]:
            assert (
                len(record["branches"]) == 3
                and all(len(record[n]) == 64 for n in ("activation", "codes", "scale"))
                and all(len(b["cpu_int32"]) == len(b["narrow"]) == 64 for b in record["branches"])
            )
        assert len(c["dense_bf16_drift"]) == 8 and all(
            len(row) == 2
            and all(0 < r["rms"] <= r["max_abs"] and isfinite(r["max_abs"]) for r in row)
            for row in c["dense_bf16_drift"]
        )
        assert c["build_failures"] == {}
        names = ["eager", "independent", "shared", "packed"]
        assert list(c["ordinary_profiles"]) == names
        for n, d in c["gates"].items():
            assert d["admitted"] == (n in names)
            for g in (d["ordinary"], d["resident"]):
                if g is None:
                    continue
                assert (
                    g["fresh"]
                    and g["container"]
                    and len(g["outputs"]) == 8
                    and all(
                        len(row) == 2
                        and all(q["geometry_match"] and q["dtype"] == "torch.bfloat16" for q in row)
                        for row in g["outputs"]
                    )
                )
                assert (
                    all(q["different_bytes"] == 0 for row in g["outputs"] for q in row)
                    == d["admitted"]
                )
        for n, num in zip(names, (59 if res else 58, 22, 18, 16), strict=True):
            pr = c["ordinary_profiles"][n]
            assert pr["calls"] == 1 and pr["count"] == sum(pr["kernel_names"].values()) == num
            if n == "eager":
                continue
            rr = c["metadata"][n]["regions"]
            assert c["setup"][n]["wall_ms"] > 0
            kinds = [r.get("kind") for r in rr]
            assert kinds.count("activation_token_quant") + kinds.count("residual_token_quant") == (
                3 if n == "independent" else 1
            )
            if n != "independent":
                (group,) = [r for r in rr if r.get("kind") == "shared_hidden_token_quantizers"]
                assert (
                    group["branches"] == 3
                    and group["eliminated"] == 2
                    and group["shape"] == [1, m, 768]
                )
            if n == "packed":
                assert any(
                    r.get("kind") == "packed_int32_projection" and len(r["branches"]) == 3
                    for r in rr
                )
        d = c["ordinary_timing"]
        s = c["serving"]
        assert len(d["windows"]) == len(s["windows"]) == 7
        rng = Random(seed + m + 373)
        rng2 = Random(seed + m + 499)
        for w, t in zip(d["windows"], s["windows"], strict=True):
            order = names.copy()
            rng.shuffle(order)
            assert order == w["order"]
            alt = sorted(names)
            rng2.shuffle(alt)
            assert alt == t["order"]
        for n in names:
            for metric, nt in [("host", 15), ("device", 7)]:
                vals = [w[metric][n]["samples_us"] for w in d["windows"]]
                assert all(
                    len(row) == nt and all(isfinite(v) and v > 0 for v in row) for row in vals
                ) and d["median_us"][metric][n] == median(map(median, vals))
            for metric, v in s["median_us"][n].items():
                vals = [w["samples"][n][metric] for w in s["windows"]]
                assert all(
                    len(row) == (5 if metric.startswith("queued") else 16)
                    and all(isfinite(x) and x > 0 for x in row)
                    for row in vals
                ) and v == median(map(median, vals))
    assert sorted(seen) == sorted((m, s) for m in (17, 33, 129) for s in (1731131, 1744213))
