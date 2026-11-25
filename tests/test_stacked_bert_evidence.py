"""Verify persisted stack admission, distinct per-layer sources, launch census and raw clocks offline."""

import json
from pathlib import Path
from statistics import median


def test_full_stacked_encoder_evidence():
    root = Path(__file__).resolve().parents[1] / "reports"
    d = json.loads((root / "stacked-bert-gb10.json").read_bytes())
    assert not d["smoke"] and (root / d["provenance"]["archive"]).is_file()
    assert len(d["cases"]) == 8
    specs = {(2, 17, False), (2, 33, True), (2, 129, False), (4, 33, True)}
    seeds = {r["seed"] for r in d["cases"]}
    assert len(seeds) == 2
    assert {(r["seed"], r["depth"], r["m"], r["residual"]) for r in d["cases"]} == {
        (seed, *sp) for seed in seeds for sp in specs
    }
    for c in d["cases"]:
        n = c["depth"]
        refs = c["ref_sha256"]
        assert len(refs) == 8 and all(len(v) == 2 * n for v in refs)
        assert (
            c["oracle_request_indices"] == [0, 7]
            and len(c["input_sha256"]) == 10
            and len({a[0] for a in c["input_sha256"]}) == 10
        )
        assert len(c["qkv_cpu_oracle_by_layer"]) == n and c["legacy_refusal"]
        assert len(c["layerwise_metadata"]) == len(c["setup"]["layerwise_stages"]) == n
        meta = c["metadata"]
        assert len({r["staging_and_state_bytes"] for r in meta.values()}) == 1
        records = [
            r
            for r in meta["packed_groups"]["regions"]
            if r.get("kind") == "packed_int32_projection"
        ]
        assert len(records) == n and len(meta["packed_groups"]["frozen_int8_weights"]) == n
        names = []
        for i, (oracle, rec, sub) in enumerate(
            zip(c["qkv_cpu_oracle_by_layer"], records, c["layerwise_metadata"], strict=True)
        ):
            assert (
                rec["shape"] == [c["m"], 768, 2304]
                and rec["module"] == f"_sparkfuse_packed_projection_{i}"
                and rec["weight"] == f"_sparkfuse_packed_rhs_{i}"
            )
            assert [w["packed_column_offset"] for w in rec["source_weights"]] == [0, 768, 1536]
            assert len(sub["frozen_int8_weights"]) == 1
            for w in rec["source_weights"]:
                name = w["name"]
                assert (
                    name in c["source_state_sha256"]
                    and name.startswith(f"layers.{i}.")
                    and w["shape"] == [768, 768]
                )
                names.append(name)
            assert len(oracle) == 2
            for o, j in zip(oracle, c["oracle_request_indices"], strict=True):
                if i:
                    assert o["activation"] == refs[j][2 * (i - 1)]
                elif not c["residual"]:
                    assert o["activation"] == c["input_sha256"][j + 2][0]
                assert len(o["branches"]) == 3 and all(
                    len(b[k]) == 64 for b in o["branches"] for k in ("cpu_int32", "narrow")
                )
        assert (
            len(set(names)) == 3 * n
            and len({c["source_state_sha256"][name] for name in names}) == 3 * n
        )
        required = {"eager", "independent", "shared", "packed_groups", "layerwise_packed"}
        admitted = set()
        for name, g in c["gates"].items():
            leaves = []
            for p in (g["request"], g["ordinary"]):
                if p is None:
                    continue
                assert len(p["outputs"]) == 8 and all(len(row) == 2 * n for row in p["outputs"])
                ok = (
                    p["fresh"]
                    and p["container"]
                    and all(
                        t["geometry_match"] and not t["different_bytes"]
                        for row in p["outputs"]
                        for t in row
                    )
                )
                assert p["admitted"] == ok
                leaves.append(ok)
            assert g["admitted"] == all(leaves)
            if all(leaves):
                admitted.add(name)
        assert required <= admitted and set(c["gates"]) | set(c["build_failures"]) == required | {
            "inductor",
            "cast_inductor",
        }
        serve = c["serving"]
        assert set(serve["median_us"]) == admitted and len(serve["windows"]) == 7
        for w in serve["windows"]:
            assert (
                len(w["order"]) == len(admitted)
                and set(w["order"]) == set(w["samples"]) == admitted
            )
        for name, metrics in serve["median_us"].items():
            for k, v in metrics.items():
                windows = [w["samples"][name][k] for w in serve["windows"]]
                assert all(len(a) == (5 if k == "queued8_us_per_forward" else 16) for a in windows)
                assert v == median(median(a) for a in windows)
        ordinary = c["ordinary"]
        assert set(ordinary["profiles"]) == admitted - {"layerwise_packed"}
        for clock, times in ordinary["median_us"].items():
            for arm, t in times.items():
                assert t == median(median(w[clock][arm]["samples_us"]) for w in ordinary["windows"])
        profiles = ordinary["profiles"]
        assert profiles["shared"]["count"] - profiles["packed_groups"]["count"] == 2 * n
