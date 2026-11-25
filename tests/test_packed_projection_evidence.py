"""Verify packed fanout admission, integer witnesses, ownership, clocks and profile attribution."""

from hashlib import sha256
import json
from math import isfinite, prod
from pathlib import Path
from random import Random
from statistics import median
from tools.verify_sources import verify_reports

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/packed-projection-gb10.json"


def sample(xs, num):
    assert len(xs) == num and all(v > 0 and isfinite(v) for v in xs)
    return median(xs)


def test_packed_fanout_evidence():
    d = json.loads(REPORT.read_bytes())
    assert not d["smoke"] and verify_reports([REPORT])["distinct_archives"] == 1
    keys = set()
    inputs = set()
    weights = set()
    cuda = {}
    for c in d["cases"]:
        label, seed = c["label"], c["seed"]
        key = label, seed
        assert key not in keys
        keys.add(key)
        m = prod(c["shape"][:-1])
        k = c["shape"][-1]
        ns = c["widths"]
        total = sum(ns)
        assert len(ns) == 3
        iq = [h for r in c["inputs_sha256"] for h in r]
        assert len(iq) == len(set(iq)) == 20 and inputs.isdisjoint(iq)
        inputs.update(iq)
        w = c["source_weights_sha256"]
        wh = [w[f"weight_{j}"] for j in range(3)]
        assert len(w) == 6 and len(set(wh)) == 3 and weights.isdisjoint(wh)
        weights.update(wh)
        refs = c["reference_sha256"]
        assert (
            len(refs) == 8
            and all(len(r) == 9 for r in refs)
            and [r[3::2] for r in refs] == c["cpu_int32_sha256"]
        )
        expect = 4 * m * k + k * total + 4 * total
        for name, nrhs in [("grouped", 3), ("packed", 1)]:
            cp = c["captures"][name]
            md = cp["metadata"]
            assert (
                cp["private_rhs_count"] == nrhs
                and cp["actual_rhs_strides"] == [[1, k]] * nrhs
                and cp["logical_band_sha256"] == wh
            )
            assert (
                md["int8_rhs_layout"] == ("column_major" if name == "grouped" else "packed_fanout")
                and md["staging_and_state_bytes"] == expect
                and md["input_bytes"] == 4 * m * k
            )
            assert len(md["output_bindings"]) == 9 and all(
                b["source"] == "copy" for b in md["output_bindings"]
            )
            assert sum(v["bytes"] for v in md["frozen_int8_weights"]) == k * total
            typed = [v["kind"] for v in md["regions"] if "kind" in v]
            assert typed == [
                "residual_token_quant",
                "grouped_int32_epilogue" if name == "grouped" else "packed_int32_projection",
            ]
            setup = c["setup"][name]
            assert (
                setup["wall_ms"] > 0
                and setup["peak_allocator_increase_bytes"]
                >= setup["retained_allocator_delta_bytes"]
            )
            if name == "packed":
                (rec,) = [r for r in md["regions"] if r.get("kind") == "packed_int32_projection"]
                assert (
                    rec["gemm"] == "aten._int_mm.default"
                    and rec["shape"] == [m, k, total]
                    and rec["snapshot_stride"] == [1, k]
                )
                assert rec["branches"] == [
                    dict(shape=[m, n], output_roles=["acc", "output"]) for n in ns
                ]
                assert [v["packed_column_offset"] for v in rec["source_weights"]] == [
                    0,
                    ns[0],
                    ns[0] + ns[1],
                ] and [v["name"] for v in rec["source_weights"]] == [
                    f"weight_{i}" for i in range(3)
                ]
                h = rec["cuda_sha256"]
                assert cuda.setdefault(label, h) == h
                if label == "h17":
                    assert (
                        h
                        == sha256(
                            (
                                ROOT / "examples/generated/token-quant/packed_17x128_64_64.cu"
                            ).read_bytes()
                        ).hexdigest()
                    )
        g = c["gates"]
        assert set(g) == {
            "eager",
            "aten",
            "grouped_once",
            "packed_once",
            "grouped_resident",
            "packed_resident",
            "inductor",
            "cast_inductor",
        }
        for n, row in g.items():
            if "error" in row:
                assert n.endswith("inductor") and not row["admitted"]
                continue
            ys = row["outputs"]
            assert len(ys) == 8 and all(len(y) == 9 for y in ys)
            assert row["admitted"] == (
                row["fresh"]
                and all(v["geometry_match"] and v["different_bytes"] == 0 for y in ys for v in y)
            )
            assert [v is None for v in row["code_witnesses"]] == [
                y[1]["different_bytes"] == 0 for y in ys
            ]
        keep = [n for n, v in g.items() if v["admitted"]]
        assert set(g) - {"inductor", "cast_inductor"} <= set(keep)
        methods = [n for n in keep if not n.endswith("_resident")]
        direct = c["ordinary"]
        assert len(direct["windows"]) == 7 and set(direct["profiles"]) == set(methods)
        rng = Random(seed + m + 127)
        for win in direct["windows"]:
            order = methods.copy()
            rng.shuffle(order)
            assert win["order"] == order
        for n in methods:
            for mode, num in [("device", 7), ("host", 15)]:
                assert direct["median_us"][mode][n] == median(
                    sample(win[mode][n]["samples_us"], num) for win in direct["windows"]
                )
        for n, count, marker in [
            ("packed_once", 3, "packed_int32_fanout"),
            ("grouped_once", 5, "grouped_int32_epilogue"),
        ]:
            p = direct["profiles"][n]
            names = p["kernel_names"]
            assert p["calls"] == 1 and p["count"] == sum(names.values()) == count
            for match in [marker, "residual_token_quant"]:
                assert sum(v for k, v in names.items() if match in k) == 1
        serve = c["serving"]
        assert (
            len(serve["windows"]) == 7
            and serve["seed"] == seed + m + 229
            and set(serve["median_us"]) == set(keep)
        )
        rng = Random(serve["seed"])
        lens = {
            k: 16
            for k in ["fixed_sync_us", "rotating_sync_us", "fixed_submit_us", "rotating_submit_us"]
        }
        lens["queued8_us_per_forward"] = 5
        for win in serve["windows"]:
            order = sorted(keep)
            rng.shuffle(order)
            assert win["order"] == order and set(win["samples"]) == set(keep)
        for n, v in serve["median_us"].items():
            assert set(v) == set(lens)
            for metric, num in lens.items():
                assert v[metric] == median(
                    sample(win["samples"][n][metric], num) for win in serve["windows"]
                )
    assert keys == {(label, s) for label in ("h17", "b33", "b129") for s in (1280471, 1294903)}
