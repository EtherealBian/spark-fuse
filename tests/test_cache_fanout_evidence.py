"""Tie one-versus-two-copy measurements to real Qwen graph contracts and raw windows."""

import json
from hashlib import sha256
from pathlib import Path
from statistics import median
from tools.verify_sources import verify_reports

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    p = ROOT / "reports" / path
    assert verify_reports([p])["verified_reports"] == 1
    return json.loads(p.read_text())


def test_cache_fanout_provenance_and_exact_layer_gate():
    d = read("qwen-decode-cache-fanout-gb10.json")
    sl = read("qwen-cache-fanout-slice-gb10.json")
    assert sl["passed"] and d["passed"] and d["fanout_panel"]
    assert all(
        sl[k] == d[k]
        for k in (
            "torch",
            "device",
            "transformers",
            "seeds",
            "timing_seed",
            "weight_seed",
            "export_seed",
        )
    )
    seeds = d["seeds"]
    assert len(set(seeds)) == 5 and d["timing_seed"] == seeds[-1] and d["export_seed"] not in seeds
    assert [c["initial_prefix"] for c in d["cases"]] == [31, 255, 1024, 1535, 2047]
    pieces = {(c["prefix"], c["step"]): c for c in sl["cases"]}
    assert set(pieces) == {(p, i) for p in (31, 1024, 2047) for i in (1, 2)}
    for c in d["cases"]:
        prefix = c["initial_prefix"]
        assert c["passed"]
        allowed = {
            k for k, r in c["backends"].items() if all(t["passed"] for t in r["trajectories"])
        }
        assert {"fanout", "separate", "eager", "exported_aten"} <= allowed
        for name, r in c["backends"].items():
            assert (
                r["passed"] == (name in allowed) == ("timing" in r)
                and [t["seed"] for t in r["trajectories"]] == seeds
            )
            if name in ("fanout", "separate"):
                for t in r["trajectories"]:
                    assert len(t["steps"]) == 2
                    for step in t["steps"]:
                        assert (
                            step["passed"]
                            and step["new_storage"]
                            and step["input_unchanged"]
                            and step["prefix_exact"]
                            and not step["masked_nonzero"]
                        )
                        assert (
                            step["output_strides"] == step["expected_strides"]
                            and len(step["outputs"]) == 4
                            and all(x["unequal_bits"] == 0 for x in step["outputs"].values())
                        )
            for t in r.get("timing", []):
                assert len(t["windows"]) == 7 and all(
                    t[m + "_us"] == median(median(w[m]["samples_us"]) for w in t["windows"])
                    for m in ("device", "host")
                )
        for i, gr in enumerate(c["graphs"], 1):
            size = prefix + i
            recs = {k: gr["partitions"][k]["records"] for k in ("separate", "fanout")}
            # Apart from the copy island, every fused scalar kernel must retain the same CUDA source.
            assert [
                r["cuda_sha256"] for r in recs["separate"] if "kind" not in r and "module" in r
            ] == [r["cuda_sha256"] for r in recs["fanout"] if "kind" not in r and "module" in r]
            assert all(
                gr["partitions"][k]["fx"].count("target=torch.ops.aten.matmul.default") == 2
                for k in recs
            )
            sep = [
                r
                for r in recs["separate"]
                if r.get("kind") in ("paired_append", "paired_head_repeat")
            ]
            fan = [r for r in recs["fanout"] if r.get("kind") == "cache_fanout"]
            assert len(sep) == 2 and len(fan) == 1
            for r in sep + fan:
                pub = ROOT / f"examples/generated/cache-fanout/p{prefix}-step{i}-{r['kind']}.cu"
                assert sha256(pub.read_bytes()).hexdigest() == r["cuda_sha256"]
            assert (
                fan[0]["logical_shape"] == [2, 2, size - 1, 1, 128]
                and fan[0]["query_groups"] == 6
                and len(fan[0]["live_outs"]) == 4
            )
            assert (
                c["backends"]["fanout"]["timing"][i - 1]["profile"]["count"] + 1
                == c["backends"]["separate"]["timing"][i - 1]["profile"]["count"]
            )
            if (prefix, i) not in pieces:
                continue
            p = pieces[prefix, i]
            assert (
                p["passed"] and p["export_graph_sha256"] == sha256(gr["aten"].encode()).hexdigest()
            )
            assert [r["cuda_sha256"] for r in p["builds"]["separate"]] == [
                r["cuda_sha256"] for r in sep
            ] and p["builds"]["fanout"][0]["cuda_sha256"] == fan[0]["cuda_sha256"]
            assert [s["seed"] for s in p["samples"]] == seeds
            for s in p["samples"]:
                assert len(s["whole_layer"]) == len(s["cache_and_gemm_operands"]) == 4 and all(
                    x["passed"] for x in s["whole_layer"] + s["cache_and_gemm_operands"]
                )
                assert [o["shape"] for o in s["outputs"]] == [[2, 2, size, 128]] * 2 + [
                    [2, 12, size, 128]
                ] * 2
                for row in s["methods"].values():
                    if "error" not in row:
                        assert row["passed"] == (
                            row["fresh"]
                            and row["unchanged"]
                            and len(row["outputs"]) == 4
                            and all(
                                v["stride_match"] and not v["unequal_bf16"] for v in row["outputs"]
                            )
                        )
            names = {
                k
                for k in p["samples"][0]["methods"]
                if all(s["methods"][k]["passed"] for s in p["samples"])
            }
            assert {"separate", "fanout", "eager"} <= names and set(p["profiles"]) == names
            assert len(p["windows"]) == 7 and all(
                len(w["order"]) == len(names) and set(w["order"]) == names for w in p["windows"]
            )
            for metric in ("host", "device"):
                assert p["median_us"][metric] == {
                    k: median(median(w[metric][k]["samples_us"]) for w in p["windows"])
                    for k in names
                }
            total = 2 * 2 * 2 * size * 128 * 2
            assert p["scheduled_payload_bytes"] == dict(
                separate=dict(read=2 * total, write=7 * total),
                fanout=dict(read=total, write=7 * total),
            )
            assert p["profiles"]["separate"]["count"] == 2 and p["profiles"]["fanout"]["count"] == 1
