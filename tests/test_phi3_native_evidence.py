"""Replay admission, sample aggregation and source identity without running a GPU benchmark."""

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import median

from tools.verify_sources import verify_reports

ROOT = Path(__file__).resolve().parents[1]
ALL = {"eager", "inductor", "base", "sliced"}
EXACT = ALL - {"inductor"}


def samples(timing, allowed):
    assert set(timing["profiles"]) == allowed
    ws = timing["windows"]
    assert len(ws) == 7
    for w in ws:
        assert len(w["order"]) == len(allowed) and set(w["order"]) == allowed
        for metric, size in [("device", 7), ("host", 15)]:
            assert set(w[metric]) == allowed
            for v in w[metric].values():
                assert len(v["samples_us"]) == size and all(
                    isfinite(z) and z > 0 for z in v["samples_us"]
                )
    for k in ("device", "host"):
        assert timing["median_us"][k] == {
            n: median(median(w[k][n]["samples_us"]) for w in ws) for n in allowed
        }


def graph_source(g, rd, t):
    assert "aten.slice" in g["aten"] and "aten.cat" in g["aten"]
    v = g["variants"]
    assert set(v) == {"base", "sliced"} and not any(
        z.get("kind") == "sliced_rotary_pair" for z in v["base"]["records"]
    )
    (hit,) = [z for z in v["sliced"]["records"] if z.get("kind") == "sliced_rotary_pair"]
    assert (hit["shape"], hit["heads"], hit["offsets"], hit["head_dim"], hit["rotary_dim"]) == (
        [2, t, 2048],
        [12, 2],
        [0, 1536],
        128,
        rd,
    )
    k = hit["cuda_sha256"]
    assert (
        sha256((ROOT / "examples/generated/sliced-rotary" / (k + ".cu")).read_bytes()).hexdigest()
        == k
    )
    return k


def layer_gate(c, *, methods=ALL, exact=EXACT):
    panel = c["trajectories"]
    assert set(panel) == methods
    for name, trajs in panel.items():
        assert [z["seed"] for z in trajs] == c["seeds"]
        for tr in trajs:
            for i, st in enumerate(tr["checks"]):
                if "error" in st:
                    assert name not in exact and not st["passed"]
                    continue
                outs = st["outputs"]
                assert set(outs) == {"hidden", "probabilities", "key", "value"}
                layout = st["expected_metadata"][-1]
                t = c["prefix"]
                if i == 0:
                    assert layout == dict(
                        stride=[t * 2048, 128, 2048, 1], offset=1792, storage_bytes=2 * t * 2048 * 2
                    )
                else:
                    length = t + sum(c.get("chunks", (1, 1))[:i])
                    assert layout == dict(
                        stride=[2 * length * 128, length * 128, 128, 1],
                        offset=0,
                        storage_bytes=2 * 2 * length * 128 * 2,
                    )
                assert st["passed"] == (
                    st["distinct_outputs"]
                    and st["input_cache_unchanged"]
                    and st["prefix_exact"]
                    and st["masked_nonzero"] == 0
                    and st["max_row_sum_error"] <= 0.004
                    and (name not in exact or st["metadata"] == st["expected_metadata"])
                    and all(
                        d["finite"]
                        and d["tolerance_pass"]
                        and (name not in exact or d["unequal_bits"] == 0)
                        for d in outs.values()
                    )
                )
            assert tr["passed"] == (
                len(tr["checks"]) == 3 and all(s["passed"] for s in tr["checks"])
            )
    allowed = {n for n, ts in panel.items() if all(t["passed"] for t in ts)}
    assert allowed == set(c["admitted_layer"]) and exact <= allowed
    return allowed


def test_phi3_native_same_graph_evidence():
    paths = [
        ROOT / "reports" / ("phi3-native-packed" + suffix + "-gb10.json")
        for suffix in ("", "-heldout")
    ]
    assert verify_reports(paths)["verified_reports"] == 2
    reps = [json.loads(p.read_text()) for p in paths]
    assert (
        len({r["weight_seed"] for r in reps}) == 2
        and len({r["provenance"]["source_tree_sha256"] for r in reps}) == 1
    )
    assert (
        reps[0]["weights"]["64"]["layer.self_attn.qkv_proj.weight"]
        != reps[1]["weights"]["64"]["layer.self_attn.qkv_proj.weight"]
    )
    for r in reps:
        assert r["passed"] and not r["verify_only"] and r["weights"]["64"] == r["weights"]["128"]
        assert [(c["rotary_dim"], c["prefix"]) for c in r["cases"]] == [
            (rd, t) for rd in (64, 128) for t in (127, 1024)
        ]
        for c in r["cases"]:
            assert c["passed"] and len(set(c["seeds"])) == 5 and r["export_seed"] not in c["seeds"]
            allowed = layer_gate(c)
            assert [h["stage"] for h in c["stages"]] == ["prefill", "decode1", "decode2"]
            for i, h in enumerate(c["stages"]):
                t = c["prefix"] if i == 0 else 1
                assert len(set(h["common_inputs_sha256"])) == (4 if i == 0 else 6)
                gr = h["graphs"]
                assert graph_source(gr["layer"], c["rotary_dim"], t) == graph_source(
                    gr["isolated"], c["rotary_dim"], t
                )
                if i or t <= 127:
                    assert not h["profile_only"]
                    samples(h["layer"], allowed)
                    profiles = h["layer"]["profiles"]
                else:
                    assert not h["layer"] and set(h["profile_only"]) == allowed
                    profiles = h["profile_only"]
                assert profiles["base"]["count"] - profiles["sliced"]["count"] == 11
                iso = h["isolated"]
                assert [z["seed"] for z in iso["rows"]] == c["seeds"][::2] and len(
                    {z["input_sha256"][0] for z in iso["rows"]}
                ) == 3
                for z in iso["rows"]:
                    assert len(z["cache_suffix"]) == 2 and all(
                        v["unequal_bits"] == 0 for v in z["cache_suffix"]
                    )
                    assert set(z["methods"]) == ALL
                    for n, m in z["methods"].items():
                        if "error" in m:
                            assert n == "inductor" and not m["passed"]
                            continue
                        assert len(m["tensors"]) == 3
                        assert m["passed"] == (
                            m["fresh_qk"]
                            and m["visible_v_alias"]
                            and m["input_unchanged"]
                            and all(
                                v["stride_match"]
                                and v["unequal_bits"] == 0
                                and v["layout"]["storage_offset"] == off
                                for v, off in zip(m["tensors"], (0, 0, 1792), strict=True)
                            )
                        )
                        assert n == "inductor" or m["passed"]
                if i < 2:
                    admitted = {
                        n for n in ALL if all(z["methods"][n]["passed"] for z in iso["rows"])
                    }
                    samples(iso, admitted)
                    assert [iso["profiles"][n]["count"] for n in ("base", "sliced")] == [12, 1]
                else:
                    assert set(iso) == {"rows"}
