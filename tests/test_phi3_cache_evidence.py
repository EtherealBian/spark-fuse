"""Cross-check two public same-export runs, including non-admitted comparator results."""

from hashlib import sha256
import json
from test_phi3_native_evidence import ROOT, layer_gate, samples
from tools.verify_sources import verify_reports

EXACT = {"eager", "sliced", "cache"}
METHODS = EXACT | {"inductor", "inductor_casts"}


def graph(graph, h, rd):
    assert "aten.cat" in graph["aten"] and "aten.slice" in graph["aten"]
    assert set(graph["variants"]) == {"sliced", "cache"}
    before = graph["variants"]["sliced"]["records"]
    after = graph["variants"]["cache"]["records"]
    assert sum(z.get("kind") == "sliced_cache_update" for z in before) == 0
    assert sum(z.get("kind") == "sliced_rotary_pair" for z in before) == 1
    assert sum(z.get("kind") == "sliced_rotary_pair" for z in after) == 0
    (rec,) = [z for z in after if z.get("kind") == "sliced_cache_update"]
    assert (
        rec["head_dim"] == 128
        and rec["rotary_dim"] == rd
        and rec["heads"] == [12, 2]
        and rec["offsets"] == [0, 1536, 1792]
    )
    assert rec["prefix"] == h["old_tokens"] and rec["history_strides"] == [
        z["stride"] for z in h["history_layout"]
    ]
    assert rec["shape"] == [2, h["new_tokens"], 2048] and rec["schedule"] == {
        "threads": 128,
        "max_packet_elements": 8,
    }
    key = rec["cuda_sha256"]
    assert (
        key
        == sha256(
            (ROOT / "examples/generated/sliced-cache" / (key + ".cu")).read_bytes()
        ).hexdigest()
    )
    return key


def test_sliced_cache_source_admission_and_windows():
    files = [ROOT / "reports" / ("phi3-sliced-cache" + s + "-gb10.json") for s in ("", "-heldout")]
    assert verify_reports(files)["verified_reports"] == 2
    reps = [json.loads(p.read_text()) for p in files]
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
        assert not r["default_emulate_precision_casts"] and r["comparator_options"] == {
            "inductor": {},
            "inductor_casts": {"emulate_precision_casts": True},
        }
        assert [(c["rotary_dim"], c["prefix"], c["chunks"]) for c in r["cases"]] == [
            (rd, n, z) for rd in (64, 128) for n in (127, 1024) for z in ([1, 1], [4, 3])
        ]
        for c in r["cases"]:
            assert c["passed"] and len(set(c["seeds"])) == 5 and r["export_seed"] not in c["seeds"]
            allowed = layer_gate(c, methods=METHODS, exact=EXACT)
            assert len(c["graphs"]) == 3 and len(c["updates"]) == 2
            pre = c["graphs"][0]
            assert (
                pre["isolated"] is None
                and pre["layer"]["variants"]["cache"] == pre["layer"]["variants"]["sliced"]
            )
            for i, h in enumerate(c["updates"]):
                old = c["prefix"] + sum(c["chunks"][:i])
                new = c["chunks"][i]
                future = new * (new - 1) // 2
                assert (
                    h["stage"],
                    h["old_tokens"],
                    h["new_tokens"],
                    h["future_masked_per_batch"],
                    h["masked_entries"],
                ) == (i + 1, old, new, future, 2 * (future + new))
                assert all(z["shape"] == [2, 2, old, 128] for z in h["history_layout"])
                kd, vd = h["history_layout"]
                dense = [2 * old * 128, old * 128, 128, 1]
                assert (kd["offset"], kd["stride"]) == (0, dense) and (
                    vd["offset"],
                    vd["stride"],
                ) == ((0, dense) if i else (1792, [old * 2048, 128, 2048, 1]))
                gs = c["graphs"][i + 1]
                assert graph(gs["layer"], h, c["rotary_dim"]) == graph(
                    gs["isolated"], h, c["rotary_dim"]
                )
                rows = h["isolated_rows"]
                assert (
                    h["timing_seed"] == c["seeds"][-1]
                    and [z["seed"] for z in rows] == c["seeds"][::2]
                    and len({z["inputs_sha256"][0] for z in rows}) == 3
                )
                hashes = h["layer_inputs_sha256"]
                assert rows[-1]["inputs_sha256"][1:] == [hashes[1], hashes[2], hashes[4], hashes[5]]
                for row in rows:
                    assert (
                        set(row["methods"]) == METHODS
                        and len(row["whole_hf_cache_oracle"]) == 2
                        and all(
                            z["unequal_bits"] == 0 and z["stride_match"]
                            for z in row["whole_hf_cache_oracle"]
                        )
                    )
                    for name, m in row["methods"].items():
                        if "error" in m:
                            assert name not in EXACT and not m["passed"]
                            continue
                        assert len(m["tensors"]) == 4 and m["expected_offset_and_span"][-1] == [
                            1792,
                            2 * new * 2048 * 2,
                        ]
                        assert m["passed"] == (
                            m["fresh"]
                            and m["raw_v_alias"]
                            and m["input_unchanged"]
                            and m["prefix_exact"]
                            and m["offset_and_span"] == m["expected_offset_and_span"]
                            and all(
                                z["unequal_bits"] == 0 and z["stride_match"] for z in m["tensors"]
                            )
                        )
                        assert name not in EXACT or m["passed"]
                iso = {k for k in METHODS if all(z["methods"][k]["passed"] for z in rows)}
                assert set(h["admitted_isolated"]) == iso
                for key, ok in [("layer", allowed), ("isolated", iso)]:
                    tm = h["timing"][key]
                    samples(tm, ok)
                    pr = tm["profiles"]
                    assert pr["sliced"]["count"] - pr["cache"]["count"] == 2
                assert h["timing"]["isolated"]["profiles"]["cache"]["count"] == 1
