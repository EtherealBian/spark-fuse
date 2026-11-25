"""Audit real-family attribution, held-out rotations, state contracts and latency windows."""

import json
from pathlib import Path
from statistics import median


def test_decode_family_composition_evidence():
    root = Path(__file__).resolve().parents[1] / "reports"
    d = json.loads((root / "decoder-family-gb10.json").read_bytes())
    assert not d["smoke"] and (root / d["provenance"]["archive"]).is_file()
    sh = {
        "qwen2": ("Qwen2DecoderLayer", 8, 2, 64),
        "llama": ("LlamaDecoderLayer", 8, 4, 64),
        "mistral": ("MistralDecoderLayer", 6, 2, 128),
    }
    assert len(d["cases"]) == 12 and {
        (c["family"], c["masked"], c["seed"]) for c in d["cases"]
    } == {(f, m, s) for f in sh for m in (True, False) for s in (3734911, 3745561)}
    for c in d["cases"]:
        p = c["site"]["plan"]
        assert (c["layer_type"], p["heads"], p["kv_heads"], p["width"]) == sh[c["family"]]
        assert (
            c["qualification"] == [17, 65]
            and c["past"] == [2, 15, 30, 31, 125, 255, 513, 2045]
            and c["steps"] == 4
        )
        assert len({xs[0] for t in c["inputs"] for xs in t}) == 32 and len(c["weights"]) >= 9
        assert len(c["variants"]["attention_only"]["binaries"]) == 2
        assert not c["variants"]["hybrid"]["rotary"]
        (r,) = c["variants"]["hybrid_rope"]["rotary"]
        assert (
            len(r["ops"]) == 14
            and len(r["live_outs"]) == 2
            and c["binaries"][c["variants"]["hybrid_rope"]["binaries"][r["module"]]][
                "source_sha256"
            ]
            == r["cuda_sha256"]
        )
        assert all(k == v["binary_sha256"] for k, v in c["binaries"].items())
        for n, z in c["variants"].items():
            assert z["setup"]["wall_ms"] > 0
            ks = z["binaries"] if isinstance(z["binaries"], list) else z["binaries"].values()
            assert all(k in c["binaries"] for k in ks)
        for start, rows in zip(c["past"], c["held_out"], strict=True):
            assert len(rows) == 4
            for j, ck in enumerate(rows):
                assert (
                    ck["sequence"] == start + j + 1
                    and ck["attention"]["structure"]
                    and not ck["attention"]["outside_tolerance"]
                    and len(ck["fp64_sha256"]) == 64
                )
                assert [v["shape"] for v in ck["rotary"]] == [
                    [2, h, 1, p["width"]] for h in (p["heads"], p["kv_heads"])
                ]
                assert all(
                    v["stride_match"]
                    and ((v["sha256"] == v["ref_sha256"]) == (v["different_bytes"] == 0))
                    for v in ck["rotary"]
                )
        accepted = set()
        assert ("inductor_dynamic" in c["gates"]) != ("inductor_dynamic" in c["build_failures"])
        for k, g in c["gates"].items():
            assert len(g["steps"]) == 8 and all(len(row) == 4 for row in g["steps"])
            flat = [z for row in g["steps"] for z in row]
            for z in flat:
                assert z["within_tolerance"] == all(o["outside"] == 0 for o in z["outputs"]) and z[
                    "exact_cache"
                ] == all(o["sha256"] == o["ref_sha256"] for o in z["outputs"][1:])
                assert len(z["outputs"]) == 3
            assert g["admitted"] == (
                g["all_live"]
                and all(
                    z["within_tolerance"] and z["geometry"] and z["fresh"] and z["prefix_bitwise"]
                    for z in flat
                )
            )
            assert g["exact_caches"] == (g["admitted"] and all(z["exact_cache"] for z in flat))
            if g["admitted"]:
                accepted.add(k)
        assert {"hf_eager", "attention_only", "exported_aten"} <= accepted
        r = c["rotating"]
        assert set(r["median_us"]) == accepted and len(r["windows"]) == 7
        for k, m in r["median_us"].items():
            for mode, t in m.items():
                assert t == median(median(w["samples"][k][mode]) for w in r["windows"])
        for h in c["hot"].values():
            assert set(h["profiles"]) == accepted and len(h["windows"]) == 7
            for mode in ("host", "device"):
                for k, t in h["median_us"][mode].items():
                    assert t == median(median(w[mode][k]["samples_us"]) for w in h["windows"])
