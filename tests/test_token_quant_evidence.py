"""Replay gates, accumulator checks, source identity and shuffled measurements offline."""

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from random import Random
from statistics import median

from tools.verify_sources import verify_reports

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/token-quant-gb10.json"


def test_quantizer_encoding_groups_and_cuda_artifact():
    d = json.loads(REPORT.read_bytes())
    assert not d["smoke"] and verify_reports([REPORT])["distinct_archives"] == 1
    enc = d["encodings"]
    assert len(enc) == 6
    for dtype, finite in [("torch.float16", 63488), ("torch.bfloat16", 65280)]:
        vs = {g["order"]: g for g in enc if g["dtype"] == dtype}
        assert set(vs) == {"ordered", "shuffled", "finite_only"}
        for order, g in vs.items():
            assert g["count"] == (finite if order == "finite_only" else 65536) and g["width"] == 257
            assert set(g["cases"]) == {"zero", "opposite", "self", "random"}
            assert all(
                v["different_bytes"] == [0, 0, 0] and len(v["reference_sha256"]) == 3
                for v in g["cases"].values()
            )
        assert len({g["input_sha256"] for g in vs.values()}) == 3
    artifact = (ROOT / "examples/generated/token-quant/residual_bf16_33x1536.cu").read_bytes()
    for c in d["cases"]:
        if c["label"] == "hidden":
            assert c["fused_regions"][0]["cuda_sha256"] == sha256(artifact).hexdigest()


def test_quantizer_panel_admission_and_chronology():
    d = json.loads(REPORT.read_bytes())
    rows = d["cases"]
    assert len(rows) == 10
    seen = set()
    input_ids = set()
    weights = set()
    for c in rows:
        key = (c["seed"], c["label"])
        assert key not in seen
        seen.add(key)
        assert (
            c["input_offsets"] == [0, 1, 3]
            and len(c["inputs_sha256"]) == len(c["references_sha256"]) == 3
        )
        ids = {v for pair in c["inputs_sha256"] for v in pair}
        assert len(ids) == 6 and not (ids & input_ids)
        input_ids |= ids
        projected = bool(c["channels"])
        width = 5 if projected else 3
        assert all(len(ref) == width for ref in c["references_sha256"])
        if projected:
            assert c["cpu_int32_oracle"] == [out[3] for out in c["references_sha256"]]
            assert (
                set(c["parameters_sha256"]) == {"weight", "channel_scale"}
                and c["shape"][-1] * 127**2 < 2**31
            )
            w = c["parameters_sha256"]["weight"]
            assert w not in weights
            weights.add(w)
        else:
            assert not c["cpu_int32_oracle"] and not c["parameters_sha256"]
        (site,) = [r for r in c["fused_regions"] if r.get("kind") == "residual_token_quant"]
        assert (
            site["output_roles"] == ["residual", "codes", "scale"]
            and site["shape"] == c["shape"]
            and len(site["ops"]) == 10
        )
        assert not any(r.get("kind") == "residual_token_quant" for r in c["mixed_regions"])
        gates = c["gates"]
        assert set(gates) == {"eager", "aten", "mixed", "inductor", "cast_inductor", "fused"}
        passed = []
        for name, g in gates.items():
            if "error" in g:
                assert name in ("inductor", "cast_inductor") and not g["admitted"]
                continue
            assert len(g["outputs"]) == len(g["first_code_witness"]) == 3
            for outs, x in zip(g["outputs"], g["first_code_witness"], strict=True):
                assert len(outs) == width
                assert (x is None) == (outs[1]["different_bytes"] == 0)
                if x is not None:
                    assert (
                        x["eager_code"] == x["round_even_from_div"]
                        and x["eager_code"] != x["actual_code"]
                    )
            ok = g["fresh_and_retained"] and all(
                q["different_bytes"] == 0 and q["geometry_match"] for r in g["outputs"] for q in r
            )
            assert g["admitted"] == ok
            if ok:
                passed.append(name)
        t = c["timing"]
        assert (
            {"fused", "eager", "aten"} <= set(passed)
            and set(passed) == set(t["profiles"])
            and len(t["windows"]) == 7
        )
        rng = Random(c["seed"] + c["shape"][-1] + c["channels"])
        for window in t["windows"]:
            order = passed.copy()
            rng.shuffle(order)
            assert window["order"] == order
        for method in ("device", "host"):
            assert t["median_us"][method].keys() == set(passed)
            for name in passed:
                samples = [w[method][name]["samples_us"] for w in t["windows"]]
                assert all(
                    len(v) == (7 if method == "device" else 15)
                    and all(isfinite(x) and x > 0 for x in v)
                    for v in samples
                )
                assert t["median_us"][method][name] == median(map(median, samples))
        assert all(
            p["count"] == sum(p["kernel_names"].values()) and p["calls"] == 1
            for p in t["profiles"].values()
        )
        p = t["profiles"]["fused"]
        assert sum(n for k, n in p["kernel_names"].items() if "residual_token_quant" in k) == 1
    assert seen == {
        (s, n) for s in (279239, 317419) for n in ("odd", "hidden", "wide", "linear2d", "linear3d")
    }
