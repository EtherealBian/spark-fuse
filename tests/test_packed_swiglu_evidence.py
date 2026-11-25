"""Recompute timing eligibility and connect whole-layer schedules to isolated kernel bytes."""

from hashlib import sha256
import json
from pathlib import Path
from statistics import median

from tools.verify_sources import verify_reports

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def windows(ws, profiles, summary, eligible):
    assert set(profiles) == eligible and len(ws) == 7
    for w in ws:
        assert set(w["order"]) == eligible and len(w["order"]) == len(eligible)
    for metric, size in (("device", 7), ("host", 15)):
        assert all(set(w[metric]) == eligible for w in ws)
        for n in eligible:
            assert all(len(w[metric][n]["samples_us"]) == size for w in ws)
        assert summary[metric] == {
            n: median(median(w[metric][n]["samples_us"]) for w in ws) for n in eligible
        }


def generated(r, packet, shape):
    assert r["kind"] == "packed_swiglu" and r["host_backend"] == "native"
    assert r["shape"] == shape and r["schedule"] == dict(max_packet_elements=packet, threads=128)
    key = r["cuda_sha256"]
    assert (
        sha256((ROOT / "examples/generated/packed-swiglu" / (key + ".cu")).read_bytes()).hexdigest()
        == key
    )
    return key


def valid(step, strict):
    if "error" in step:
        return False
    assert set(step["outputs"]) == {"hidden", "probabilities", "key", "value"}
    return (
        step["new_storage"]
        and step["input_unchanged"]
        and step["prefix_exact"]
        and step["masked_nonzero"] == 0
        and step["max_row_sum_error"] <= 0.004
        and all(
            o["tolerance_pass"] and (not strict or o["unequal_bits"] == 0)
            for o in step["outputs"].values()
        )
        and (not strict or step["output_strides"] == step["expected_strides"])
    )


def test_packed_projection_whole_layer_and_slice_evidence():
    old = json.loads((REPORTS / "qwen-mlp-projection-admission-gb10.json").read_bytes())["panels"]
    assert len(set(v["weight_sha256"][-1] for v in old)) == 2
    verified = []
    for suffix, prefixes, p in zip(("", "-heldout"), ([31, 1024, 2047], [1024]), old, strict=True):
        paths = [
            REPORTS / f"{name}{suffix}-gb10.json"
            for name in ("qwen-decode-packed-swiglu", "packed-swiglu-epilogue")
        ]
        verified.extend(paths)
        layer, epi = (json.loads(f.read_bytes()) for f in paths)
        assert layer["passed"] and not layer["verify_only"]
        assert all(
            layer[k] == epi[k]
            for k in ("device", "torch", "weight_seed", "export_seed", "timing_seed")
        )
        assert layer["weight_seed"] == p["weight_seed"]
        ck = layer["checkpoint"]
        assert ck["packed_sha256"] == epi["packed_sha256"] == p["weight_sha256"][-1]
        assert ck["old_projection_sha256"] == p["weight_sha256"][:2]
        assert ck["converted_mlp_state_keys"] == ["gate_up.weight", "down_proj.weight"]
        assert ck["packed_parameter_bytes"] == ck["original_projection_bytes"] == 2 * 17920 * 1536
        seeds = layer["seeds"]
        assert (
            len(seeds) == len(set(seeds)) == 5
            and layer["export_seed"] not in seeds
            and layer["timing_seed"] == seeds[-1]
        )
        decode = {}
        assert [(c["prefix"], c["step"]) for c in epi["cases"]] == [
            (31, "decode1"),
            (31, "prefill"),
            (127, "prefill"),
        ]
        for c in epi["cases"]:
            samples = c["samples"]
            assert (
                seeds == [s["seed"] for s in samples]
                and len({s["input_sha256"] for s in samples}) == 5
            )
            seq = 1 if c["step"] == "decode1" else c["prefix"]
            for s in samples:
                assert s["packed_tensor"]["shape"] == [2, seq, 17920]
                assert len(s["projections"]) == 2 and all(
                    t["unequal_bits"] == 0 for t in s["projections"]
                )
                assert set(s["outputs"]) == {"eager", "inductor", "scalar", "adaptive"}
            eligible = {
                n
                for n in samples[0]["outputs"]
                if all(
                    s["outputs"][n].get("unequal_bits") == 0 and s["outputs"][n]["stride_match"]
                    for s in samples
                )
            }
            assert eligible >= {"eager", "scalar", "adaptive"}
            windows(c["windows"], c["profiles"], c["median_us"], eligible)
            for sched, packet in (("scalar", 1), ("adaptive", 8)):
                key = generated(c["schedules"][sched], packet, [2, seq, 8960])
                assert c["profiles"][sched]["count"] == 1
                if seq == 1:
                    decode[sched] = key
        assert [c["prefix"] for c in layer["cases"]] == prefixes
        for c in layer["cases"]:
            assert c["passed"] and set(c["backends"]) == {
                "eager",
                "packed_eager",
                "two_linear",
                "packed_framework_gate",
                "packed_cuda_gate",
                "packed_scalar_gate",
                "inductor",
            }
            eligible = set()
            for name, b in c["backends"].items():
                assert [t["seed"] for t in b["trajectories"]] == seeds
                for t in b["trajectories"]:
                    assert all(s["passed"] == valid(s, name != "inductor") for s in t["steps"])
                    assert t["passed"] == (
                        len(t["steps"]) == 2 and all(s["passed"] for s in t["steps"])
                    )
                assert b["passed"] == all(t["passed"] for t in b["trajectories"])
                if b["passed"]:
                    eligible.add(name)
                else:
                    assert name == "inductor"
            assert len(c["graphs"]) == 2
            for g, ws, prof, md in zip(
                c["graphs"], c["windows"], c["profiles"], c["median_us"], strict=True
            ):
                windows(ws, prof, md, eligible)
                for path, sched, packet in (
                    ("packed_scalar_gate", "scalar", 1),
                    ("packed_cuda_gate", "adaptive", 8),
                ):
                    (r,) = [
                        r
                        for r in g["variants"][path]["records"]
                        if r.get("kind") == "packed_swiglu"
                    ]
                    assert generated(r, packet, [2, 1, 8960]) == decode[sched]
                    assert prof[path]["count"] == prof["packed_framework_gate"]["count"] - 1
                    assert (
                        sum(
                            v
                            for k, v in prof[path]["kernel_names"].items()
                            if f"packed_swiglu<{packet}>" in k
                        )
                        == 1
                    )
    assert verify_reports(verified)["verified_reports"] == 4
