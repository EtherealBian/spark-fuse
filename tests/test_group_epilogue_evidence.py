"""Offline verification of grouped tail numerical gates, launch attribution and shuffled samples."""

from hashlib import sha256
import json
from math import isfinite, prod
from pathlib import Path
from random import Random
from statistics import median
from tools.verify_sources import verify_reports

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports/grouped-epilogue-gb10.json"


def test_grouped_int32_three_branch_evidence():
    d = json.loads(R.read_bytes())
    assert not d["smoke"] and verify_reports([R])["distinct_archives"] == 1
    seen = set()
    ref_digests = set()
    weights = set()
    source_by_shape = {}
    for c in d["cases"]:
        seed, label, scope = c["seed"], c["label"], c["scope"]
        key = seed, label, scope
        assert key not in seen
        seen.add(key)
        full = scope == "fanout"
        m = prod(c["shape"][:-1])
        widths = c["widths"]
        assert len(widths) == 3
        assert len(c["inputs_sha256"]) == 4 and all(
            len(r) == (2 if full else 7) for r in c["inputs_sha256"]
        )
        assert all(k not in ref_digests for r in c["inputs_sha256"] for k in r)
        ref_digests.update(k for r in c["inputs_sha256"] for k in r)
        refs = c["reference_sha256"]
        assert len(refs) == 4 and all(len(y) == (9 if full else 3) for y in refs)
        if full:
            assert (
                c["cpu_int32_sha256"] == [y[3::2] for y in refs] and len(c["weights_sha256"]) == 6
            )
            w = c["weights_sha256"]["weight_0"]
            assert w not in weights
            weights.add(w)
        else:
            assert c["cpu_int32_sha256"] == [] and not c["weights_sha256"]
        gates = c["gates"]
        assert set(gates) == {"eager", "aten", "separate", "grouped", "inductor", "cast_inductor"}
        for name, g in gates.items():
            if "error" in g:
                assert not g["admitted"] and name in ("inductor", "cast_inductor")
                continue
            assert len(g["outputs"]) == 4 and all(len(v) == len(refs[0]) for v in g["outputs"])
            assert g["admitted"] == (
                g["fresh"]
                and all(
                    r["geometry_match"] and r["different_bytes"] == 0
                    for v in g["outputs"]
                    for r in v
                )
            )
            if full:
                for y, ex in zip(g["outputs"], g["code_witnesses"], strict=True):
                    assert (ex is None) == (y[1]["different_bytes"] == 0)
        t = c["timing"]
        keep = [k for k, g in gates.items() if g["admitted"]]
        assert {"grouped", "separate", "aten", "eager"} <= set(keep) == set(t["profiles"])
        for name, recs in c["records"].items():
            kinds = [v.get("kind") for v in recs]
            assert (
                kinds.count("grouped_int32_epilogue") == (name == "grouped")
                and kinds.count("ordered_int32_epilogue") == (3 if name == "separate" else 0)
                and kinds.count("residual_token_quant") == int(full)
            )
            if name == "grouped":
                (group,) = [v for v in recs if v.get("kind") == "grouped_int32_epilogue"]
                assert (
                    group["branches"]
                    == [dict(shape=[m, n], output_roles=["output"]) for n in widths]
                    and len(group["live_outs"]) == 3
                    and len(group["live_ins"]) == 7
                )
                code = group["cuda_sha256"]
                assert source_by_shape.setdefault(label, code) == code
                if label == "h17":
                    assert (
                        code
                        == sha256(
                            (
                                ROOT / "examples/generated/token-quant/group_17x128_64_64.cu"
                            ).read_bytes()
                        ).hexdigest()
                    )
            p = t["profiles"][name]
            assert (
                p["count"]
                == (3 * int(full) + int(full) + (1 if name == "grouped" else 3))
                == sum(p["kernel_names"].values())
            )
            assert sum(
                n for k, n in p["kernel_names"].items() if "grouped_int32_epilogue" in k
            ) == int(name == "grouped")
        tseed = seed + m + (231 if full else 971)
        assert c["timing_seed"] == tseed
        rng = Random(tseed)
        assert len(t["windows"]) == 7
        for w in t["windows"]:
            names = keep.copy()
            rng.shuffle(names)
            assert w["order"] == names
        for mode, num in [("device", 7), ("host", 15)]:
            for name in keep:
                values = [w[mode][name]["samples_us"] for w in t["windows"]]
                assert all(len(s) == num and all(v > 0 and isfinite(v) for v in s) for s in values)
                assert t["median_us"][mode][name] == median(map(median, values))
    assert seen == {
        (seed, label, scope)
        for seed in (1191169, 1201331)
        for label in ("h17", "b33", "b129")
        for scope in ("tails", "fanout")
    }
