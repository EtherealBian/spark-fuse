"""Offline reconstruction of a replay-only, frozen-projection ablation."""

from pathlib import Path
from statistics import median
from hashlib import sha256
from math import isfinite
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit import require
from metrics import p95, summarize

METHODS = {"two_eager", "two_fused", "packed_eager", "packed_fused", "inductor"}
PAIRS = (
    ("packed_fused", "two_fused"),
    ("packed_eager", "two_eager"),
    ("packed_fused", "packed_eager"),
    ("packed_fused", "inductor"),
)


def audit_packing(j, original):
    require(
        j["format"] == "sparkfuse.pretrained-projections.v1"
        and all(
            j[k] == original[k]
            for k in (
                "model",
                "revision",
                "weights_sha256",
                "device",
                "torch",
                "cuda",
                "transformers",
            )
        ),
        "checkpoint or environment",
    )
    require(
        j["layers"] == 24
        and j["weight_pack_bytes"] == 24 * 2 * 4864 * 896 * 2
        and isfinite(j["pack_and_weight_check_ms"])
        and j["pack_and_weight_check_ms"] > 0,
        "packed weight budget",
    )
    require(
        j["gate_cuda_sha256"] == {"1": original["sparkfuse_gate_source_sha256"]["1"]},
        "gate changed",
    )
    require(
        j["inductor_graphs_after_setup"] == j["inductor_graphs_after_windows"] > 0,
        "Inductor compile inside timing",
    )
    require(
        set(j["driver_sha256"])
        == {"run.py", "model.py", "metrics.py", "projection.py", "projection_panel.py"},
        "missing code fingerprint",
    )
    for f, h in j["driver_sha256"].items():
        require(sha256(Path(__file__).with_name(f).read_bytes()).hexdigest() == h, "driver digest")
    require(set(j["cases"]) == {"32", "128"}, "missing length")
    answer = {}
    for n, c in j["cases"].items():
        original_inputs = {tuple(p) for p in original["cases"][n]["inputs"]}
        inputs = c["inputs"]
        require(
            len(inputs) == 4
            and len(set(map(tuple, inputs))) == 4
            and all(len(p) == int(n) and tuple(p) not in original_inputs for p in inputs),
            "independent held-out prompts",
        )
        rows = c["windows"]
        require(len(rows) == 12, "paired block count")
        for i, r in enumerate(rows):
            require(
                r["prompt"] == i % 4 + 1 and len(r["order"]) == 5 and set(r["order"]) == METHODS,
                "paired order",
            )
            for m in METHODS:
                require(len(r[m]["decode_ms"]) == 8, "fixed token budget")
                p95([r[m]["ttft_ms"], *r[m]["decode_ms"]])
        require(set(c["validation"]) == METHODS and set(c["summary"]) == METHODS, "missing methods")
        elig = {}
        for m in METHODS:
            vals = c["validation"][m]
            require(len(vals) == 4, "token evidence")
            gates = []
            for i, v in enumerate(vals):
                ref = c["validation"]["two_eager"][i]["reference"]
                a = v["teacher_forced_logits"]
                require(
                    len(ref) == len(v["generated"]) == len(a) == 9 and ref == v["reference"],
                    "eager reference",
                )
                require(
                    all(isfinite(s[k]) for s in a for k in ("mean_abs", "max_abs", "cosine"))
                    and all(
                        0 <= s["mean_abs"] <= s["max_abs"] and -1.00001 <= s["cosine"] <= 1.00001
                        for s in a
                    ),
                    "logit metrics",
                )
                good = v["generated"] == ref and all(
                    s["max_abs"] <= 1 and s["cosine"] >= 0.999 for s in a
                )
                require(good == v["passed"], "eligibility gate")
                gates.append(good)
            elig[m] = all(gates)
            ds = [x for r in rows for x in r[m]["decode_ms"]]
            require(
                c["summary"][m] == dict(eligible=elig[m], p50_ms=median(ds), p95_ms=p95(ds)),
                "P50/P95 reconstruction",
            )
        pairs = [summarize(rows, a, b, elig[a] and elig[b]) for a, b in PAIRS]
        require(pairs == c["comparisons"], "paired decision mismatch")
        require(
            set(c["uncaptured_profile"]) == {"two_fused", "packed_fused"}
            and all(p["kernel_events"] > 0 for p in c["uncaptured_profile"].values()),
            "ablation trace",
        )
        answer[n] = dict(
            eligible=elig, comparisons=pairs, requests=12 * 5, timed_decode_tokens=12 * 5 * 8
        )
    met = all(c["comparisons"][-1]["target_met"] for c in answer.values())
    require(j["deployment_target_met"] == met, "cross-length target")
    return dict(
        format="sparkfuse.pretrained-projections-audit.v1", cases=answer, deployment_target_met=met
    )


if __name__ == "__main__":
    p, q = map(Path, sys.argv[1:])
    res = audit_packing(json.loads(p.read_bytes()), json.loads(q.read_bytes()))
    res["report_sha256"] = sha256(p.read_bytes()).hexdigest()
    res["original_report_sha256"] = sha256(q.read_bytes()).hexdigest()
    print(json.dumps(res, indent=2, sort_keys=True))
