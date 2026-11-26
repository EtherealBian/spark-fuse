"""Recompute the deployment decision without CUDA, torch, transformers or weights."""

from hashlib import sha256
from math import isfinite
from pathlib import Path
from statistics import median
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import p95, summarize

METHODS = ("eager", "sparkfuse", "inductor", "eager_replay", "sparkfuse_replay", "inductor_replay")
PAIRS = (
    ("sparkfuse", "inductor"),
    ("sparkfuse_replay", "inductor_replay"),
    ("sparkfuse", "eager"),
    ("sparkfuse_replay", "eager_replay"),
    ("sparkfuse_replay", "sparkfuse"),
    ("eager_replay", "eager"),
    ("inductor_replay", "inductor"),
)


def require(test, why):
    if not test:
        raise ValueError(why)


def audit(j):
    require(
        j["format"] == "sparkfuse.pretrained-qwen.v1"
        and j["revision"] == "c540970f9e29518b1d8f06ab8b24cba66ad77b6d"
        and j["weights_sha256"]
        == "130282af0dfa9fe5840737cc49a0d339d06075f83c5a315c3372c9a0740d0b96",
        "different model/revision",
    )
    require(
        (
            j["batch"],
            j["layers"],
            j["generated_tokens"],
            j["max_cache_len"],
            j["dtype"],
            j["attention"],
        )
        == (1, 24, 9, 137, "bfloat16", "sdpa"),
        "workload changed",
    )
    require(
        set(j["cases"]) == {"32", "128"}
        and set(j["sparkfuse_gate_source_sha256"]) == {"1", "32", "128"},
        "shape coverage",
    )
    require(set(j["driver_sha256"]) == {"model.py", "run.py", "metrics.py"}, "missing driver")
    for f, h in j["driver_sha256"].items():
        require(
            sha256(Path(__file__).with_name(f).read_bytes()).hexdigest() == h, "driver digest: " + f
        )
    require(
        j["inductor_graphs_after_setup"] == j["inductor_graphs_after_windows"] > 0,
        "Inductor compiled new graphs after warmup",
    )
    out = {}
    for n, c in j["cases"].items():
        rows = c["windows"]
        require(len(rows) >= 12 and len(rows) % 3 == 0, "not a release-length panel")
        require(
            len(c["inputs"]) == 3
            and all(len(v) == int(n) for v in c["inputs"])
            and len({tuple(v) for v in c["inputs"]}) == 3,
            "prompt coverage",
        )
        for i, r in enumerate(rows):
            require(
                set(r["order"]) == set(METHODS)
                and len(r["order"]) == 6
                and r["prompt"] == 1 + i % 3,
                "incomplete paired block",
            )
            for m in METHODS:
                require(len(r[m]["decode_ms"]) == 8, "token count")
                p95([r[m]["ttft_ms"], *r[m]["decode_ms"]])
        require(set(c["validation"]) == set(METHODS), "method coverage")
        elig = {}
        for m in METHODS:
            vs = c["validation"][m]
            require(len(vs) == 3, "prompt evidence")
            gates = []
            for i, v in enumerate(vs):
                steps = v["teacher_forced_logits"]
                ref = c["validation"]["eager"][i]["generated"]
                require(
                    len(ref) == len(v["generated"]) == len(steps) == 9 and ref == v["reference"],
                    "reference tokens",
                )
                require(
                    all(isfinite(s[k]) for s in steps for k in ("cosine", "max_abs", "mean_abs"))
                    and all(
                        0 <= s["mean_abs"] <= s["max_abs"] and -1.00001 <= s["cosine"] <= 1.00001
                        for s in steps
                    ),
                    "invalid logits metrics",
                )
                good = v["generated"] == ref and all(
                    s["cosine"] >= 0.999 and s["max_abs"] <= 1 for s in steps
                )
                require(v["passed"] == good, "logits or token gate")
                gates.append(good)
            elig[m] = all(gates)
            tt = [r[m]["ttft_ms"] for r in rows]
            dt = [d for r in rows for d in r[m]["decode_ms"]]
            expected = dict(
                eligible=elig[m],
                ttft_p50_ms=median(tt),
                ttft_p95_ms=p95(tt),
                decode_p50_ms=median(dt),
                decode_p95_ms=p95(dt),
            )
            require(expected == c["summary"][m], "summary not derived from timings")
        for m in ("eager", "inductor", "sparkfuse"):
            p = c["profile"][m]
            require(p["kernel_events"] > 0 and p["sum_event_us"] > 0 and p["top"], "no GPU profile")
        comps = [summarize(rows, a, b, elig[a] and elig[b]) for a, b in PAIRS]
        require(comps == c["comparisons"], "paired decision mismatch")
        out[n] = dict(
            eligible=elig,
            primary=comps[:2],
            requests=len(rows) * len(METHODS),
            timed_decode_tokens=len(rows) * len(METHODS) * 8,
        )
    target = all(c["target_met"] for v in out.values() for c in v["primary"])
    require(target == j["deployment_target_met"], "cross-length decision mismatch")
    return dict(
        format="sparkfuse.pretrained-qwen-audit.v1", cases=out, deployment_target_met=target
    )


if __name__ == "__main__":
    p = Path(sys.argv[1])
    r = audit(json.loads(p.read_bytes()))
    r["measured_report_sha256"] = sha256(p.read_bytes()).hexdigest()
    print(json.dumps(r, indent=2, sort_keys=True))
