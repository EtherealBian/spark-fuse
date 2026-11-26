"""Rebuild an exploratory evidence table from six independent raw job records (stdlib)."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from checkpoint7b import REPO, REV
from metrics import p95


def require(test, message):
    if not test:
        raise ValueError(message)


def check(r, n, backend):
    require(
        (
            r["format"],
            r["model"],
            r["revision"],
            r["backend"],
            r["length"],
            r["generated"],
            r["batch"],
            r["dtype"],
            r["attention"],
            r["cache_capacity"],
        )
        == (
            "sparkfuse.qwen2-7b.pilot.v1",
            REPO,
            REV,
            backend,
            n,
            128,
            1,
            "bfloat16",
            "sdpa",
            n + 128,
        ),
        "case identity",
    )
    require(set(r["validation"]) == {"1", "2"}, "validation count")
    ok = r["graphs_after_setup"] == r["graphs_after_windows"]
    for v in r["validation"].values():
        require(
            len(v["input_ids"]) == n and len(v["reference"]) == len(v["output"]) == 128,
            "token shape",
        )
        require(
            all(
                type(t) is int and 0 <= t < 152064
                for t in v["input_ids"] + v["output"] + v["reference"]
            ),
            "token id",
        )
        c, a = v["cosine_by_step"], v["max_abs_by_step"]
        require(
            len(c) == len(a) == 128
            and all(math.isfinite(x) and -1.00001 <= x <= 1.00001 for x in c)
            and all(math.isfinite(x) and x >= 0 for x in a),
            "logits shape/finite",
        )
        passed = v["reference"] == v["output"] and min(c) >= 0.999 and max(a) <= 1.0
        require(v["passed"] is passed, "incorrect prompt status")
        ok = ok and passed
    ws = r["windows"]
    require([w["prompt"] for w in ws] == [1, 2, 2, 1], "window coverage")
    for w in ws:
        require(len(w["decode_ms"]) == 127 and len(w["output"]) == 128, "window length")
        ok = ok and w["output"] == r["validation"][str(w["prompt"])]["reference"]
    tt = [w["ttft_ms"] for w in ws]
    d = [v for w in ws for v in w["decode_ms"]]
    require(all(math.isfinite(x) and x > 0 for x in tt + d), "invalid timing")
    stats = [median(tt), median(d), p95(d)]
    require(
        stats == [r["ttft_p50_ms"], r["decode_p50_ms"], r["decode_p95_ms"]] and r["eligible"] is ok,
        "summary mismatch",
    )
    before, after = r["model_storage_before"], r["model_storage_after"]
    require(
        before == after
        and before >= 15_000_000_000
        and r["warm_peak_reserved"] >= r["warm_peak_allocated"] >= after,
        "storage accounting",
    )
    return dict(
        eligible=ok,
        ttft_p50_ms=stats[0],
        decode_p50_ms=stats[1],
        decode_p95_ms=stats[2],
        warm_peak_allocated_bytes=r["warm_peak_allocated"],
        warm_peak_reserved_bytes=r["warm_peak_reserved"],
        resident_model_storage_bytes=after,
        model_storage_delta_bytes=after - before,
        setup_peak_allocated_bytes=r["setup_peak_allocated"],
        first_token_mismatch={
            k: next(
                (i + 1 for i, (a, b) in enumerate(zip(v["reference"], v["output"])) if a != b), None
            )
            for k, v in r["validation"].items()
        },
    )


def build(directory):
    rows = {}
    hashes = {}
    identity = None
    drivers = (
        "checkpoint7b.py",
        "resident.py",
        "scale7b.py",
        "projection.py",
        "run.py",
        "metrics.py",
    )
    expected = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in drivers
    }
    gates = set()
    for n in (128, 2048):
        references = None
        rows[str(n)] = {}
        for k in ("eager", "resident", "inductor"):
            f = directory / f"qwen7b-{n}-{k}.json"
            data = f.read_bytes()
            r = json.loads(data)
            hashes[f.name] = hashlib.sha256(data).hexdigest()
            require(r["source_sha256"] == expected, "driver does not match this source tree")
            require(
                r["validation"]["1"]["input_ids"] != r["validation"]["2"]["input_ids"],
                "duplicate held-out prompt",
            )
            if k == "resident":
                h = r["gate_source_sha256"]
                require(
                    isinstance(h, str) and len(h) == 64 and all(c in "0123456789abcdef" for c in h),
                    "native gate identity",
                )
                gates.add(h)
            val = check(r, n, k)
            rows[str(n)][k] = val
            same = (
                r["source_sha256"],
                r["device"],
                r["cuda"],
                r["torch"],
                r["transformers"],
                r["model_storage_before"],
            )
            if identity is None:
                identity = same
            require(same == identity, "mixed environment or driver")
            trajectory = {j: (v["input_ids"], v["reference"]) for j, v in r["validation"].items()}
            if references is None:
                references = trajectory
            require(trajectory == references, "different prompt or reference")
    require(len(gates) == 1, "mixed decode gate across lengths")
    return dict(
        format="sparkfuse.qwen2-7b.summary.v1",
        revision=REV,
        case_sha256=hashes,
        cases=rows,
        windows_per_case=4,
        generated_tokens_per_request=128,
        distinct_measured_prompts_per_length=2,
        default_changed=False,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    a = p.parse_args()
    print(json.dumps(build(a.directory), indent=2, allow_nan=False))
