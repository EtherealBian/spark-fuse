"""Offline reconstruction of the matched 32B pilot from two raw JSON receipts (stdlib)."""

import hashlib
import json
import math
import re
import sys
from pathlib import Path
from statistics import median

from checkpoint32b import MODEL, PARAMETERS, REV
from metrics import p95

FILES = (
    "scale32b.py",
    "checkpoint32b.py",
    "qwen32b.lock.json",
    "scale7b.py",
    "run.py",
    "resident.py",
    "projection.py",
)


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def tokens(xs, n):
    return len(xs) == n and all(type(x) is int and 0 <= x < 152064 for x in xs)


def digest(s):
    return isinstance(s, str) and re.fullmatch(r"[a-f0-9]{64}", s) is not None


def case(report, n, after):
    need(report["length"] == n and report["capacity"] == n + 32, "cache or prompt length")
    checks, windows = report["checks"], report["windows"]
    need(
        [c["prompt"] for c in checks] == [1, 2]
        and checks[0]["input_ids"] != checks[1]["input_ids"],
        "held-out prompt coverage",
    )
    need([w["prompt"] for w in windows] == [1, 2, 2, 1], "window coverage")
    ok = True
    refs = []
    for c in checks:
        d = c["max_abs_by_step"]
        need(
            tokens(c["input_ids"], n) and tokens(c["reference"], 32) and tokens(c["free"], 32),
            "token shape/ID",
        )
        need(
            len(d) == 32
            and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in d),
            "logit errors",
        )
        need(
            c["score_dtype"] in ("torch.bfloat16", "torch.float32")
            and digest(c["reference_logits_sha256"])
            and digest(c["logits_sha256"]),
            "logit digest/dtype",
        )
        eq = c["logits_sha256"] == c["reference_logits_sha256"]
        need(c["logits_bitwise_equal"] is eq and (not eq or max(d) == 0), "inconsistent logits")
        ok = ok and eq and c["free"] == c["reference"]
        refs.append(
            (c["input_ids"], c["reference"], c["score_dtype"], c["reference_logits_sha256"])
        )
    for w in windows:
        need(len(w["decode_ms"]) == 31 and tokens(w["tokens"], 32), "window length")
        ok = ok and w["tokens"] == refs[w["prompt"] - 1][1]
    tt = [w["ttft_ms"] for w in windows]
    d = [v for w in windows for v in w["decode_ms"]]
    need(all(type(x) in (float, int) and math.isfinite(x) and x > 0 for x in tt + d), "timings")
    alloc, reserv = report["warm_peak_allocated"], report["warm_peak_reserved"]
    need(
        type(alloc) is int
        and type(reserv) is int
        and reserv >= alloc >= after
        and report["setup_peak_allocated"] >= after,
        "allocator",
    )
    return refs, dict(
        eligible=ok,
        windows=len(tt),
        decode_samples=len(d),
        ttft_p50_ms=median(tt),
        decode_p50_ms=median(d),
        decode_p95_ms=p95(d),
        warm_peak_allocated_bytes=alloc,
        warm_peak_reserved_bytes=reserv,
        max_abs_logit_error=max(max(c["max_abs_by_step"]) for c in checks),
        all_logits_byte_equal=all(c["logits_bitwise_equal"] for c in checks),
        all_tokens_exact=all(c["free"] == c["reference"] for c in checks)
        and all(w["tokens"] == refs[w["prompt"] - 1][1] for w in windows),
    )


def build(directory):
    expected = {
        f: hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest() for f in FILES
    }
    out = dict(
        model=MODEL,
        revision=REV,
        bf16_parameters=PARAMETERS,
        raw_sha256={},
        model_storage={},
        cases={},
    )
    oracle = {}
    env = None
    for backend in ("eager", "resident"):
        f = directory / f"qwen32b-{backend}.json"
        raw = f.read_bytes()
        r = json.loads(raw)
        out["raw_sha256"][f.name] = hashlib.sha256(raw).hexdigest()
        need(
            (
                r["format"],
                r["backend"],
                r["model"],
                r["revision"],
                r["parameters"],
                r["layers"],
                r["batch"],
                r["attention"],
                r["generated"],
                r["dtype"],
            )
            == (
                "sparkfuse.qwen25-32b.v1",
                backend,
                MODEL,
                REV,
                PARAMETERS,
                64,
                1,
                "sdpa",
                32,
                "bfloat16",
            ),
            "experiment identity",
        )
        need(r["source_sha256"] == expected, "source hashes")
        runtime = {k: r[k] for k in ("torch", "transformers", "accelerate", "cuda", "device")}
        need(
            all(isinstance(v, str) and v for v in runtime.values())
            and runtime["device"] == "NVIDIA GB10",
            "runtime identity",
        )
        if env is None:
            env = runtime
        need(env == runtime, "unmatched runtime")
        before, after = r["model_storage_before"], r["model_storage_after"]
        need(
            type(before) is int
            and type(after) is int
            and PARAMETERS * 2 <= before <= PARAMETERS * 2 + 2**20
            and after >= PARAMETERS * 2,
            "model storage",
        )
        if backend == "resident":
            need(
                digest(r["gate_source_sha256"])
                and before == out["model_storage"]["eager"]["before"],
                "resident identity",
            )
        else:
            need(after == before, "eager model changed")
        out["model_storage"][backend] = dict(before=before, after=after, delta=after - before)
        need([c["length"] for c in r["cases"]] == [128, 2048], "case coverage")
        for row in r["cases"]:
            n = row["length"]
            refs, stats = case(row, n, after)
            if backend == "eager":
                oracle[n] = refs
            need(oracle[n] == refs, "unmatched eager oracle")
            out["cases"].setdefault(str(n), {})[backend] = stats
    out["runtime"] = env
    out["strict_pilot_passed"] = out["model_storage"]["resident"]["delta"] == 0 and all(
        b["eligible"] for length in out["cases"].values() for b in length.values()
    )
    return out


if __name__ == "__main__":
    print(json.dumps(build(Path(sys.argv[1])), indent=2, allow_nan=False))
