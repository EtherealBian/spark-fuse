"""Small cold-prompt fixed-length feasibility baseline for a single Spark; no model-quality claim."""

import argparse
import json
import math
from pathlib import Path
from statistics import median
from prefix_replay import complete, digest, post

LENGTHS = (128, 1024)
STEMS = (
    "The ledger stores a previous version alongside each timestamp. ",
    "A service assigns every request a durable identifier. ",
)
QUESTIONS = (
    "Why might an append-only ledger help recovery?",
    "How do identifiers help detect repeated submissions?",
)


def require(x, why):
    if not x:
        raise ValueError(why)


def run(port, checkpoint=None):
    def tok(s):
        return post(port, "tokenize", dict(content=s, add_special=False, parse_special=False))[
            "tokens"
        ]

    complete(port, tok("Warm up the inference backend."), False, 3)
    cases = {}
    for n in LENGTHS:
        for j in range(2):
            suffix = tok("\nQuestion: " + QUESTIONS[j] + " Answer briefly:")
            stem = tok(STEMS[j] * 190)
            require(0 < len(suffix) < n < len(stem), "bad prompt")
            cases[f"{n}:{j}"] = stem[: n - len(suffix)] + suffix
    out = []
    for rep in (0, 1):
        for k in list(cases) if rep == 0 else list(reversed(cases)):
            w = complete(port, cases[k], False, 16)
            out.append(dict(rep=rep, case=k, input_sha256=digest(cases[k]), **w))
            if checkpoint is not None:
                checkpoint(dict(inputs=cases, rows=out))
            print("case", rep, k, w["processed"], w["predicted"], flush=True)
    return dict(inputs=cases, rows=out)


def audit(raw, eog):
    keys = [f"{n}:{j}" for n in LENGTHS for j in range(2)]
    spec = raw["inputs"]
    rows = raw["rows"]
    require(
        set(spec) == set(keys)
        and len(set(digest(s) for s in spec.values())) == 4
        and len(rows) == 8,
        "incomplete or duplicate prompts",
    )
    summ = {}
    for i, (r, k) in enumerate(zip(rows, keys + list(reversed(keys)), strict=True)):
        size = int(k.split(":")[0])
        ids = spec[k]
        require(
            all(type(r[x]) is int for x in ("rep", "processed", "reused", "predicted"))
            and r["rep"] == i // 4
            and r["case"] == k
            and r["input_sha256"] == digest(ids)
            and len(ids) == r["processed"] == size
            and r["reused"] == 0
            and all(type(t) is int and 0 <= t < 151552 for t in ids),
            "input/ordering mismatch",
        )
        require(
            r["stop"] == "limit"
            and r["predicted"] == len(r["ids"]) == 16
            and all(type(t) is int and 0 <= t < 151552 and t not in eog for t in r["ids"]),
            "bad output",
        )
        require(
            all(
                type(r[x]) in (int, float) and math.isfinite(r[x]) and r[x] > 0
                for x in ("wall_ms", "decode_ms", "prompt_ms")
            )
            and r["wall_ms"] + 3 >= r["decode_ms"] + r["prompt_ms"],
            "timing mismatch",
        )
    for n in LENGTHS:
        cohort = [r for r in rows if len(spec[r["case"]]) == n]
        matches = sum(
            next(r["ids"] for r in cohort if r["case"] == f"{n}:{j}" and r["rep"] == 0)
            == next(r["ids"] for r in cohort if r["case"] == f"{n}:{j}" and r["rep"] == 1)
            for j in range(2)
        )
        summ[str(n)] = {
            k: median(r[k] for r in cohort) for k in ("wall_ms", "decode_ms", "prompt_ms")
        }
        first = []
        for j in range(2):
            a, b = [
                next(r["ids"] for r in cohort if r["case"] == f"{n}:{j}" and r["rep"] == rep)
                for rep in (0, 1)
            ]
            first.append(
                next((i for i, (x, y) in enumerate(zip(a, b, strict=True), 1) if x != y), None)
            )
        summ[str(n)].update(
            repeated_greedy_exact=matches, paired_cases=2, first_different_token=first
        )
    return dict(
        cases=summ,
        output_per_request=16,
        scope="Four synthetic prompts, repeated in reverse order in one process. Warm weights, cold prompt KV. No quality or cross-model speed comparison.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["run", "audit"])
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--port", type=int)
    a = p.parse_args()
    r = json.loads(a.runtime.read_bytes())
    require(
        r["gpu_layers_including_output"] == 94
        and r["context"] == dict(n_seq_max=1, n_ctx=4096, n_ctx_seq=4096, n_batch=512, n_ubatch=128)
        and r["device"] == "NVIDIA GB10",
        "wrong GLM-4.5 runtime",
    )
    eog = r["eog_token_ids"]
    if a.mode == "run":
        partial = a.rows.with_name(a.rows.name + ".partial")
        temp = partial.with_name(partial.name + ".tmp")
        if any(x.exists() for x in (a.rows, partial, temp)):
            raise FileExistsError("refuse to overwrite complete or partial evidence")

        def checkpoint(observations):
            temp.write_text(json.dumps(observations, separators=(",", ":"), allow_nan=False) + "\n")
            temp.replace(partial)

        result = run(a.port, checkpoint)
        audit(result, eog)
        partial.replace(a.rows)
    else:
        print(json.dumps(audit(json.loads(a.rows.read_bytes()), eog), indent=2, allow_nan=False))
