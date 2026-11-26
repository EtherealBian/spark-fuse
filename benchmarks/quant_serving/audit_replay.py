"""Offline validation of counters, design and token equality. No GPU or model required."""

import argparse
import json
import math
import re
import hashlib
from pathlib import Path
from statistics import median

N = 24
TAGS = ("antimony", "indigo", "cobalt", "ochre")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def audit(rows):
    groups = {}
    require(isinstance(rows, list) and len(rows) == 8, "expected eight paired requests")
    for size in (256, 2048):
        cohort = [r for r in rows if r["length"] == size]
        require(len(cohort) == 4 and {r["case"] for r in cohort} == set(TAGS), "coverage")
        require(len({r["target"] for r in cohort}) == 4, "distinct target prompts required")
        for r in cohort:
            i = TAGS.index(r["case"])
            off = int(size == 2048)
            expected = ["cold", "reuse"] if (i + off) % 2 == 0 else ["reuse", "cold"]
            require(
                r["order"] == expected
                and all(re.fullmatch("[0-9a-f]{64}", r[k]) for k in ("primer", "target"))
                and r["primer"] != r["target"],
                "pair identity or interleaving",
            )
            target, primer = r["target_ids"], r["primer_ids"]
            for key, ids, expected_len in [
                ("target", target, size),
                ("primer", primer, r["primer_length"]),
            ]:
                require(
                    isinstance(ids, list)
                    and len(ids) == expected_len
                    and all(type(v) is int and 0 <= v < 152064 for v in ids),
                    "input token accounting",
                )
                require(
                    hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
                    == r[key],
                    "input digest mismatch",
                )
            exact_prefix = next(
                (i for i, (a, b) in enumerate(zip(target, primer)) if a != b),
                min(len(target), len(primer)),
            )
            require(exact_prefix == r["shared"], "incorrect reported common prefix")
            shared = r["shared"]
            require(
                type(shared) is int and size - 80 <= shared < min(size, r["primer_length"]),
                "shared prefix bounds",
            )
            for arm in ("cold", "reuse"):
                w = r[arm]
                require(
                    w["stop"] == "limit"
                    and type(w["predicted"]) is int
                    and w["predicted"] == N
                    and len(w["ids"]) == N
                    and all(type(v) is int and 0 <= v < 152064 for v in w["ids"]),
                    "output length, vocabulary or stop",
                )
                require(
                    all(type(w[k]) is int and w[k] >= 0 for k in ("processed", "reused"))
                    and w["processed"] + w["reused"] == size,
                    "prompt accounting",
                )
                require(
                    all(
                        type(w[k]) in (float, int) and math.isfinite(w[k]) and w[k] > 0
                        for k in ("prompt_ms", "decode_ms", "wall_ms")
                    ),
                    "timing",
                )
                require(
                    w["wall_ms"] + 3 >= w["prompt_ms"] + w["decode_ms"],
                    "server/client clock duration",
                )
            require(
                r["cold"]["reused"] == 0 and r["reuse"]["reused"] == shared,
                "cold/reuse did not execute the expected work",
            )
        eq = sum(r["cold"]["ids"] == r["reuse"]["ids"] for r in cohort)
        groups[str(size)] = dict(
            pairs=len(cohort),
            exact_token_pairs=eq,
            all_tokens_exact=(eq == len(cohort)),
            reused_range=[min(r["shared"] for r in cohort), max(r["shared"] for r in cohort)],
            **{
                k: {
                    m: median(r[k][m] for r in cohort)
                    for m in ("prompt_ms", "decode_ms", "wall_ms")
                }
                for k in ("cold", "reuse")
            },
            median_paired_wall_ms_saved=median(
                r["cold"]["wall_ms"] - r["reuse"]["wall_ms"] for r in cohort
            ),
        )
    return dict(
        schema=1,
        output_tokens_per_request=N,
        cases=groups,
        all_tokens_exact=all(v["all_tokens_exact"] for v in groups.values()),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("rows", type=Path)
    a = p.parse_args()
    print(json.dumps(audit(json.loads(a.rows.read_text())), indent=2, allow_nan=False))
