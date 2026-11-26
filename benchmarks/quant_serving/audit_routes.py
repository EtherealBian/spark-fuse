"""Audit the routing decisions independently using the original tokenized workloads."""

import argparse
import json
import math
from pathlib import Path
from prefix_replay import digest
from routed_trace import LENGTH, NGEN, PATTERNS, POLICIES
from statistics import median


def check(ok, msg):
    if not ok:
        raise ValueError(msg)


def audit(traces, inputs):
    check(len(traces) == 8 and len(inputs) == 4, "incomplete suite")
    index = 0
    outputs = {}
    for pi, pattern in enumerate(PATTERNS):
        sums = {p: [] for p in POLICIES}
        matched = 0
        for rep in range(2):
            spec = inputs[f"{pattern}:{rep}"]
            check(
                len(spec) == 6
                and all(
                    len(ids) == LENGTH and all(type(i) is int and 0 <= i < 152064 for i in ids)
                    for ids in spec
                )
                and len(set(map(digest, spec))) == 6,
                "invalid inputs",
            )
            # Workload validation is independent of the measured routing/timing.
            for i in range(6):
                for j in range(i):
                    a, b = spec[i], spec[j]
                    shared = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), len(a))
                    same = PATTERNS[pattern][i] == PATTERNS[pattern][j]
                    check(
                        (shared >= LENGTH * 0.9 if same else shared < 32),
                        "input IDs do not encode the claimed traffic pattern",
                    )
            arms = POLICIES if (pi + rep) % 2 == 0 else POLICIES[::-1]
            by = {}
            for p in arms:
                tr = traces[index]
                index += 1
                check(
                    (tr["pattern"], tr["rep"], tr["policy"]) == (pattern, rep, p)
                    and len(tr["rows"]) == 6,
                    "invalid experiment order",
                )
                prev = [[], []]
                age = [-1, -1]
                for i, (r, ids) in enumerate(zip(tr["rows"], spec, strict=True)):
                    gains = []
                    for old in prev:
                        match = next(
                            (j for j, (x, y) in enumerate(zip(old, ids)) if x != y),
                            min(len(old), len(ids)),
                        )
                        gains.append(match if match >= 32 else 0)
                    ix = (
                        i % 2
                        if p == "round_robin"
                        else min(range(2), key=lambda j: (-gains[j], age[j], j))
                    )
                    n = gains[ix]
                    check(
                        r["slot"] == ix and r["prefix"] == n and r["target"] == digest(ids),
                        "route disagrees with reference",
                    )
                    check(
                        r["reused"] == n and r["processed"] == LENGTH - n, "cache counter mismatch"
                    )
                    prev[ix] = ids
                    age[ix] = i
                    check(
                        r["predicted"] == NGEN
                        and len(r["ids"]) == NGEN
                        and r["stop"] == "limit"
                        and all(type(t) is int and 0 <= t < 152064 for t in r["ids"]),
                        "bad output",
                    )
                    check(
                        all(
                            type(r[k]) in (int, float) and math.isfinite(r[k]) and r[k] > 0
                            for k in ("decode_ms", "prompt_ms", "wall_ms")
                        )
                        and r["wall_ms"] + 3 >= r["decode_ms"] + r["prompt_ms"],
                        "bad timings",
                    )
                sums[p].append(
                    {
                        k: sum(r[k] for r in tr["rows"])
                        for k in ("reused", "processed", "wall_ms", "prompt_ms", "decode_ms")
                    }
                )
                by[p] = tr["rows"]
            matched += sum(
                a["ids"] == b["ids"] for a, b in zip(by[arms[0]], by[arms[1]], strict=True)
            )
        outputs[pattern] = dict(
            request_pairs=12,
            exact_token_pairs=matched,
            estimator="median of two six-request trace totals",
            **{p: {k: median(r[k] for r in sums[p]) for k in sums[p][0]} for p in POLICIES},
        )
    return dict(
        schema=1,
        prompt_tokens_per_request=LENGTH,
        generated_tokens_per_request=NGEN,
        patterns=outputs,
        all_tokens_exact=all(
            r["exact_token_pairs"] == r["request_pairs"] for r in outputs.values()
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("traces", type=Path)
    p.add_argument("inputs", type=Path)
    a = p.parse_args()
    print(
        json.dumps(
            audit(json.loads(a.traces.read_text()), json.loads(a.inputs.read_text())),
            indent=2,
            allow_nan=False,
        )
    )
