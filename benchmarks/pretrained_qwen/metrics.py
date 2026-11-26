"""CPU-only decision rule. Each window pairs backends on one identical prompt."""

from math import ceil, isfinite
from statistics import median


def p95(samples):
    if not samples or not all(isfinite(x) and x > 0 for x in samples):
        raise ValueError("expected positive finite latency samples")
    return sorted(samples)[ceil(0.95 * len(samples)) - 1]


def summarize(rows, left, right, eligible):
    if len(rows) < 2:
        raise ValueError("insufficient paired requests")
    lo = [r[left]["decode_ms"] for r in rows]
    hi = [r[right]["decode_ms"] for r in rows]
    if any(not a or len(a) != len(b) for a, b in zip(lo, hi, strict=True)):
        raise ValueError("unpaired token budget")
    flat = lambda v: [x for req in v for x in req]
    a, b = flat(lo), flat(hi)
    ap, bp = p95(a), p95(b)
    paired = [median(x) / median(y) for x, y in zip(lo, hi, strict=True)]
    wins = sum(q <= 0.9 for q in paired)
    ratio = median(a) / median(b)
    return dict(
        candidate=left,
        baseline=right,
        eligible=eligible,
        pooled_p50_ratio=ratio,
        pooled_p95_ratio=ap / bp,
        median_paired_ratio=median(paired),
        windows_at_least_10pct_faster=wins,
        windows=len(rows),
        target_met=bool(
            eligible and ratio <= 0.9 and ap / bp <= 1.05 and wins >= ceil(2 * len(rows) / 3)
        ),
    )
