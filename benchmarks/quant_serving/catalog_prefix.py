"""Reproduce the eight original prompts: does preparing a block-aligned prefix preserve greedy tokens?"""

import argparse
import json
import math
import random
from pathlib import Path
from statistics import median
from prefix_replay import complete, digest


def arms(size, shared):
    policy = {"cold-a": (shared, False), "cold-b": (shared, False), "full": (shared, True)}
    for block in (128, 512):
        cut = (shared // block) * block
        if 0 < cut < shared and cut < size:
            policy[f"block-{block}"] = (cut, True)
    return policy


def order(ix, options):
    names = list(options)
    random.Random(204823 + ix).shuffle(names)
    return names


def run(port, earlier):
    rows = []
    for ix, r in enumerate(earlier):
        ids = r["target_ids"]
        size = len(ids)
        same = r["shared"]
        pol = arms(size, same)
        row = dict(case=r["case"], size=size, target=digest(ids), shared=same, arms={})
        for name in order(ix, pol):
            cut, reuse = pol[name]
            primer = complete(port, ids[:cut], False, 1)
            res = complete(port, ids, reuse, 24)
            row["arms"][name] = dict(cut=cut, prime=primer, request=res)
        rows.append(row)
        print(
            "catalog",
            size,
            r["case"],
            {
                n: (
                    w["request"]["reused"],
                    w["request"]["ids"] == row["arms"]["cold-a"]["request"]["ids"],
                )
                for n, w in row["arms"].items()
            },
            flush=True,
        )
    return rows


def check(ok, msg):
    if not ok:
        raise ValueError(msg)


def audit(raw, earlier):
    check(len(raw) == len(earlier) == 8, "eight source cases required")
    stats = {}
    for ix, (row, src) in enumerate(zip(raw, earlier, strict=True)):
        size = row["size"]
        same = row["shared"]
        ids = src["target_ids"]
        check(
            size == src["length"] == len(ids)
            and row["case"] == src["case"]
            and row["target"] == digest(ids) == src["target"]
            and same == src["shared"],
            "unmatched source",
        )
        opts = arms(size, same)
        aa = row["arms"]
        check(list(aa) == order(ix, opts), "incorrect predeclared arm order")
        ref = aa["cold-a"]["request"]["ids"]
        cold_ref = src["cold"]["ids"]
        sg = stats.setdefault(
            str(size),
            {"cases": 0, "two_new_cold_exact": 0, "new_cold_matches_original": 0, "arms": {}},
        )
        sg["cases"] += 1
        sg["two_new_cold_exact"] += int(aa["cold-b"]["request"]["ids"] == ref)
        sg["new_cold_matches_original"] += int(ref == cold_ref)
        for name, (cut, reuse) in opts.items():
            w = aa[name]
            check(type(w["cut"]) is int and cut == w["cut"], "incorrect prepared length")
            prime = w["prime"]
            rr = w["request"]
            expected = cut if reuse else 0
            for a, n, p, k in [(prime, 1, cut, 0), (rr, 24, size - expected, expected)]:
                check(
                    a["stop"] == "limit"
                    and type(a["predicted"]) is int
                    and a["predicted"] == n
                    and len(a["ids"]) == n
                    and all(type(x) is int and 0 <= x < 152064 for x in a["ids"]),
                    "unmatched generation",
                )
                check(
                    all(type(a[k]) is int for k in ("processed", "reused"))
                    and (a["processed"], a["reused"]) == (p, k),
                    "cache boundary differs from intended policy",
                )
                check(
                    all(
                        type(a[x]) in (int, float) and math.isfinite(a[x]) and a[x] >= 0
                        for x in ("prompt_ms", "decode_ms", "wall_ms")
                    )
                    and a["prompt_ms"] > 0
                    and (n == 1 or a["decode_ms"] > 0)
                    and a["wall_ms"] > 0
                    and a["wall_ms"] + 3 >= a["prompt_ms"] + a["decode_ms"],
                    "invalid latency",
                )
            eq = rr["ids"] == ref
            first = next((j for j, (x, y) in enumerate(zip(rr["ids"], ref)) if x != y), 24)
            v = sg["arms"].setdefault(
                name,
                dict(
                    exact=0,
                    first_divergent_steps=[],
                    evaluated=[],
                    prepare_wall=[],
                    request_wall=[],
                    prompt_ms=[],
                ),
            )
            if reuse:
                # Compare same-case target wall times. A catalog entry costs the entire
                # measured primer invocation (including its one generated token).
                delta = aa["cold-a"]["request"]["wall_ms"] - rr["wall_ms"]
                v.setdefault("paired_estimates", []).append(
                    dict(
                        case=row["case"],
                        equal_to_cold=eq,
                        saved_per_hit_ms=delta,
                        # Number of hits to recover at least the one-time preparation cost.
                        break_even_hits=math.ceil(prime["wall_ms"] / delta) if delta > 0 else None,
                    )
                )
            v["exact"] += int(eq)
            if not eq:
                v["first_divergent_steps"].append(first + 1)
            for key, value in [
                ("evaluated", rr["processed"]),
                ("prepare_wall", prime["wall_ms"]),
                ("request_wall", rr["wall_ms"]),
                ("prompt_ms", rr["prompt_ms"]),
            ]:
                v[key].append(value)
    for grp in stats.values():
        for arm in grp["arms"].values():
            for key in ("evaluated", "prepare_wall", "request_wall", "prompt_ms"):
                arm[key + "_p50"] = median(arm.pop(key))
    return dict(
        cases=stats,
        estimates="Per-case break-even hits are projections, not repeated-hit measurements. Assumes the same hit cost, no eviction or reload; null means no positive target saving. Cold primer is not charged to the no-catalog baseline.",
        scope="Fixed prepared-prefix lengths on the same eight original prompts; not arbitrary dynamic slot trimming or held-out validation.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("run", "audit"))
    p.add_argument("--port", type=int)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--rows", type=Path, required=True)
    a = p.parse_args()
    old = json.loads(a.source.read_bytes())
    if a.mode == "run":
        a.rows.write_text(
            json.dumps(run(a.port, old), separators=(",", ":"), allow_nan=False) + "\n"
        )
    else:
        print(json.dumps(audit(json.loads(a.rows.read_bytes()), old), indent=2, allow_nan=False))
