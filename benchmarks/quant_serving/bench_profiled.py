"""Paired CPU-only holdout for the opt-in exact scope+16 prefix matcher."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import median
import time
from slot_policy import SlotRouter

CASES = (
    (64, "empty"),
    (64, "tenant-switch"),
    (64, "same-tenant-miss"),
    (64, "tied-prefixes"),
    (64, "clustered-hits"),
    (64, "miss-after-one"),
    (64, "miss-after-fifteen"),
    (64, "one-eligible-edge"),
    (64, "dense-sixteen"),
    (2, "fallback-control"),
)
REPS = 8
SIZE = 2048


def need(test, reason):
    if not test:
        raise ValueError(reason)


def fixture(n, family, rep):
    require = (n, family) in CASES
    need(require, "unknown case")
    # No generated text; all inputs are owned by the benchmark. New tuples avoid pointer-only matches across a whole prompt.
    if family in ("miss-after-one", "miss-after-fifteen", "one-eligible-edge", "dense-sixteen"):
        seeds, qs = [], []
        common = {"miss-after-one":1, "miss-after-fifteen":15, "one-eligible-edge":15, "dense-sixteen":16}[family]
        head = (400 + rep,) * common
        for i in range(n):
            tail = tuple(7000 + (i + k + rep) % 113 for k in range(SIZE - common - 1))
            seeds.append(("pool", head + (100 + i,) + tail))
        for i in range(n):
            j = (13*i + rep) % n if family == "dense-sixteen" else i
            eligible = family == "dense-sixteen" or (family == "one-eligible-edge" and i == 0)
            v = seeds[j][1][:80] if eligible else head + (9000 + i,)
            qs.append(("pool", v + (20000 + i,) * (SIZE - len(v))))
        return seeds, qs
    prefix = 1920 if family == "tied-prefixes" else 2016
    seeds = []
    qs = []
    for j in range(n):
        cls = 0 if family == "tied-prefixes" else j % 8
        body = tuple(
            1000 * cls + (k + rep) % 19
            if k < 1792
            else (j * 3 + 4000 + (k + rep) % 17 if family != "tied-prefixes" else (k + rep) % 17)
            for k in range(SIZE)
        )
        seeds.append(("pool", body))
    for i in range(n):
        j = (13 * i + rep) % n
        if family == "same-tenant-miss":
            ids = (900000 + i,) + seeds[j][1][1:]
        elif family in ("empty", "tenant-switch"):
            ids = seeds[j][1][:prefix] + (900000 + i,) * (SIZE - prefix)
        else:
            ids = seeds[j][1][:prefix] + (900000 + i,) * (SIZE - prefix)
        qs.append(("fresh" if family == "tenant-switch" else "pool", ids))
    return ([] if family == "empty" else seeds), qs


def oracle(n, family, rep):
    if family in ("miss-after-one", "miss-after-fifteen"):
        return [[i, 0, False] for i in range(n)]
    if family == "one-eligible-edge":
        return [[0,80,True]] + [[i,0,False] for i in range(1,n)]
    if family == "dense-sixteen":
        return [[(13*i + rep) % n, 80, True] for i in range(n)]
    if family in ("empty", "tenant-switch", "same-tenant-miss"):
        return [[i, 0, False] for i in range(n)]
    if family == "tied-prefixes":
        return [[i, 1920, True] for i in range(n)]
    return [[(13 * i + rep) % n, 2016, True] for i in range(n)]


def order(g, rep):
    return ["scan", "scoped16"] if (g + rep) % 2 == 0 else ["scoped16", "scan"]


def one(mode, n, seeds, qs):
    r = SlotRouter(n, SIZE, batch_matcher=mode)
    if seeds:
        # Individual reservations hold all slots before any completion is committed.
        leases = [r.acquire(t, v) for t, v in seeds]
        need([x.slot for x in leases] == list(range(n)), "preparation order changed")
        for x in leases:
            r.finish(x, success=True)
    t = time.perf_counter_ns()
    leases = r.acquire_batch(qs, strategy="joint")
    elapsed = time.perf_counter_ns() - t
    return dict(ns=elapsed, leases=[[x.slot, x.prefix, x.reuse] for x in leases])


def bench():
    cpus = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(cpus)})
    was = gc.isenabled()
    gc.disable()
    try:
        rows = []
        for g, (n, family) in enumerate(CASES):
            for rep in range(-1, REPS):
                seeds, qs = fixture(n, family, rep)
                seq = order(g, rep)
                expected = oracle(n, family, rep)
                arms = {}
                # Each arm builds a fresh router before the timed full acquire_batch call.
                for m in seq:
                    x = one(m, n, seeds, qs)
                    need(x["leases"] == expected, "reservation differs from constructed oracle")
                    arms[m] = x
                if rep >= 0:
                    rows.append(dict(n=n, case=family, rep=rep, order=seq, arms=arms))
            print("done", n, family, flush=True)
        return dict(
            schema=1,
            python=platform.python_version(),
            machine=platform.machine(),
            cpu_affinity_count=len(os.sched_getaffinity(0)),
            rows=rows,
        )
    finally:
        os.sched_setaffinity(0, cpus)
        if was:
            gc.enable()


def audit(raw):
    need(
        raw["schema"] == 1
        and raw["machine"] == "aarch64"
        and type(raw["cpu_affinity_count"]) is int
        and raw["cpu_affinity_count"] == 1
        and len(raw["rows"]) == len(CASES) * REPS,
        "unexpected environment or incomplete matrix",
    )
    rows = iter(raw["rows"])
    out = {}
    wins = 0
    for g, (n, family) in enumerate(CASES):
        orders = {"scan": [], "scoped16": []}
        a = []
        b = []
        for rep in range(REPS):
            r = next(rows)
            seq = order(g, rep)
            need(
                (r["n"], r["case"], r["rep"], r["order"]) == (n, family, rep, seq)
                and set(r["arms"]) == set(seq),
                "case/order mismatch",
            )
            expected = oracle(n, family, rep)
            for arm in r["arms"].values():
                need(
                    type(arm["ns"]) is int
                    and arm["ns"] > 0
                    and type(arm["leases"]) is list
                    and len(arm["leases"]) == n
                    and all(
                        type(x) is list
                        and len(x) == 3
                        and type(x[0]) is int
                        and type(x[1]) is int
                        and type(x[2]) is bool
                        for x in arm["leases"]
                    )
                    and arm["leases"] == expected,
                    "lease or timing mismatch",
                )
            x = r["arms"]["scan"]["ns"]
            y = r["arms"]["scoped16"]["ns"]
            orders[seq[0]].append((x, y))
            a.append(x)
            b.append(y)
        gains = {k: median((x - y) / x for x, y in v) for k, v in orders.items()}
        deltas = {k: median((y - x) / 1e6 for x, y in v) for k, v in orders.items()}
        no_regression = all(
            gains[k] >= (-0.05 if n == 64 else -0.10) or deltas[k] <= (0.50 if n == 64 else 0.05)
            for k in gains
        )
        wins += int(n == 64 and min(gains.values()) >= 0.10)
        out[family] = dict(
            slots=n,
            scan_ms=median(a) / 1e6,
            scoped16_ms=median(b) / 1e6,
            paired_fraction_saved_by_first=gains,
            paired_increase_ms_by_first=deltas,
            within_regression_bounds=no_regression,
        )
    return dict(
        cases=out,
        matched_pairs=len(raw["rows"]),
        ten_percent_wins_on_64=wins,
        passes_opt_in_gate=all(x["within_regression_bounds"] for x in out.values()) and wins >= 2,
        scope="One pinned Spark CPU core; 8 order-balanced paired repetitions of complete 2048-token batch reservation per synthetic holdout. No GPU, server queue or model-output latency. Default stays scan. Results do not establish benefit for an unmeasured workload.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["run", "audit"])
    p.add_argument("rows", type=Path)
    o = p.parse_args()
    if o.mode == "audit":
        b = o.rows.read_bytes()
        r = audit(json.loads(b))
        r["rows_sha256"] = hashlib.sha256(b).hexdigest()
        print(json.dumps(r, indent=2, allow_nan=False))
    else:
        tmp = o.rows.with_suffix(".partial")
        need(not tmp.exists() and not o.rows.exists(), "refuse to overwrite")
        r = bench()
        audit(r)
        tmp.write_text(json.dumps(r, separators=(",", ":"), allow_nan=False) + "\n")
        tmp.replace(o.rows)
