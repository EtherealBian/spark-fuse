"""Offline exercise of two-slot admission with the six *published* 72B input groups.

There is no server in this experiment. One slot stays busy while a complete pair
of real input ID lists arrives. The previous observed assignments are the oracle.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
from threading import Event, Thread
from time import perf_counter_ns, sleep, thread_time_ns
from audit_batch import rebuild
from slot_policy import SlotRouter
from slot_admission import BatchAdmission

PIN = "8df5bf1fc2bdc824f3b8c4e95624d7319aae268616a9969ee1e0363a288e0634"
HOLDS = (2250000, 4750000, 7250000, 9750000)
MODES = ("poll-1ms", "fifo")
G = ("contended", "contended-reversed", "disjoint")
SOURCE = ("slot_admission.py", "admission_trace.py", "slot_policy.py", "prefix_index.py")


def need(x, msg):
    if not x:
        raise ValueError(msg)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sources():
    from slot_policy import __file__ as router

    return {
        k: sha(
            Path(router if k in ("slot_policy.py", "prefix_index.py") else __file__).with_name(k)
        )
        for k in SOURCE
    }


def inputs(ref):
    need(
        sha(ref / "rows.json") == PIN
        and rebuild(ref) == json.loads((ref / "summary.json").read_bytes()),
        "unverified historical input or oracle",
    )
    return json.loads((ref / "rows.json").read_bytes())


def identity(case):
    ref = next(a for a in case["arms"] if a["mode"] == "joint")
    return [
        [r["response"]["observed_slot"], r["estimated_prefix"], r["response"]["cache_prompt"]]
        for r in ref["requests"]
    ]


def one(case, mode, hold):
    broker = (
        BatchAdmission(2, 4096, max_waiters=1, max_waiting_tokens=2048) if mode == "fifo" else None
    )
    base = SlotRouter(2, 4096) if broker is None else None
    reserve = (lambda req: broker.reserve(req, timeout_s=3)) if broker else base.acquire_batch
    finish = (broker or base).finish
    seeds = [("one", ids) for ids in case["inputs"]["seeds"]]
    targets = [("one", ids) for ids in case["inputs"]["tasks"]]
    leases = reserve(seeds)
    finish(leases[0], success=True)
    start = Event()
    out = {}
    errors = []

    def caller():
        try:
            wall = perf_counter_ns()
            cpu = thread_time_ns()
            start.set()
            attempts = 0
            while True:
                attempts += 1
                ls = reserve(targets)
                if ls is not None:
                    break
                sleep(0.001)
                if perf_counter_ns() - wall > 3_000_000_000:
                    raise TimeoutError("poll control")
            end = perf_counter_ns()
            active = thread_time_ns() - cpu
            out.update(
                start=wall,
                end=end,
                worker_cpu_ns=active,
                api_calls=attempts,
                leases=[[lease.slot, lease.prefix, lease.reuse] for lease in ls],
            )
            for lease in ls:
                finish(lease, success=True)
        except BaseException as e:
            errors.append(e)

    w = Thread(target=caller, daemon=True)
    w.start()
    need(start.wait(2), "caller did not start")
    sleep(hold / 1e9)
    before = perf_counter_ns()
    cpu = thread_time_ns()
    finish(leases[1], success=True)
    release_cpu = thread_time_ns() - cpu
    w.join(4)
    need(not w.is_alive() and not errors, "caller failed: " + repr(errors))
    route_calls = (broker.snapshot()["router_calls"] - 1) if broker else out["api_calls"]
    waits = broker.snapshot()["wait_calls"] if broker else out["api_calls"] - 1
    need(waits >= 1 and out["leases"] == identity(case), "no contention or changed slot decision")
    return dict(
        leases=out["leases"],
        api_calls=out["api_calls"],
        router_calls=route_calls,
        waits=waits,
        worker_cpu_ns=out["worker_cpu_ns"],
        release_cpu_ns=release_cpu,
        arrival_to_grant_ns=out["end"] - out["start"],
        actual_hold_ns=before - out["start"],
        release_to_grant_ns=out["end"] - before,
    )


def schedule():
    for i in range(6):
        for j, h in enumerate(HOLDS):
            for flip in (0, 1):
                yield i, h, MODES if (i + j + flip) % 2 == 0 else MODES[::-1]


def measure(ref):
    cpus = os.sched_getaffinity(0)
    cpu = min(cpus)
    os.sched_setaffinity(0, {cpu})
    cases = inputs(ref)
    for m in MODES:
        one(cases[0], m, HOLDS[-1])
    rows = []
    for idx, h, order in schedule():
        c = cases[idx]
        rows.append(
            dict(case=idx, hold_ns=h, order=list(order), arms={m: one(c, m, h) for m in order})
        )
    return dict(
        schema=1,
        source_sha256=sources(),
        history_sha256=PIN,
        affinity_cpus=len(os.sched_getaffinity(0)),
        rows=rows,
    )


def audit(obj, ref):
    cases = inputs(ref)
    need(
        obj["schema"] == 1
        and obj["source_sha256"] == sources()
        and obj["history_sha256"] == PIN
        and obj["affinity_cpus"] == 1
        and len(obj["rows"]) == 48,
        "identity or row count",
    )
    bins = {m: [] for m in MODES}
    counts = {m: 0 for m in MODES}
    for r, (i, h, order) in zip(obj["rows"], schedule(), strict=True):
        need(
            r["case"] == i
            and r["hold_ns"] == h
            and tuple(r["order"]) == order
            and set(r["arms"]) == set(MODES),
            "case or order",
        )
        expected = identity(cases[i])
        v = r["arms"]
        for m, row in v.items():
            need(
                row["leases"] == expected
                and all(
                    type(slot) is int and type(prefix) is int and type(hit) is bool
                    for slot, prefix, hit in row["leases"]
                )
                and all(
                    type(row[k]) is int and row[k] > 0
                    for k in (
                        "api_calls",
                        "router_calls",
                        "waits",
                        "worker_cpu_ns",
                        "release_cpu_ns",
                        "arrival_to_grant_ns",
                        "actual_hold_ns",
                        "release_to_grant_ns",
                    )
                ),
                "invalid measurement",
            )
            need(
                row["actual_hold_ns"] >= h
                and row["arrival_to_grant_ns"]
                == row["actual_hold_ns"] + row["release_to_grant_ns"],
                "invalid clock boundary",
            )
            need(
                (
                    row["api_calls"] == row["router_calls"] == 1
                    if m == "fifo"
                    else row["router_calls"] == row["api_calls"] == row["waits"] + 1
                ),
                "bad retry accounting",
            )
            counts[m] += row["router_calls"]
        bins[order[0]].append(v)
    result = {}
    for name, rows in bins.items():

        def p95(m):
            return sorted(r[m]["release_to_grant_ns"] for r in rows)[
                math.ceil(0.95 * len(rows)) - 1
            ]

        metrics = {
            m: dict(
                waiter_cpu_median_us=median(r[m]["worker_cpu_ns"] for r in rows) / 1e3,
                finisher_cpu_median_us=median(r[m]["release_cpu_ns"] for r in rows) / 1e3,
                release_to_grant_p95_us=p95(m) / 1e3,
            )
            for m in MODES
        }
        cut = median(1 - r["fifo"]["worker_cpu_ns"] / r["poll-1ms"]["worker_cpu_ns"] for r in rows)
        result[name] = dict(
            metrics=metrics,
            paired_waiter_cpu_saving_median=cut,
            meets_test_gate=cut >= 0.10 and p95("fifo") <= p95("poll-1ms"),
        )
    return dict(
        pairs=48,
        measured_batch_grants=96,
        warmups_excluded=2,
        matching_target_leases=192,
        router_calls=counts,
        by_first_mode=result,
        tested_gate=all(x["meets_test_gate"] for x in result.values()),
        scope="Single pinned Spark CPU core, two slots, one waiting client, fixed 1ms polling comparator and controlled hold times. Real 72B input IDs and independently replayed historical slot decisions, not a new inference, model output or service latency benchmark. FIFO fairness can delay later smaller batches.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["run", "audit"])
    p.add_argument("ref", type=Path)
    p.add_argument("rows", type=Path)
    a = p.parse_args()
    if a.mode == "run":
        need(not a.rows.exists(), "refusing to overwrite")
        out = measure(a.ref)
        audit(out, a.ref)
        with a.rows.open("x") as f:
            json.dump(out, f, indent=2, allow_nan=False)
            f.write("\n")
    else:
        print(json.dumps(audit(json.loads(a.rows.read_bytes()), a.ref), indent=2, allow_nan=False))
