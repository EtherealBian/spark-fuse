"""Source- and history-bound CPU replay of actual SSE arrival, two leases and HTTP responses."""

import json
from pathlib import Path
from statistics import median
from admission_trace import inputs, identity, PIN, MODES, sha
from batch_trace import validate
from audit_bundle import attest, ROOT
from prefix_replay import digest
from physical_guard import accepted


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def sources():
    live = Path(__file__).resolve().parent
    a = live.parent
    b = ROOT
    return {
        p.name: sha(p)
        for p in (
            live / "live72.py",
            live / "live72_review.py",
            a / "slot_admission.py",
            a / "admission_trace.py",
            *(
                b / n
                for n in (
                    "batch_trace.py",
                    "audit_batch.py",
                    "stream_wave.py",
                    "prefix_replay.py",
                    "slot_policy.py",
                    "prefix_index.py",
                    "physical_guard.py",
                    "audit_bundle.py",
                    "runtime_receipt.py",
                    "verify_lock.py",
                )
            ),
        )
    }


def order(i, rep):
    return MODES if (i + rep) % 2 == 0 else MODES[::-1]


def num(x):
    return type(x) is int and x > 0


def check_arm(r, c, m, eog):
    hist = next(x for x in c["arms"] if x["mode"] == "joint")
    seeds = c["inputs"]["seeds"]
    validate(r["preparation"], seeds[0], 0, 0, 1, eog)
    b = r["blocker"]
    s = b["response"]
    ev = s["events"]
    times = [s["start_ns"], *(x[0] for x in ev), s["end_ns"]]
    cnt = s["partial_slot_ids"]
    vals = [x[1] for x in ev]
    need(
        s["terminal_slot"] == 1
        and type(s["terminal_slot"]) is int
        and s["input_sha256"] == digest(seeds[1])
        and s["masked_eog"] == eog
        and all(type(x) is int for x in s["masked_eog"])
        and s["stop"] == "limit"
        and type(s["predicted"]) is int
        and type(s["processed"]) is int
        and type(s["cached"]) is int
        and (s["predicted"], s["processed"], s["cached"]) == (16, 1024, 0)
        and len(vals) == 16
        and all(len(x) == 2 for x in ev)
        and all(type(t) is int and 0 <= t < 152064 and t not in eog for t in vals)
        and set(cnt) <= {"-1", "1"}
        and all(num(x) for x in cnt.values())
        and sum(cnt.values()) == 16,
        "unbound streaming blocker",
    )
    need(
        all(num(x) for x in times)
        and times == sorted(times)
        and times[0] < times[1]
        and ev[0][0]
        <= r["arrival_ns"]
        < ev[-1][0]
        <= s["end_ns"]
        <= b["release_start_ns"]
        <= b["release_end_ns"]
        and all(
            num(k)
            for k in (
                r["arrival_ns"],
                r["grant_ns"],
                r["waiter_cpu_ns"],
                r["calls"],
                b["release_start_ns"],
                b["release_end_ns"],
                b["release_cpu_ns"],
            )
        )
        and b["release_start_ns"] <= r["grant_ns"],
        "target did not wait across real blocker token events",
    )
    snap = r["broker"]
    calls = r["calls"]
    if m == "fifo":
        need(
            calls == 1
            and isinstance(snap, dict)
            and snap["closed"] is False
            and all(
                type(snap[k]) is int
                for k in (
                    "active_slots",
                    "waiting_batches",
                    "waiting_tokens",
                    "overloads",
                    "timeouts",
                    "closed_waiters",
                    "router_calls",
                    "queued_batches",
                    "granted_batches",
                    "wait_calls",
                )
            )
            and [
                snap[k]
                for k in (
                    "active_slots",
                    "waiting_batches",
                    "waiting_tokens",
                    "overloads",
                    "timeouts",
                    "closed_waiters",
                )
            ]
            == [0] * 6
            and (snap["router_calls"], snap["granted_batches"], snap["queued_batches"]) == (2, 2, 1)
            and snap["wait_calls"] >= 1,
            "unverified FIFO queue or unfinished lease",
        )
        calls = snap["router_calls"] - 1
    else:
        need(m == "poll-1ms" and snap is None and calls >= 2, "unverified polling contention")
    rr = r["targets"]
    ts = []
    hist_matches = (
        hist["preparations"][0]["ids"] == r["preparation"]["ids"]
        and hist["preparations"][1]["ids"] == vals[:1]
    )
    for j, (row, (slot, hit, on)) in enumerate(zip(rr, identity(c), strict=True)):
        w = row["response"]
        validate(w, c["inputs"]["tasks"][j], slot, hit, 16, eog)
        start, end = row["start_ns"], row["end_ns"]
        need(
            type(row["slot"]) is int
            and type(row["prefix"]) is int
            and (row["slot"], row["prefix"]) == (slot, hit)
            and on
            and num(start)
            and num(end)
            and r["grant_ns"] <= start < end
            and (end - start) / 1e6 + 0.005 >= w["wall_ms"],
            "target slot, dispatch or interval",
        )
        ts.append(w["ids"])
        hist_matches &= w["ids"] == hist["requests"][j]["response"]["ids"]
    need(
        max(x["start_ns"] for x in rr) < min(x["end_ns"] for x in rr),
        "no overlapping target responses",
    )
    return dict(
        waiter_cpu_ns=r["waiter_cpu_ns"],
        release_cpu_ns=b["release_cpu_ns"],
        router_calls=calls,
        arrival_to_last_ns=max(x["end_ns"] for x in rr) - r["arrival_ns"],
        release_to_grant_ns=r["grant_ns"] - b["release_start_ns"],
        arrival_to_grant_ns=r["grant_ns"] - r["arrival_ns"],
        matches_historical_targets_and_seed_heads=hist_matches,
    ), [vals, ts, r["preparation"]["ids"]]


def review(obj, ref, eog):
    cases = inputs(ref)
    need(
        obj["schema"] == "sparkfuse.live-admission.1"
        and obj["history"] == PIN
        and obj["sources"] == sources()
        and type(obj["affinity"]) is list
        and obj["affinity"]
        and all(type(k) is int and k >= 0 for k in obj["affinity"])
        and obj["affinity"] == sorted(set(obj["affinity"]))
        and set(obj["warmup"]) == set(MODES),
        "identity or warmup",
    )
    warm = [check_arm(obj["warmup"][m], cases[0], m, eog) for m in MODES]
    summ = []
    groups = {m: [] for m in MODES}
    record = {}
    for p, (i, rep) in zip(
        obj["pairs"],
        [(i, r) for r in (0, 1) for i in (range(6) if not r else reversed(range(6)))],
        strict=True,
    ):
        need(
            type(p["case"]) is int
            and type(p["repeat"]) is int
            and (p["case"], p["repeat"]) == (i, rep),
            "bad case order",
        )
        arms = order(i, rep)
        rows = {}
        tokens = {}
        for d, m in zip(p["arms"], arms, strict=True):
            need(d["mode"] == m, "bad arm order")
            rows[m], tokens[m] = check_arm(d["row"], cases[i], m, eog)
        matched = tokens[MODES[0]] == tokens[MODES[1]]
        record.setdefault(i, []).extend(tokens.values())
        groups[arms[0]].append(rows)
        summ.append(dict(case=i, repeat=rep, outputs_identical=matched, arms=rows))
    sums = {}
    for first, pairs in groups.items():
        sums[first] = {
            k: {m: median(p[m][k] for p in pairs) for m in MODES}
            for k in (
                "waiter_cpu_ns",
                "release_cpu_ns",
                "arrival_to_last_ns",
                "arrival_to_grant_ns",
                "release_to_grant_ns",
            )
        }
        sums[first]["paired_median_saving"] = {
            k: median(1 - p["fifo"][k] / p["poll-1ms"][k] for p in pairs)
            for k in ("waiter_cpu_ns", "arrival_to_last_ns")
        }
        sums[first]["target_router_calls"] = {
            m: sum(p[m]["router_calls"] for p in pairs) for m in MODES
        }
    stable = all(all(t == v[0] for t in v) for v in record.values())
    matched = stable and all(
        r["outputs_identical"]
        and all(x["matches_historical_targets_and_seed_heads"] for x in r["arms"].values())
        for r in summ
    )
    return dict(
        paired_waves=12,
        timed_blockers=24,
        timed_targets=48,
        untimed_warmup_waves=len(warm),
        identical_outputs_within_and_across_orders=stable,
        matched_including_historical_targets=matched,
        by_first=sums,
        waiter_cpu_gate=matched
        and all(s["paired_median_saving"]["waiter_cpu_ns"] >= 0.10 for s in sums.values()),
        end_to_end_gate=matched
        and all(s["paired_median_saving"]["arrival_to_last_ns"] >= 0.03 for s in sums.values()),
        pairs=summ,
        scope="One live blocked waiter, 1 or 2 occupied slots, six historical prompt pairs. CPU consumption and arrival-to-last completed HTTP request have separate gates. Warm 72B single server, one streaming blocker, no generalized throughput or p99 claim.",
    )


def bundle(run, ref):
    def j(k):
        return json.loads((run / k).read_bytes())

    ref_rt = json.loads((ref / "runtime.json").read_bytes())
    cp = j("checkpoint.json")
    rt = j("runtime.json")
    sig = attest(ROOT / "qwen72b-q4km.lock.json", cp, rt, 81, 2, 128)
    need(
        cp == json.loads((ref / "checkpoint.json").read_bytes())
        and rt == ref_rt
        and cp["total_bytes"] == 44010465472,
        "pinned 72B runtime/shards mismatch",
    )
    a = j("admission.json")
    need(
        a["admitted"] is True
        and all(type(a[k]) is int for k in ("available", "required", "free_disk", "required_disk"))
        and a["available"] >= a["required"] == 64 * 2**30
        and a["free_disk"] >= a["required_disk"] == 2 * 2**30,
        "launch admission",
    )
    m = j("memory-guard.json")
    accepted(m)
    return dict(
        runtime=sig,
        verified_bytes=cp["total_bytes"],
        memory_guard=m,
        measurement=review(j("rows.json"), ref, rt["eog_token_ids"]),
        evidence_sha256={
            k: sha(run / k)
            for k in (
                "runtime.json",
                "checkpoint.json",
                "memory-guard.json",
                "admission.json",
                "rows.json",
            )
        },
    )


if __name__ == "__main__":
    import sys

    print(json.dumps(bundle(*map(Path, sys.argv[1:])), indent=2, allow_nan=False))
