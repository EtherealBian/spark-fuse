"""Use real, already-warm slots to test joint versus greedy assignment of two ready requests."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
import math
import os
from pathlib import Path
from statistics import median
from threading import Barrier
import time
from prefix_replay import common, digest, post
from slot_policy import SlotRouter

GROUPS = ("contended", "contended-reversed", "disjoint")
SIZE = 1024
NGEN = 16


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def order(ix, rep):
    return ("greedy", "joint") if (ix + rep) % 2 == 0 else ("joint", "greedy")


def build(tokenize, index, rep):
    group = GROUPS[index]

    def body(s):
        return tokenize((f"Run {rep} {group}: " + s + " ") * 240)[:SIZE]

    base = body("The archive records the field identifier, its unit and a revision number.")
    alt = body("Factory notes list the sampling time and inspection status.")
    marks = [tokenize(c) for c in ("A", "B", "C", "D", "E", "F")]
    require(
        all(len(m) == 1 for m in marks) and len({m[0] for m in marks}) == 6,
        "need six distinct marker tokens",
    )
    a, b, c, d, e, f = [m[0] for m in marks]
    seed0 = base.copy()
    seed0[320] = a
    seed0[640] = c
    seed0[960] = e
    seed1 = base[:320] + [b] + alt[321:]
    seed1[640] = c
    first = (seed1 if group == "disjoint" else seed0)[:640] + [d] + alt[641:]
    second = seed0[:960] + [f] + alt[961:]
    tasks = [first, second] if group != "contended-reversed" else [second, first]
    return dict(seeds=[seed0, seed1], tasks=tasks)


def complete(port, ids, reuse, n, *, slot):
    eog = json.loads(os.environ["SPARKFUSE_EOG_IDS"])
    require(
        isinstance(eog, list)
        and eog
        and all(type(t) is int and 0 <= t < 152064 for t in eog)
        and eog == sorted(set(eog)),
        "invalid runtime EOG",
    )
    payload = dict(
        prompt=ids,
        id_slot=slot,
        cache_prompt=reuse,
        n_predict=n,
        temperature=0,
        seed=1513,
        ignore_eos=True,
        logit_bias=[[t, False] for t in eog],
        return_tokens=True,
        stream=False,
    )
    start = time.perf_counter_ns()
    v = post(port, "completion", payload)
    ms = (time.perf_counter_ns() - start) / 1e6
    require(type(v["id_slot"]) is int and v["id_slot"] == slot, "response from unexpected slot")
    t = v["timings"]
    return dict(
        input_sha256=digest(ids),
        observed_slot=v["id_slot"],
        masked_eog=eog,
        cache_prompt=reuse,
        ids=v["tokens"],
        stop=v["stop_type"],
        predicted=t["predicted_n"],
        processed=t["prompt_n"],
        reused=t["cache_n"],
        prompt_ms=t["prompt_ms"],
        decode_ms=t["predicted_ms"],
        wall_ms=ms,
    )


def run(port, save):
    def tok(s):
        return post(port, "tokenize", dict(content=s, add_special=False, parse_special=False))[
            "tokens"
        ]

    out = []
    for ix, group in enumerate(GROUPS):
        for rep in (0, 1):
            spec = build(tok, ix, rep)
            case = dict(group=group, rep=rep, inputs=spec, arms=[])
            out.append(case)
            for mode in order(ix, rep):
                router = SlotRouter(2, 4096, 16)
                ls = [router.acquire("one", ids) for ids in spec["seeds"]]
                pr = []
                for slot, lease in enumerate(ls):
                    require(lease.slot == slot and not lease.reuse, "seed assignment")
                    pr.append(complete(port, spec["seeds"][slot], False, 1, slot=slot))
                    router.finish(lease, success=True)
                epoch = time.perf_counter_ns()
                leases = router.acquire_batch(
                    [("one", ids) for ids in spec["tasks"]], strategy=mode
                )
                epoch_assign = time.perf_counter_ns() - epoch
                gate = Barrier(2)

                def req(j):
                    lease = leases[j]
                    gate.wait(timeout=10)
                    start = time.perf_counter_ns() - epoch
                    try:
                        w = complete(port, spec["tasks"][j], lease.reuse, NGEN, slot=lease.slot)
                    except Exception:
                        router.finish(lease, success=False)
                        raise
                    end = time.perf_counter_ns() - epoch
                    router.finish(lease, success=True)
                    return dict(
                        slot=lease.slot,
                        estimated_prefix=lease.prefix,
                        start_ns=start,
                        end_ns=end,
                        input_sha256=digest(spec["tasks"][j]),
                        response=w,
                    )

                with ThreadPoolExecutor(max_workers=2) as pool:
                    rr = list(pool.map(req, (0, 1)))
                case["arms"].append(
                    dict(mode=mode, preparations=pr, assignment_ns=epoch_assign, requests=rr)
                )
                save(out)
                print(group, rep, mode, [r["response"]["reused"] for r in rr], flush=True)
    return out


def validate(w, ids, slot, hit, count, eog):
    n = len(ids)
    require(
        type(w["observed_slot"]) is int
        and w["observed_slot"] == slot
        and w["input_sha256"] == digest(ids)
        and isinstance(w["masked_eog"], list)
        and all(type(x) is int for x in w["masked_eog"])
        and w["masked_eog"] == eog
        and w["cache_prompt"] is bool(hit),
        "invalid slot, input or request controls",
    )
    require(
        all(type(w[k]) is int for k in ("processed", "reused", "predicted"))
        and (w["processed"], w["reused"], w["predicted"]) == (n - hit, hit, count),
        "observed backend work differs from assigned boundary",
    )
    require(
        w["stop"] == "limit"
        and len(w["ids"]) == count
        and all(type(t) is int and 0 <= t < 152064 and t not in eog for t in w["ids"]),
        "non-matched token stream",
    )
    require(
        all(
            type(w[k]) in (int, float) and math.isfinite(w[k]) and w[k] >= 0
            for k in ("prompt_ms", "decode_ms", "wall_ms")
        )
        and w["prompt_ms"] > 0
        and w["wall_ms"] > 0
        and (count == 1 or w["decode_ms"] > 0)
        and w["wall_ms"] + 3 >= w["prompt_ms"] + w["decode_ms"],
        "invalid timings",
    )


def audit(raw, eog):
    require(
        isinstance(eog, list)
        and eog
        and all(type(t) is int and 0 <= t < 152064 for t in eog)
        and eog == sorted(set(eog)),
        "invalid runtime EOG",
    )
    require(
        [(c["group"], c["rep"]) for c in raw] == [(g, r) for g in GROUPS for r in (0, 1)],
        "incomplete design",
    )
    seen = set()
    groups = {}
    for i, case in enumerate(raw):
        group, rep = case["group"], case["rep"]
        seeds = case["inputs"]["seeds"]
        tasks = case["inputs"]["tasks"]
        require(
            len(seeds) == len(tasks) == 2
            and all(
                len(ids) == SIZE and all(type(v) is int and 0 <= v < 152064 for v in ids)
                for ids in seeds + tasks
            ),
            "invalid prompts",
        )
        h = {digest(ids) for ids in tasks}
        require(len(h) == 2 and seen.isdisjoint(h), "duplicate target prompts")
        seen.update(h)
        m = [[common(t, s) for s in seeds] for t in tasks]
        declared = {
            "contended": [[640, 320], [960, 320]],
            "contended-reversed": [[960, 320], [640, 320]],
            "disjoint": [[320, 640], [960, 320]],
        }
        require(m == declared[group], "wrong workload matrix")
        require(
            [arm["mode"] for arm in case["arms"]] == list(order(i // 2, rep)),
            "missing or out-of-order arm",
        )
        results = {}
        outputs = {}
        for arm in case["arms"]:
            mode = arm["mode"]
            ps = arm["preparations"]
            rr = arm["requests"]
            require(len(ps) == len(rr) == 2, "missing paired requests")
            for slot, w in enumerate(ps):
                validate(w, seeds[slot], slot, 0, 1, eog)
            if mode == "joint":
                assignment = max(
                    itertools.permutations((0, 1)),
                    key=lambda perm: sum(m[j][slot] for j, slot in enumerate(perm)),
                )
            else:
                lead = max(range(2), key=lambda s: m[0][s])
                assignment = (lead, 1 - lead)
            require(
                type(arm["assignment_ns"]) is int
                and arm["assignment_ns"] >= 0
                and all(type(r["slot"]) is int for r in rr)
                and [r["slot"] for r in rr] == list(assignment),
                "offline assignment oracle differs",
            )
            outputs[mode] = []
            for j, r in enumerate(rr):
                hit = m[j][assignment[j]]
                require(
                    r["estimated_prefix"] == hit and r["input_sha256"] == digest(tasks[j]),
                    "incorrect input/estimate",
                )
                w = r["response"]
                validate(w, tasks[j], assignment[j], hit, NGEN, eog)
                outputs[mode].append(w["ids"])
                a, b = r["start_ns"], r["end_ns"]
                require(
                    type(a) is int
                    and type(b) is int
                    and arm["assignment_ns"] <= a < b
                    and (b - a) / 1e6 + 0.01 >= w["wall_ms"],
                    "invalid client clocks",
                )
            overlap = min(r["end_ns"] for r in rr) - max(r["start_ns"] for r in rr)
            require(overlap > 0, "not two simultaneously in-flight clients")
            results[mode] = dict(
                processed=sum(r["response"]["processed"] for r in rr),
                ready_to_finish_ms=max(r["end_ns"] for r in rr) / 1e6,
                overlap_ms=overlap / 1e6,
                assignment_us=arm["assignment_ns"] / 1e3,
            )
        first = [
            next((p for p, (a, b) in enumerate(zip(x, y, strict=True), 1) if a != b), None)
            for x, y in zip(outputs["greedy"], outputs["joint"], strict=True)
        ]
        groups.setdefault(group, []).append(
            dict(rep=rep, arms=results, first_different_token=first)
        )
    return dict(
        groups={
            k: dict(
                repeats=v,
                exact_pairs=sum(x is None for r in v for x in r["first_different_token"]),
                median_paired_ms_saved=median(
                    r["arms"]["greedy"]["ready_to_finish_ms"]
                    - r["arms"]["joint"]["ready_to_finish_ms"]
                    for r in v
                ),
            )
            for k, v in groups.items()
        },
        scope="Two ready, synthetic requests; matched slot preparation excluded from target wall time. Wave includes issuer work and local HTTP, not proof of GPU kernel overlap or production tail latency.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("run", "audit"))
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--port", type=int)
    a = p.parse_args()
    eog = json.loads(a.runtime.read_bytes())["eog_token_ids"]
    if a.mode == "run":
        partial = a.rows.with_name(a.rows.name + ".partial")
        require(not a.rows.exists() and not partial.exists(), "output exists")

        def save(obj):
            tmp = partial.with_suffix(".tmp")
            tmp.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n")
            tmp.replace(partial)

        run(a.port, save)
        partial.replace(a.rows)
    else:
        print(json.dumps(audit(json.loads(a.rows.read_bytes()), eog), indent=2, allow_nan=False))
