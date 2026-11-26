"""Run a contended two-slot wave; the first real SSE token triggers arrival, not a sleep."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from threading import Barrier, Event
from time import perf_counter_ns, thread_time_ns, sleep
from urllib.request import Request, urlopen
from admission_trace import inputs, PIN, MODES
from batch_trace import complete
from prefix_replay import digest
from stream_wave import stream
from slot_admission import BatchAdmission
from slot_policy import SlotRouter
from live72_review import order, sources, need


def blocking_request(port, ids, eog, now, first):
    body = dict(
        prompt=ids,
        n_predict=16,
        cache_prompt=False,
        id_slot=1,
        temperature=0,
        seed=1513,
        ignore_eos=True,
        logit_bias=[[t, False] for t in eog],
        stream=True,
        return_tokens=True,
    )
    req = Request(
        f"http://127.0.0.1:{port}/completion",
        json.dumps(body).encode(),
        {"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    begin = now()
    with urlopen(req, timeout=180) as response:
        need(response.headers.get_content_type() == "text/event-stream", "expected SSE")

        def watched():
            for wire in response:
                yield wire  # stream() checks framing before requesting the next line
                if not first.is_set() and wire.strip().startswith(b"data:"):
                    data = wire.strip()[5:].strip()
                    if data != b"[DONE]" and json.loads(data)["stop"] is False:
                        first.set()

        w = stream(watched(), now, 1)
    return dict(start_ns=begin, input_sha256=digest(ids), masked_eog=eog, **w)


def wave(case, mode, call, block, eog):
    broker = (
        BatchAdmission(2, 4096, max_waiters=1, max_waiting_tokens=2048) if mode == "fifo" else None
    )
    router = broker if broker else SlotRouter(2, 4096)
    reserve = (lambda req: broker.reserve(req, timeout_s=120)) if broker else router.acquire_batch
    seeds, tasks = (case["inputs"][key] for key in ("seeds", "tasks"))
    ss = reserve([("one", ids) for ids in seeds])
    need([s.slot for s in ss] == [0, 1], "seed slot order")
    prep_ok = False
    try:
        pre = call(seeds[0], False, 1, slot=0)
        prep_ok = True
    finally:
        router.finish(ss[0], success=prep_ok)
    first = Event()
    origin = perf_counter_ns()

    def now():
        return perf_counter_ns() - origin

    def holder():
        ok = False
        try:
            response = block(seeds[1], eog, now, first)
            ok = True
        finally:
            release = now()
            c = thread_time_ns()
            router.finish(ss[1], success=ok)
            cost = thread_time_ns() - c
            ended = now()
        return dict(
            response=response, release_start_ns=release, release_end_ns=ended, release_cpu_ns=cost
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        h = pool.submit(holder)
        # On an early stream failure, surface the backend exception rather than silently polling.
        for _ in range(240):
            if first.wait(0.5):
                break
            if h.done():
                h.result()
                raise ValueError("blocker terminated without data")
        else:
            raise TimeoutError("no first blocker token")
        attempt = 0
        start = now()
        cpu = thread_time_ns()
        req = [("one", ids) for ids in tasks]
        while True:
            attempt += 1
            ls = reserve(req)
            if ls is not None:
                break
            if now() - start > 120_000_000_000:
                raise TimeoutError("poll control expired")
            sleep(0.001)
        used = thread_time_ns() - cpu
        grant = now()
        bar = Barrier(2)

        def target(j):
            lease = ls[j]
            ok = False
            try:
                bar.wait(5)
                begin = now()
                w = call(tasks[j], lease.reuse, 16, slot=lease.slot)
                end = now()
                ok = True
            finally:
                router.finish(lease, success=ok)
            return dict(
                slot=lease.slot, prefix=lease.prefix, start_ns=begin, end_ns=end, response=w
            )

        ts = list(pool.map(target, (0, 1)))
        b = h.result()
    snap = broker.snapshot() if broker else None
    return dict(
        preparation=pre,
        blocker=b,
        arrival_ns=start,
        grant_ns=grant,
        waiter_cpu_ns=used,
        calls=attempt,
        broker=snap,
        targets=ts,
    )


def run(port, ref, out, eog):
    cases = inputs(ref)
    raw = dict(
        schema="sparkfuse.live-admission.1",
        history=PIN,
        sources=sources(),
        affinity=sorted(os.sched_getaffinity(0)),
        warmup={},
        pairs=[],
    )

    def call(ids, hit, n, slot):
        return complete(port, ids, hit, n, slot=slot)

    def blocker(ids, eog, now, first):
        return blocking_request(port, ids, eog, now, first)

    def save():
        p = out.with_name(out.name + ".partial")
        p.write_text(json.dumps(raw, indent=2, allow_nan=False) + "\n")
        p.replace(out)

    # Keep explicit warm-ups so JIT/server first use is never credited to a timed arm.
    for m in MODES:
        raw["warmup"][m] = wave(cases[0], m, call, blocker, eog)
        save()
    for rep in (0, 1):
        for i in range(6) if not rep else reversed(range(6)):
            arms = order(i, rep)
            p = dict(case=i, repeat=rep, arms=[])
            raw["pairs"].append(p)
            for m in arms:
                p["arms"].append(dict(mode=m, row=wave(cases[i], m, call, blocker, eog)))
                save()
            print("completed paired serving wave", rep, i, flush=True)


if __name__ == "__main__":
    import sys

    port, ref, out = sys.argv[1:]
    run(int(port), Path(ref), Path(out), json.loads(os.environ["SPARKFUSE_EOG_IDS"]))
