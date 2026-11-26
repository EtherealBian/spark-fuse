"""Replay paired cold streaming requests; compare serial dispatch to two in-flight requests."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from statistics import median
import threading
import time
from urllib.request import Request, urlopen
from prefix_replay import complete, digest, post

SHAPES = {"balanced": (512, 512), "mixed": (256, 2048)}
TOKENS = 16


def require(x, reason):
    if not x:
        raise ValueError(reason)


def stream(lines, now, slot):
    events = []
    final = None
    partial_slot_ids = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith(b":"):
            continue
        require(line.startswith(b"data:"), "not a native SSE data record")
        data = line[5:].strip()
        if data == b"[DONE]":
            require(final is not None, "DONE without native stop")
            break
        received = now()
        v = json.loads(data)
        require(
            final is None
            and type(v.get("stop")) is bool
            and type(v.get("id_slot")) is int
            and type(v.get("index")) is int
            and v["index"] == 0,
            "invalid native result framing",
        )
        reported = v["id_slot"]
        if v["stop"]:
            # b8183 send_final_response assigns slot.id. Never infer it from the request.
            require(reported == slot, "terminal slot does not match the assigned slot")
            require(
                v["tokens"] == [] and v["truncated"] is False,
                "unexpected terminal tokens/truncation",
            )
            t = v["timings"]
            final = dict(
                end_ns=received,
                terminal_slot=reported,
                partial_slot_ids=partial_slot_ids,
                stop=v["stop_type"],
                predicted=t["predicted_n"],
                processed=t["prompt_n"],
                cached=t["cache_n"],
            )
        else:
            # In b8183 send_partial_response does not assign id_slot: the base class
            # serializes its -1 sentinel. Correlate by this dedicated HTTP stream and
            # require the actual slot in its terminal record. Reject any other slot.
            require(reported in (-1, slot), "conflicting partial slot")
            require(
                isinstance(v.get("tokens"), list) and len(v["tokens"]) == 1,
                "expected exactly one token per native partial result",
            )
            label = str(reported)
            partial_slot_ids[label] = partial_slot_ids.get(label, 0) + 1
            events.append([received, v["tokens"][0]])
    require(final is not None, "stream closed without final counts")
    return dict(events=events, **final)


def invoke(port, slot, ids, epoch, eog, barrier=None):
    req = Request(
        f"http://127.0.0.1:{port}/completion",
        json.dumps(
            dict(
                prompt=ids,
                n_predict=TOKENS,
                cache_prompt=False,
                id_slot=slot,
                temperature=0,
                seed=1513,
                ignore_eos=True,
                logit_bias=[[i, False] for i in eog],
                stream=True,
                return_tokens=True,
            )
        ).encode(),
        {"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    if barrier is not None:
        barrier.wait(timeout=10)

    def now():
        return time.perf_counter_ns() - epoch

    begin = now()
    with urlopen(req, timeout=150) as response:
        require(
            response.headers.get_content_type() == "text/event-stream", "non-streaming response"
        )
        out = stream(response, now, slot)
    return dict(slot=slot, input_sha256=digest(ids), start_ns=begin, **out)


def specification(port, key, rep):
    def tok(text):
        return post(port, "tokenize", dict(content=text, add_special=False, parse_special=False))[
            "tokens"
        ]

    tasks = []
    for slot, n in enumerate(SHAPES[key]):
        stem = (
            "In transaction journals the owner records the previous revision. "
            if slot == 0
            else "Our queue retries the latest unacknowledged event. "
        )
        suffix = tok(f"\nCase {key} {rep}: explain one reason for this design.")
        base = tok(stem * 350)
        require(0 < len(suffix) < n <= len(base), "bad workload")
        tasks.append(base[: n - len(suffix)] + suffix)
    return tasks


def run(port, checkpoint=None):
    eog = json.loads(os.environ["SPARKFUSE_EOG_IDS"])
    for s in (0, 1):
        complete(port, [87, 43, 73], False, 3, slot=s)
    result = {"inputs": {}, "waves": []}
    for i, key in enumerate(SHAPES):
        for rep in (0, 1):
            spec = specification(port, key, rep)
            result["inputs"][f"{key}:{rep}"] = spec
            for mode in ("serial", "parallel") if (i + rep) % 2 == 0 else ("parallel", "serial"):
                epoch = time.perf_counter_ns()
                # The serial leader flips on repetition so the long request does not always arrive last.
                order = [rep, 1 - rep]
                if mode == "serial":
                    rows = [invoke(port, s, spec[s], epoch, eog) for s in order]
                else:
                    gate = threading.Barrier(2)
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        fs = [
                            pool.submit(invoke, port, s, spec[s], epoch, eog, gate) for s in order
                        ]
                        rows = [f.result() for f in fs]
                result["waves"].append(dict(key=key, rep=rep, mode=mode, rows=rows))
                if checkpoint is not None:
                    checkpoint(result)
                print("wave", key, rep, mode, [len(r["events"]) for r in rows], flush=True)
    return result


def audit(raw, eog):
    waves = iter(raw["waves"])
    stats = {}
    seen = set()
    require(
        set(raw["inputs"]) == {f"{s}:{r}" for s in SHAPES for r in (0, 1)}
        and len(raw["waves"]) == 8,
        "incomplete design",
    )
    for i, key in enumerate(SHAPES):
        samples = {"parallel": [], "serial": []}
        exact = 0
        acceptance = []
        for rep in (0, 1):
            spec = raw["inputs"][f"{key}:{rep}"]
            hashes = [digest(ids) for ids in spec]
            require(
                len(spec) == 2 and len(set(hashes)) == 2 and seen.isdisjoint(hashes),
                "duplicate inputs",
            )
            seen.update(hashes)
            by = {}
            for mode in ("serial", "parallel") if (i + rep) % 2 == 0 else ("parallel", "serial"):
                w = next(waves)
                rows = w["rows"]
                require(
                    (w["key"], w["rep"], w["mode"]) == (key, rep, mode)
                    and [r["slot"] for r in rows] == [rep, 1 - rep],
                    "wrong order/slots",
                )
                ttft = {}
                itl = []
                by[mode] = {}
                for r in rows:
                    s = r["slot"]
                    ids = spec[s]
                    ev = r["events"]
                    expected = SHAPES[key][s]
                    counts = r["partial_slot_ids"]
                    require(
                        type(r["terminal_slot"]) is int
                        and r["terminal_slot"] == s
                        and isinstance(counts, dict)
                        and set(counts) <= {"-1", str(s)}
                        and all(type(x) is int and x > 0 for x in counts.values())
                        and sum(counts.values()) == len(ev),
                        "unbound stream slot/counters",
                    )
                    starts = r["start_ns"]
                    ends = r["end_ns"]
                    require(
                        all(type(r[k]) is int for k in ("slot", "processed", "cached", "predicted"))
                        and len(ids) == r["processed"] == expected
                        and r["cached"] == 0
                        and r["input_sha256"] == hashes[s]
                        and all(type(t) is int and 0 <= t < 152064 for t in ids),
                        "input/counter mismatch",
                    )
                    require(
                        len(ev) == r["predicted"] == TOKENS
                        and r["stop"] == "limit"
                        and all(
                            len(e) == 2
                            and type(e[1]) is int
                            and 0 <= e[1] < 152064
                            and e[1] not in eog
                            for e in ev
                        ),
                        "bad output",
                    )
                    clocks = [starts] + [v[0] for v in ev] + [ends]
                    require(
                        all(type(t) is int and t >= 0 for t in clocks)
                        and clocks == sorted(clocks)
                        and ev[0][0] > starts,
                        "invalid monotonic trace",
                    )
                    ttft[s] = (ev[0][0] - starts) / 1e6
                    itl += [(b[0] - a[0]) / 1e6 for a, b in zip(ev, ev[1:])]
                    by[mode][s] = [e[1] for e in ev]
                first = min(r["start_ns"] for r in rows)
                last = max(r["end_ns"] for r in rows)
                overlap = min(r["end_ns"] for r in rows) - max(r["start_ns"] for r in rows)
                require(
                    (
                        overlap > 0
                        if mode == "parallel"
                        else rows[1]["start_ns"] >= rows[0]["end_ns"]
                    ),
                    "client in-flight interval differs from dispatch policy",
                )
                ms = (last - first) / 1e6
                samples[mode].append(
                    dict(
                        wave_ms=ms,
                        aggregate_output_tps=2 * TOKENS * 1000 / ms,
                        slot0_ttft_ms=ttft[0],
                        slot1_ttft_ms=ttft[1],
                        inter_token_p50_ms=median(itl),
                        client_overlap_ms=max(0, overlap) / 1e6,
                        # Both requests are logically ready at wave epoch zero. Measurement
                        # includes client scheduling and, in serial mode, wait for the leader.
                        ready_to_last_stop_ms=last / 1e6,
                        ready_to_slot0_first_ms=next(
                            r["events"][0][0] for r in rows if r["slot"] == 0
                        )
                        / 1e6,
                        ready_to_slot1_first_ms=next(
                            r["events"][0][0] for r in rows if r["slot"] == 1
                        )
                        / 1e6,
                        slower_ready_to_first_ms=max(r["events"][0][0] for r in rows) / 1e6,
                        start_skew_ms=(max(r["start_ns"] for r in rows) - first) / 1e6,
                        ready_to_last_output_tps=2 * TOKENS * 1e9 / last,
                    )
                )
            pos = {
                str(s): next(
                    (
                        j
                        for j, (a, b) in enumerate(
                            zip(by["serial"][s], by["parallel"][s], strict=True), 1
                        )
                        if a != b
                    ),
                    None,
                )
                for s in (0, 1)
            }
            exact += sum(p is None for p in pos.values())
            ser = samples["serial"][-1]
            par = samples["parallel"][-1]
            acceptance.append(
                dict(
                    rep=rep,
                    serial_leader=rep,
                    first_different_token_by_slot=pos,
                    ready_to_last_saved_ms=ser["ready_to_last_stop_ms"]
                    - par["ready_to_last_stop_ms"],
                    ready_to_first_saved_ms_by_slot={
                        str(s): ser[f"ready_to_slot{s}_first_ms"]
                        - par[f"ready_to_slot{s}_first_ms"]
                        for s in (0, 1)
                    },
                )
            )
        stats[key] = {
            mode: {k: median(v[k] for v in seq) for k in seq[0]} for mode, seq in samples.items()
        }
        stats[key].update(exact_output_pairs=exact, pairs=4, paired_waves=acceptance)
    return dict(
        cases=stats,
        scope="Two fixed-length cold native SSE requests per wave on loopback. TTFT is first token-ID arrival; overlap is client-side in-flight overlap, not proof of GPU kernel concurrency. Ready-to-first starts from a shared client epoch before either request is dispatched; includes serial queuing and client scheduling. Per-request TTFT starts at that request's HTTP send. Two repetitions; no tail-latency or sustained-load estimate.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["run", "audit"])
    p.add_argument("--port", type=int)
    p.add_argument("--rows", required=True, type=Path)
    p.add_argument("--runtime", required=True, type=Path)
    a = p.parse_args()
    eog = json.loads(a.runtime.read_text())["eog_token_ids"]
    if a.mode == "run":
        partial = a.rows.with_name(a.rows.name + ".partial")
        if any(p.exists() for p in (a.rows, partial)):
            raise FileExistsError("refusing to replace existing measured waves")

        def save(data):
            temp = partial.with_name(partial.name + ".tmp")
            temp.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
            temp.replace(partial)

        result = run(a.port, checkpoint=save)
        require(len(result["waves"]) == 8, "unfinished set of waves")
        partial.replace(a.rows)
    else:
        print(json.dumps(audit(json.loads(a.rows.read_text()), eog), indent=2, allow_nan=False))
