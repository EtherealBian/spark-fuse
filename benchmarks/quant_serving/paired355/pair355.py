"""Two independent client requests: serial release vs simultaneous release on the same two-slot runtime."""

import argparse
import concurrent.futures as futures
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import json, math, re, threading, time, urllib.request
from pathlib import Path
from prefix_replay import digest, post

KEYS = (
    "prompt_tokens_total",
    "tokens_predicted_total",
    "n_decode_total",
    "n_busy_slots_per_decode",
    "requests_processing",
    "requests_deferred",
)


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def schedule(block):
    need(type(block) is int and block in (0, 1), "invalid block")
    waves = []
    for rep in range(2):
        for k, n in enumerate((128, 1024) if (block + rep) % 2 == 0 else (1024, 128)):
            flip = (block + rep + (n == 1024)) % 2
            for mode in ("together", "serial") if flip else ("serial", "together"):
                waves.append((rep, n, mode, (rep + block) % 2))
    return waves


def counter_snapshot(raw, epoch):
    need(
        isinstance(epoch, str) and re.fullmatch("[1-9][0-9]{0,18}", epoch),
        "missing metrics process epoch",
    )
    v = {}
    for line in raw.splitlines():
        if not line.startswith("llamacpp:"):
            continue
        parts = line.split()
        key = parts[0][9:]
        if key not in KEYS:
            continue
        need(
            len(parts) == 2
            and key not in v
            and re.fullmatch(r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]{1,2})?", parts[1]),
            "bad metrics value",
        )
        v[key] = parts[1]
    need(set(v) == set(KEYS), "incomplete metrics")
    return {"epoch": epoch, "values": v}


def counters(s):
    v = s["values"]
    counter_snapshot("\n".join(f"llamacpp:{k} {x}" for k, x in v.items()), s["epoch"])
    out = {}
    for k in KEYS:
        d = Decimal(v[k])
        need(d.is_finite() and d >= 0, "invalid counter")
        if k == "n_busy_slots_per_decode":
            continue
        need(
            d == d.to_integral_value() and d < 1000000,
            "rounded or large integer counter unsupported",
        )
        out[k] = int(d)
    calls = out["n_decode_total"]
    ratio = Decimal(v["n_busy_slots_per_decode"])
    need(ratio <= 2 and (calls or ratio == 0), "impossible two-slot average")
    need(out["requests_processing"] == out["requests_deferred"] == 0, "boundary was not idle")
    # C++ prints a float quotient using six significant digits; for values in [0,2]
    # both float32 rounding and text rendering are covered by 6e-6 absolute error.
    radius = Decimal("0.000006")
    lo = max(0, int(((ratio - radius) * calls).to_integral_value(rounding=ROUND_CEILING)))
    hi = min(2 * calls, int(((ratio + radius) * calls).to_integral_value(rounding=ROUND_FLOOR)))
    need(lo == hi, "busy-slot numerator is ambiguous or not representable")
    out["busy_slot_sum"] = lo
    return out


def metrics(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=20) as r:
        return counter_snapshot(r.read(65537).decode(), r.headers["Process-Start-Time-Unix"])


def completion(port, ids, slot, mask, n=16):
    r = post(
        port,
        "completion",
        dict(
            prompt=ids,
            id_slot=slot,
            cache_prompt=False,
            n_predict=n,
            temperature=0,
            seed=1513,
            return_tokens=True,
            stream=False,
            ignore_eos=True,
            logit_bias=[[t, False] for t in mask],
        ),
    )
    t = r["timings"]
    return dict(
        slot=r["id_slot"],
        input_sha256=digest(ids),
        ids=r["tokens"],
        stop=r["stop_type"],
        truncated=r["truncated"],
        processed=t["prompt_n"],
        reused=t["cache_n"],
        predicted=t["predicted_n"],
        prompt_ms=t["prompt_ms"],
        decode_ms=t["predicted_ms"],
    )


def wave(port, inputs, mask, rep, n, mode, lead):
    before = metrics(port)
    shared = {}
    waitsecond = threading.Event()
    barrier = threading.Barrier(
        3, action=lambda: shared.update(origin=time.monotonic_ns()), timeout=30
    )

    def client(j):
        barrier.wait()
        if mode == "serial" and j != lead:
            need(waitsecond.wait(245), "lead request did not finish")
        start = time.monotonic_ns() - shared["origin"]
        slot = j ^ lead
        try:
            row = completion(port, inputs[f"{n}:{j}"], slot, mask)
            end = time.monotonic_ns() - shared["origin"]
            return dict(case=f"{n}:{j}", start_ns=start, end_ns=end, **row)
        finally:
            if j == lead:
                waitsecond.set()

    with futures.ThreadPoolExecutor(max_workers=2) as pool:
        handles = [pool.submit(client, j) for j in range(2)]
        barrier.wait()
        rows = [f.result() for f in handles]
    return dict(
        rep=rep, length=n, mode=mode, lead=lead, before=before, after=metrics(port), rows=rows
    )


def run(port, raw, mask, block, p):
    need(
        not any(x.exists() for x in [p, p.with_suffix(".partial"), p.with_suffix(".tmp")]),
        "output exists",
    )
    inputs = raw["inputs"]
    warm = []
    for slot in (0, 1):
        warm.append(completion(port, inputs["128:0"], slot, mask, 3))
    result = dict(schema="glm355.paired-waves.v1", block=block, warm=warm, waves=[])
    for args in schedule(block):
        w = wave(port, inputs, mask, *args)
        result["waves"].append(w)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
        tmp.replace(p.with_suffix(".partial"))
        print("completed", args, flush=True)
    p.with_suffix(".partial").replace(p)


def validate_completion(row, ids, mask, slot, n):
    need(
        type(row["slot"]) is int and row["slot"] == slot and row["input_sha256"] == digest(ids),
        "wrong slot or prompt",
    )
    need(
        row["stop"] == "limit"
        and row["truncated"] is False
        and all(type(x) is int for x in (row["processed"], row["reused"], row["predicted"]))
        and row["processed"] == len(ids)
        and row["reused"] == 0
        and row["predicted"] == len(row["ids"]) == n,
        "different work",
    )
    need(
        all(type(i) is int and 0 <= i < 151552 and i not in mask for i in row["ids"]),
        "invalid token",
    )
    need(
        all(
            type(v) in (int, float) and math.isfinite(v) and v > 0
            for v in (row["prompt_ms"], row["decode_ms"])
        ),
        "invalid timings",
    )


def reduce_block(data, raw, mask, block):
    inputs = raw["inputs"]
    expected = {r["case"]: r["ids"] for r in raw["rows"] if r["rep"] == 0}
    need(
        data["schema"] == "glm355.paired-waves.v1"
        and type(data["block"]) is int
        and data["block"] == block
        and len(data["warm"]) == 2
        and len(data["waves"]) == 8,
        "incomplete or reordered launch",
    )
    for slot, r in enumerate(data["warm"]):
        validate_completion(r, inputs["128:0"], mask, slot, 3)
    last = None
    summary = []
    for (rep, n, mode, lead), w in zip(schedule(block), data["waves"], strict=True):
        need(
            all(type(w[k]) is int for k in ("rep", "length", "lead"))
            and (w["rep"], w["length"], w["mode"], w["lead"]) == (rep, n, mode, lead),
            "wrong schedule",
        )
        a, b = w["before"], w["after"]
        need(a["epoch"] == b["epoch"] and (last is None or last == a), "process/boundary changed")
        last = b
        v0, v1 = counters(a), counters(b)
        dc = v1["n_decode_total"] - v0["n_decode_total"]
        busy = v1["busy_slot_sum"] - v0["busy_slot_sum"]
        need(
            dc > 0
            and dc <= busy <= 2 * dc
            and v1["tokens_predicted_total"] - v0["tokens_predicted_total"] == 32
            and v1["prompt_tokens_total"] - v0["prompt_tokens_total"] == 2 * n,
            "foreign work or invalid counters",
        )
        rows = w["rows"]
        need(len(rows) == 2, "incomplete wave")
        exact = []
        for j, r in enumerate(rows):
            case = f"{n}:{j}"
            need(r["case"] == case, "out-of-order case")
            validate_completion(r, inputs[case], mask, j ^ lead, 16)
            need(
                all(type(r[k]) is int for k in ("start_ns", "end_ns"))
                and 0 <= r["start_ns"] < r["end_ns"],
                "invalid client interval",
            )
            need(
                (r["end_ns"] - r["start_ns"]) / 1e6 + 4 >= r["prompt_ms"] + r["decode_ms"],
                "client interval shorter than backend timings",
            )
            exact.append(r["ids"] == expected[case])
        overlap = min(r["end_ns"] for r in rows) - max(r["start_ns"] for r in rows)
        need(
            (rows[lead]["end_ns"] <= rows[1 - lead]["start_ns"] and busy == dc)
            if mode == "serial"
            else overlap > 0,
            "dispatch contract not observed",
        )
        summary.append(
            dict(
                rep=rep,
                length=n,
                mode=mode,
                lead=lead,
                first_ms=min(r["end_ns"] for r in rows) / 1e6,
                makespan_ms=max(r["end_ns"] for r in rows) / 1e6,
                overlap_ms=max(0, overlap) / 1e6,
                decode_calls=dc,
                busy_slot_sum=busy,
                excess_busy_slots=busy - dc,
                baseline_exact=exact,
                ids=[x["ids"] for x in rows],
            )
        )
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("port", type=int)
    p.add_argument("block", type=int)
    p.add_argument("capacity", type=Path)
    p.add_argument("out", type=Path)
    u = p.parse_args()
    raw = json.loads((u.capacity / "rows.json").read_bytes())
    mask = json.loads((u.capacity / "runtime.json").read_bytes())["eog_token_ids"]
    run(u.port, raw, mask, u.block, u.out)
