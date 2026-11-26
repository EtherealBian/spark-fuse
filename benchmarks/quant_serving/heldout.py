"""Matched bursts on disjoint tokenized inputs: explicitly prepare, reuse opportunistically, or always cold."""

import argparse
import json
import math
import time
from pathlib import Path
from statistics import median
from prefix_replay import common, complete, digest, post

DOMAINS = {
    "coast": "Coastal survey card. A tide gauge logs water levels, calibration offsets and instrument status. ",
    "kiln": "Ceramics production note. Firing temperature, dwell time, glaze thickness and lot number are recorded. ",
}
QUESTIONS = (
    "Alpha: Summarize the fields in the record and mention which measurement should be inspected first. ",
    "Bravo: Explain what might make consecutive record values inconsistent even if the instrument is working. ",
    "Delta: Describe a simple check to run before accepting the next report from this process. ",
    "Echo: Identify which missing piece of metadata would most hinder an independent review. ",
)
ARMS = ("cold", "on-demand", "prepared")
LENS = (256, 2048)
OUT = 24


def need(x, message):
    if not x:
        raise ValueError(message)


def arm_order(ix, repeat):
    seq = ARMS if ix % 2 == 0 else ARMS[::-1]
    off = (ix + repeat) % 3
    return list(seq[off:] + seq[:off])


def case_plan():
    return [(n, d) for n in LENS for d in DOMAINS]


def inputs(port, n, domain):
    def tok(text):
        return post(port, "tokenize", dict(content=text, add_special=False, parse_special=False))[
            "tokens"
        ]

    prefix = tok(DOMAINS[domain] * 220)[: n - 32]
    return [prefix + tok(q * 4)[:32] for q in QUESTIONS]


def inspect_inputs(ids, n, excluded):
    need(
        len(ids) == 4
        and all(len(r) == n and all(type(i) is int and 0 <= i < 152064 for i in r) for r in ids),
        "input vocabulary/length",
    )
    hashes = [digest(r) for r in ids]
    need(len(set(hashes)) == 4 and set(hashes).isdisjoint(excluded), "reused prompt")
    need(
        all(common(a, b) == n - 32 for i, a in enumerate(ids) for b in ids[i + 1 :]),
        "branches do not split at designed token boundary",
    )
    return hashes


def run(port, source):
    excluded = {x["target"] for x in source}
    cases = []
    for ix, (n, domain) in enumerate(case_plan()):
        ids = inputs(port, n, domain)
        hashes = inspect_inputs(ids, n, excluded)
        excluded.update(hashes)
        case = dict(length=n, domain=domain, prompts=ids, prompt_sha256=hashes, runs=[])
        cases.append(case)
        for repeat in range(3):
            for arm in arm_order(ix, repeat):
                started = time.perf_counter()
                prime = complete(port, ids[0][:-32], False, 1) if arm == "prepared" else None
                rr = []
                elapsed = []
                for j, prompt in enumerate(ids):
                    rr.append(
                        complete(
                            port, prompt, arm == "prepared" or (j > 0 and arm == "on-demand"), OUT
                        )
                    )
                    elapsed.append(1000 * (time.perf_counter() - started))
                case["runs"].append(
                    dict(
                        arm=arm,
                        repeat=repeat,
                        primer=prime,
                        requests=rr,
                        cumulative_wall_ms=elapsed,
                    )
                )
                print(
                    domain,
                    n,
                    repeat,
                    arm,
                    [r["reused"] for r in rr],
                    round(elapsed[-1]),
                    flush=True,
                )
    return cases


def validate(r, n, hit, generated, eog):
    need(
        all(type(r[k]) is int for k in ("processed", "reused", "predicted"))
        and (r["processed"], r["reused"], r["predicted"]) == (n - hit, hit, generated),
        "counter mismatch",
    )
    need(
        r["stop"] == "limit"
        and len(r["ids"]) == generated
        and all(type(v) is int and 0 <= v < 152064 and v not in eog for v in r["ids"]),
        "token receipt",
    )
    need(
        all(
            type(r[k]) in (int, float) and math.isfinite(r[k]) and r[k] >= 0
            for k in ("prompt_ms", "decode_ms", "wall_ms")
        )
        and r["wall_ms"] > 0
        and r["prompt_ms"] > 0
        and (generated == 1 or r["decode_ms"] > 0)
        and r["wall_ms"] + 3 >= r["prompt_ms"] + r["decode_ms"],
        "latency",
    )


def audit(cases, source, eog):
    seen = {x["target"] for x in source}
    result = []
    need(len(cases) == 4, "coverage")
    for ix, ((n, domain), case) in enumerate(zip(case_plan(), cases, strict=True)):
        need((case["length"], case["domain"]) == (n, domain), "case identity")
        hashes = inspect_inputs(case["prompts"], n, seen)
        seen.update(hashes)
        need(hashes == case["prompt_sha256"], "prompt hashes")
        runs = case["runs"]
        need(
            [(r["repeat"], r["arm"]) for r in runs]
            == [(t, a) for t in range(3) for a in arm_order(ix, t)],
            "Latin-square order",
        )
        d = {(r["repeat"], r["arm"]): r for r in runs}
        for r in runs:
            arm = r["arm"]
            level = 0.0
            need(len(r["requests"]) == len(r["cumulative_wall_ms"]) == 4, "burst length")
            if arm == "prepared":
                validate(r["primer"], n - 32, 0, 1, eog)
                level = r["primer"]["wall_ms"]
            else:
                need(r["primer"] is None, "unexpected preparation")
            for j, (req, total) in enumerate(
                zip(r["requests"], r["cumulative_wall_ms"], strict=True)
            ):
                hit = n - 32 if arm == "prepared" or (arm == "on-demand" and j > 0) else 0
                validate(req, n, hit, OUT, eog)
                need(
                    type(total) in (int, float)
                    and math.isfinite(total)
                    and total + 0.01 >= level + req["wall_ms"],
                    "cumulative clock omits request or primer",
                )
                level = total

        def counters(arm):
            return sum(
                d[t, arm]["requests"][j]["ids"] == d[t, "cold"]["requests"][j]["ids"]
                for t in range(3)
                for j in range(4)
            )

        result.append(
            dict(
                domain=domain,
                length=n,
                cold_repeat_exact_vs_first=sum(
                    d[t, "cold"]["requests"][j]["ids"] == d[0, "cold"]["requests"][j]["ids"]
                    for t in (1, 2)
                    for j in range(4)
                ),
                arms={
                    a: dict(
                        exact_vs_same_repeat_cold=counters(a),
                        median_total_wall_ms=[
                            median(d[t, a]["cumulative_wall_ms"][j] for t in range(3))
                            for j in range(4)
                        ],
                        median_paired_wall_saved_vs_on_demand_ms=[
                            median(
                                d[t, "on-demand"]["cumulative_wall_ms"][j]
                                - d[t, a]["cumulative_wall_ms"][j]
                                for t in range(3)
                            )
                            for j in range(4)
                        ],
                    )
                    for a in ARMS
                },
            )
        )
    return dict(
        schema=1,
        cases=result,
        scope="Serial four-branch bursts; includes explicit primer wall time; 16 distinct synthetic held-out prompts, not natural traffic or a quality evaluation.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("run", "audit"))
    p.add_argument("--port", type=int)
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    a = p.parse_args()
    source = json.loads(a.source.read_bytes())
    eog = set(json.loads(a.runtime.read_bytes())["eog_token_ids"])
    if a.mode == "run":
        a.rows.write_text(
            json.dumps(run(a.port, source), separators=(",", ":"), allow_nan=False) + "\n"
        )
    else:
        print(
            json.dumps(
                audit(json.loads(a.rows.read_bytes()), source, eog), indent=2, allow_nan=False
            )
        )
