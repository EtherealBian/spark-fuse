"""Held-out cold streaming panel. Client event timing is not GPU kernel timing."""

import argparse, hashlib, json, math, sys, time
from pathlib import Path
from statistics import median
from urllib.request import Request, urlopen
from prefix_replay import digest, post

CAP = 48
KEYS = ("512:key", "512:reason", "2048:key", "2048:reason")
BASE = dict(
    n_predict=CAP, temperature=0, seed=1513, id_slot=0, cache_prompt=False, return_tokens=True
)
# No stop string, EOG suppression, lookahead, or model-specific grammar.
STEM = {
    "key": "Audit note: a parcel was checked at the station; the reading was archived. ",
    "reason": "A maintenance journal lists the unit, an observation, and its recorded time. ",
}
TAIL = {
    "key": "The last verified shelf label was AMBER. What is the shelf label? Reply with only the label.",
    "reason": "Why keep past observations when repairing a machine? Answer in one short English sentence.",
}
SLOTS = (128, 256, 256, 128)


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def integer(x):
    return type(x) is int


def token(x):
    return integer(x) and 0 <= x < 152064


def recipe(port):
    def tok(s, special=False):
        return post(port, "tokenize", dict(content=s, add_special=False, parse_special=special))[
            "tokens"
        ]

    head = tok("<|im_start|>user\n", True)
    tail = tok("<|im_end|>\n<|im_start|>assistant\n", True)
    need(
        head.count(151644) == 1 and tail.count(151644) == tail.count(151645) == 1,
        "unexpected role-token encoding",
    )
    spec = {}
    for k in KEYS:
        n, f = k.split(":")
        end = tok("\n" + TAIL[f]) + tail
        body = tok(STEM[f] * 350)
        p = int(n) - len(head) - len(end)
        need(0 < p < len(body), "token budget")
        spec[k] = head + body[:p] + end
    return spec


ORACLE_SHA256 = "c4b6ba00ae8ef501f53d25a2e880542ac6694dee28ed77c81fa46f2fcc8f41cb"


def verify_inputs(spec, eog):
    data = Path(__file__).with_name("input-contract.json").read_bytes()
    need(hashlib.sha256(data).hexdigest() == ORACLE_SHA256, "frozen oracle changed")
    c = json.loads(data)
    need(
        c["eog_token_ids"] == eog
        and c["inputs"] == {k: dict(tokens=len(v), sha256=digest(v)) for k, v in spec.items()},
        "native inputs differ from the independent tokenizer",
    )


def profile(stream):
    return dict(**BASE, stream=stream)


def terminal(o):
    t = o["timings"]
    need(
        o["stop"] is True
        and o["truncated"] is False
        and type(o["id_slot"]) is int
        and o["id_slot"] == 0
        and type(o["index"]) is int
        and o["index"] == 0,
        "invalid native terminal frame",
    )
    return dict(
        predicted=t["predicted_n"],
        processed=t["prompt_n"],
        cached=t["cache_n"],
        stop=o["stop_type"],
        tokens_predicted=o["tokens_predicted"],
        tokens_evaluated=o["tokens_evaluated"],
        prompt_ms=t["prompt_ms"],
    )


def collect(lines, now):
    events = []
    final = None
    done = False
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith(b":"):
            continue
        need(line.startswith(b"data:") and not done, "non-SSE data or bytes after DONE")
        b = line[5:].strip()
        if b == b"[DONE]":
            need(final is not None, "early DONE")
            done = True
            continue
        ns = now()
        o = json.loads(b)
        need(final is None and type(o.get("stop")) is bool, "unexpected frame")
        if o["stop"]:
            need(o.get("tokens") == [] and o.get("content") == "", "terminal duplicated output")
            final = dict(end_ns=ns, **terminal(o))
        else:
            need(
                type(o.get("index")) is int
                and o["index"] == 0
                and type(o.get("id_slot")) is int
                and o["id_slot"] in (-1, 0)
                and "prompt_progress" not in o
                and isinstance(o["tokens"], list)
                and len(o["tokens"]) == 1
                and type(o.get("content")) is str,
                "invalid partial",
            )
            events.append([ns, o["tokens_predicted"], o["tokens"][0], o["content"]])
    need(final is not None, "stream closed without terminal")
    return dict(events=events, **final)


def invoke(port, ids, stream):
    body = profile(stream)
    payload = dict(prompt=ids, **body)
    req = Request(
        f"http://127.0.0.1:{port}/completion",
        json.dumps(payload).encode(),
        {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        },
    )
    start = time.perf_counter_ns()
    now = lambda: time.perf_counter_ns() - start
    with urlopen(req, timeout=180) as resp:
        if stream:
            need(resp.headers.get_content_type() == "text/event-stream", "not SSE")
            r = collect(resp, now)
        else:
            o = json.load(resp)
            r = dict(ids=o["tokens"], content=o["content"], **terminal(o))
    return dict(profile=body, input_sha256=digest(ids), **r)


def run(port, save, eog):
    spec = recipe(port)
    verify_inputs(spec, eog)
    invoke(port, [578, 220, 16, 10, 16], False)
    out = dict(inputs=spec, warmup_excluded=1, witnesses={}, streams=[])
    for rep, order in enumerate((KEYS, KEYS[::-1])):
        for k in order:
            out["streams"].append(dict(case=k, rep=rep, **invoke(port, spec[k], True)))
            save(out)
    # Independent non-streaming requests, never included in latency; they expose all sampled IDs.
    for k in KEYS:
        out["witnesses"][k] = invoke(port, spec[k], False)
        save(out)
    return out


def basecheck(r, spec, k, stream):
    n = int(k.split(":")[0])
    need(
        r["profile"] == profile(stream)
        and json.dumps(r["profile"], sort_keys=True) == json.dumps(profile(stream), sort_keys=True)
        and r["input_sha256"] == digest(spec[k]),
        "wrong request or policy",
    )
    need(
        all(
            integer(r[f])
            for f in ("predicted", "tokens_predicted", "processed", "tokens_evaluated", "cached")
        )
        and r["processed"] == r["tokens_evaluated"] == len(spec[k]) == n
        and r["cached"] == 0
        and 1 <= r["predicted"] == r["tokens_predicted"] <= CAP
        and (r["stop"] == "eos" or (r["stop"] == "limit" and r["predicted"] == CAP)),
        "invalid cold counts or stop",
    )
    need(
        type(r["prompt_ms"]) in (int, float)
        and math.isfinite(r["prompt_ms"])
        and r["prompt_ms"] > 0,
        "prompt timing",
    )


def audit(o, eog):
    spec = o["inputs"]
    need(
        o["warmup_excluded"] == 1
        and set(spec) == set(o["witnesses"]) == set(KEYS)
        and len({digest(s) for s in spec.values()}) == 4
        and len(o["streams"]) == 8
        and all(all(token(t) for t in ids) for ids in spec.values()),
        "incomplete panel",
    )
    results = {k: [] for k in KEYS}
    for k, w in o["witnesses"].items():
        basecheck(w, spec, k, False)
        ids = w["ids"]
        need(
            type(w["content"]) is str
            and isinstance(ids, list)
            and len(ids) == w["predicted"]
            and all(token(t) for t in ids)
            and not (set(ids[:-1]) & set(eog))
            and (ids[-1] in eog) == (w["stop"] == "eos"),
            "invalid reference ID sequence",
        )
    for i, (k, r) in enumerate(zip(KEYS + KEYS[::-1], o["streams"], strict=True)):
        need(r["case"] == k and integer(r["rep"]) and r["rep"] == i // 4, "wrong case/order")
        basecheck(r, spec, k, True)
        w = o["witnesses"][k]
        prev_t = 0
        prev_j = 0
        first = None
        parts = []
        idx = []
        ev = r["events"]
        end = r["end_ns"]
        need(integer(end) and end > 0 and isinstance(ev, list), "bad clock")
        for t, j, id_, text in ev:
            need(
                integer(t)
                and prev_t <= t <= end
                and t > 0
                and integer(j)
                and prev_j < j <= r["predicted"]
                and token(id_)
                and type(text) is str,
                "bad event",
            )
            need(id_ not in eog or (j == r["predicted"] and r["stop"] == "eos"), "early EOG")
            prev_t = t
            prev_j = j
            idx.append((j, id_))
            parts.append(text)
            if text.strip() and first is None:
                first = t
        eq = (
            r["predicted"] == len(w["ids"])
            and r["stop"] == w["stop"]
            and all(j <= len(w["ids"]) and w["ids"][j - 1] == t for j, t in idx)
            and "".join(parts) == w["content"]
        )
        complete = len(ev) == r["predicted"]
        results[k].append(
            dict(
                first_visible_ms=None if first is None else first / 1e6,
                first_id_ms=ev[0][0] / 1e6 if ev else None,
                terminal_ms=end / 1e6,
                delivery_matches_witness=eq,
                all_sampled_ids_observed=complete,
                delivered_ids=len(ev),
                backend_predicted=r["predicted"],
                stop=r["stop"],
            )
        )
    return results


def compare(sets, eog):
    need(len(sets) == 4, "requires four fresh launches")
    panels = [audit(v, eog) for v in sets]
    need(all(v["inputs"] == sets[0]["inputs"] for v in sets), "different input IDs across tiles")
    result = {}
    for k in KEYS:
        ref = [s["witnesses"][k] for s in sets]
        exact = all(
            (w["ids"], w["content"], w["stop"])
            == (ref[0]["ids"], ref[0]["content"], ref[0]["stop"])
            for w in ref
        )
        obs = all(
            q["delivery_matches_witness"] and q["all_sampled_ids_observed"]
            for panel in panels
            for q in panel[k]
        )
        orders = []
        for a, b in ((0, 1), (3, 2)):
            pair = {}
            for field in ("first_visible_ms", "terminal_ms"):
                xs = [q[field] for q in panels[a][k]]
                ys = [q[field] for q in panels[b][k]]
                pair[field] = dict(
                    control=xs,
                    candidate=ys,
                    median_paired_saving=None
                    if None in xs + ys
                    else median(1 - y / x for x, y in zip(xs, ys, strict=True)),
                )
            orders.append(pair)
        result[k] = dict(
            case_sensitive_label_match_by_block=[w["content"].strip() == "AMBER" for w in ref]
            if k.endswith(":key")
            else None,
            witnesses_exact=exact,
            all_streams_fully_observed_and_exact=obs,
            actual_eos_in_all=all(w["stop"] == "eos" for w in ref),
            crossed_orders=orders,
        )
    timing = all(
        v is not None and v >= 0.05
        for c in result.values()
        for order in c["crossed_orders"]
        for v in [order["first_visible_ms"]["median_paired_saving"]]
    ) and all(
        v >= -0.03
        for c in result.values()
        for order in c["crossed_orders"]
        for v in [order["terminal_ms"]["median_paired_saving"]]
    )
    exact = all(
        c["witnesses_exact"] and c["all_streams_fully_observed_and_exact"] for c in result.values()
    )
    natural = any(c["actual_eos_in_all"] for c in result.values())
    return dict(
        cases=result,
        block_observations=panels,
        timing_gate=timing,
        exact_output_gate=exact,
        observed_eos_gate=natural,
        qualified_for_tested_inputs=timing and exact and natural,
        scope="Native cold SSE; first text containing a non-whitespace character vs terminal latency, natural EOG enabled, 48-token safety cap. Complete non-stream witnesses are excluded from timing. Missing SSE IDs are reported, never reconstructed as delivered. No production default or general-quality claim.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["run", "audit"])
    p.add_argument("--block", type=int, required=True, choices=range(4))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--port", type=int)
    a = p.parse_args()
    rt = json.loads((a.out / "runtime.json").read_bytes())
    need(rt["context"]["n_ubatch"] == SLOTS[a.block], "wrong tile")
    dest = a.out / "rows.json"
    partial = dest.with_suffix(".partial")
    temp = dest.with_suffix(".tmp")
    if a.mode == "run":
        need(not any(t.exists() for t in (dest, partial, temp)), "not fresh")

        def save(obj):
            temp.write_text(
                json.dumps(obj, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n"
            )
            temp.replace(partial)

        r = run(a.port, save, rt["eog_token_ids"])
        audit(r, rt["eog_token_ids"])
        partial.replace(dest)
    print(
        json.dumps(
            audit(json.loads(dest.read_bytes()), rt["eog_token_ids"]), indent=2, allow_nan=False
        )
    )
