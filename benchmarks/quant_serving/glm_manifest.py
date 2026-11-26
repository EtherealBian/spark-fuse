"""Can a warm slot answer from an edited manifest instead of its preceding revision?

A behavioral and latency canary, not a general quality evaluation. No EOG mask,
answer grammar, forced prefix, stop string, or answer-choice logit bias.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import time
from prefix_replay import common, digest, post

SIZES = (16, 96)
PLACEMENTS = ("early", "late")
CODES = (
    "CEDAR",
    "MAPLE",
    "AMBER",
    "IVORY",
    "COBALT",
    "OPAL",
    "SILVER",
    "VIOLET",
    "TEAL",
    "ONYX",
    "JADE",
    "PEARL",
    "BRONZE",
    "OCHRE",
    "CORAL",
    "INDIGO",
)
LIMIT = 12
V = 151552


def need(v, msg):
    if not v:
        raise ValueError(msg)


def tasks():
    for size in SIZES:
        for pos in PLACEMENTS:
            for v in (0, 1):
                yield size, pos, v


def make(size, pos, v):
    need(size in SIZES and pos in PLACEMENTS and type(v) is int and v in (0, 1), "unknown manifest")
    g = SIZES.index(size) * 2 + PLACEMENTS.index(pos)
    j = 2 if pos == "early" else size - 3
    target = (g * 3 + 1) % len(CODES)
    word = CODES[(target + v * 7) % len(CODES)]
    query = None
    lines = []
    for i in range(size):
        key = f"R-{100 + (37 * i + 19 * g) % 251}"
        value = CODES[(5 * i + 3 * g) % len(CODES)] if i != j else word
        if i == j:
            query = key
        lines.append(f"{key} | {value}")
    system = "Read the supplied manifest. Reply with only the uppercase code for the requested record, with no punctuation."
    user = "Manifest (record | code):\n" + "\n".join(lines) + f"\n\nWhat is the code for {query}?"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], word


def label(n, p, v):
    return f"{n}:{p}:{v}"


def arm_order(n, p, v):
    return (
        ("cold", "reuse")
        if (SIZES.index(n) + PLACEMENTS.index(p) + v) % 2 == 0
        else ("reuse", "cold")
    )


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def capture(postfn, msg):
    prompt = postfn("apply-template", {"messages": msg})["prompt"]
    need(
        all(prompt.count(m["content"]) == 1 for m in msg),
        "chat template did not retain both message bodies",
    )
    ids = postfn("tokenize", dict(content=prompt, add_special=False, parse_special=True))["tokens"]
    return dict(
        rendered=prompt, rendered_sha256=sha(prompt), input_ids=ids, input_sha256=digest(ids)
    )


def invoke(postfn, ids, reuse, eog):
    payload = dict(
        prompt=ids,
        n_predict=LIMIT,
        cache_prompt=reuse,
        id_slot=0,
        temperature=0,
        seed=1513,
        stream=False,
        return_tokens=True,
    )
    start = time.perf_counter_ns()
    raw = postfn("completion", payload)
    wall = (time.perf_counter_ns() - start) / 1e6
    tokens = raw["tokens"]
    display = postfn("detokenize", {"tokens": [t for t in tokens if t not in eog]})["content"]
    t = raw["timings"]
    return dict(
        slot=raw["id_slot"],
        input_sha256=digest(ids),
        ids=tokens,
        text=raw["content"],
        decoded=display,
        stop=raw["stop_type"],
        truncated=raw["truncated"],
        evaluated=raw["tokens_evaluated"],
        processed=t["prompt_n"],
        cached=t["cache_n"],
        predicted=t["predicted_n"],
        prompt_ms=t["prompt_ms"],
        decode_ms=t["predicted_ms"],
        wall_ms=wall,
    )


def run(postfn, eog, save):
    entries = {}
    for n, p, v in tasks():
        msg, _ = make(n, p, v)
        entries[label(n, p, v)] = capture(postfn, msg)
    rows = []
    result = dict(schema=1, inputs=entries, rows=rows)
    for n, p, v in tasks():
        key = label(n, p, v)
        prev = label(n, p, 1 - v)
        for arm in arm_order(n, p, v):
            # Both arms first run a real, cold, naturally stopping completion on the stale revision.
            a = invoke(postfn, entries[prev]["input_ids"], False, eog)
            b = invoke(postfn, entries[key]["input_ids"], arm == "reuse", eog)
            rows.append(dict(case=key, arm=arm, primer=a, target=b))
            save(result)
            print("revision", key, arm, b["processed"], b["cached"], flush=True)
    return result


def check_response(r, ids, cache, eog):
    need(
        all(type(r[k]) is int for k in ("slot", "processed", "cached", "predicted", "evaluated"))
        and r["slot"] == 0
        and r["truncated"] is False
        and r["evaluated"] == len(ids)
        and r["processed"] + r["cached"] == len(ids)
        and r["cached"] == cache
        and r["input_sha256"] == digest(ids),
        "slot, cache or length mismatch",
    )
    tok = r["ids"]
    need(
        isinstance(tok, list)
        and 1 <= len(tok) == r["predicted"] <= LIMIT
        and all(type(t) is int and 0 <= t < V for t in tok),
        "invalid generated tokens",
    )
    need(
        r["stop"] in ("eos", "limit")
        and not any(t in eog for t in tok[:-1])
        and ((tok[-1] in eog) == (r["stop"] == "eos"))
        and (r["stop"] != "limit" or len(tok) == LIMIT),
        "natural stop not reconciled",
    )
    need(
        isinstance(r["text"], str) and r["text"] == r["decoded"],
        "native output and independently detokenized IDs differ",
    )
    # Native predicted_ms may be zero for an immediate EOG; no per-token rate for this case.
    need(
        all(
            type(r[k]) in (int, float) and math.isfinite(r[k]) and r[k] >= 0
            for k in ("wall_ms", "decode_ms", "prompt_ms")
        )
        and r["wall_ms"] > 0
        and r["wall_ms"] + 3 >= r["prompt_ms"] + r["decode_ms"],
        "bad timing",
    )


def audit(raw, eog):
    need(
        raw["schema"] == 1
        and raw["schema"] is not True
        and len(raw["rows"]) == 16
        and set(raw["inputs"]) == {label(*t) for t in tasks()},
        "incomplete design",
    )
    need(
        isinstance(eog, (list, tuple, set))
        and bool(eog)
        and all(type(t) is int and 0 <= t < V for t in eog),
        "invalid EOG",
    )
    inputs = raw["inputs"]
    cases = {}
    seen = set()
    cursor = iter(raw["rows"])
    pools = {a: [] for a in ("cold", "reuse")}
    for n, p, v in tasks():
        k = label(n, p, v)
        old = label(n, p, 1 - v)
        msg, expected = make(n, p, v)
        _, stale = make(n, p, 1 - v)
        i = inputs[k]
        b = inputs[old]
        ids = i["input_ids"]
        prev = b["input_ids"]
        text = i["rendered"]
        peer = b["rendered"]
        message_old, _ = make(n, p, 1 - v)
        need(
            isinstance(text, str)
            and text.count(msg[0]["content"]) == text.count(msg[1]["content"]) == 1
            and text.replace(msg[1]["content"], message_old[1]["content"], 1) == peer
            and sha(text) == i["rendered_sha256"],
            "prompt template or manifest substitution",
        )
        need(
            type(ids) is list
            and 0 < len(ids) <= 4096 - LIMIT
            and all(type(t) is int and 0 <= t < V for t in ids)
            and digest(ids) == i["input_sha256"]
            and i["input_sha256"] not in seen,
            "tokenized prompt mismatch",
        )
        seen.add(i["input_sha256"])
        prefix = common(ids, prev)
        need(0 < prefix < min(len(ids), len(prev)), "no measurable edit boundary")
        arms = {}
        outputs = {}
        for arm in arm_order(n, p, v):
            r = next(cursor)
            need((r["case"], r["arm"]) == (k, arm), "unmatched arm/order")
            a = r["primer"]
            c = r["target"]
            check_response(a, prev, 0, eog)
            check_response(c, ids, prefix if arm == "reuse" else 0, eog)
            out = c["text"].strip()
            verdict = "correct" if out == expected else ("stale" if out == stale else "other")
            outputs[arm] = c["ids"]
            aa = dict(
                verdict=verdict,
                prompt_ms=c["prompt_ms"],
                target_wall_ms=c["wall_ms"],
                primer_and_target_wall_ms=a["wall_ms"] + c["wall_ms"],
                actual_cached=c["cached"],
                natural_eos=c["stop"] == "eos",
                primer_correct=a["text"].strip() == stale,
            )
            arms[arm] = aa
            pools[arm].append(aa)
        cases[k] = dict(
            expected=expected,
            stale=stale,
            input_tokens=len(ids),
            first_changed_token_1based=prefix + 1,
            matched_output_ids=outputs["cold"] == outputs["reuse"],
            arms=arms,
        )
    for n in SIZES:
        for v in (0, 1):
            need(
                cases[label(n, "early", v)]["first_changed_token_1based"]
                < cases[label(n, "late", v)]["first_changed_token_1based"],
                "early/late edit geometry not observed",
            )
    totals = {
        a: {
            **{k: sum(r["verdict"] == k for r in seq) for k in ("correct", "stale", "other")},
            "natural_eos": sum(r["natural_eos"] for r in seq),
            "primer_correct": sum(r["primer_correct"] for r in seq),
            "target_wall_median_ms": median(r["target_wall_ms"] for r in seq),
        }
        for a, seq in pools.items()
    }
    return dict(
        cases=cases,
        totals=totals,
        exact_token_pairs=sum(r["matched_output_ids"] for r in cases.values()),
        pairs=len(cases),
        scope="Eight edited synthetic manifests in both cache policies: one paired observation per target, not an accuracy benchmark. Correct/stale/other score the entire whitespace-trimmed answer against this revision and its predecessor. Natural EOS, no forced answer or masked tokens. Target wall excludes tokenization and detokenization; primer+target is also reported. Upstream llama.cpp supplies inference.",
    )


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("mode", choices=["run", "audit"])
    a.add_argument("--rows", type=Path, required=True)
    a.add_argument("--runtime", type=Path, required=True)
    a.add_argument("--port", type=int)
    o = a.parse_args()
    rt = json.loads(o.runtime.read_bytes())
    eog = rt["eog_token_ids"]
    need(
        rt["gpu_layers_including_output"] == 94
        and rt["device"] == "NVIDIA GB10"
        and rt["context"]
        == dict(n_seq_max=1, n_ctx=4096, n_ctx_seq=4096, n_batch=512, n_ubatch=128),
        "unsupported runtime",
    )
    if o.mode == "audit":
        print(json.dumps(audit(json.loads(o.rows.read_bytes()), eog), indent=2, allow_nan=False))
        return
    need(o.port and 1 <= o.port <= 65535, "loopback port required")
    partial = o.rows.with_name(o.rows.name + ".partial")
    temp = partial.with_name(partial.name + ".tmp")
    need(not any(p.exists() for p in (partial, temp, o.rows)), "refusing to overwrite evidence")

    def save(raw):
        temp.write_text(json.dumps(raw, separators=(",", ":"), allow_nan=False) + "\n")
        temp.replace(partial)

    data = run(lambda path, payload: post(o.port, path, payload), eog, save)
    audit(data, eog)
    partial.replace(o.rows)


if __name__ == "__main__":
    main()
