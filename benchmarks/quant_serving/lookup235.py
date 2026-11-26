"""Test no-draft-model speculative decoding on copy, edited text, and new prose. Native greedy responses."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import time
from urllib.request import urlopen
from prefix_replay import digest, post

ARMS = ("none", "ngram-simple", "ngram-simple", "none")
GEN = 64
V = 151936
CORPUS = {
    "copy:0": (
        "Repeat this paragraph word for word, with nothing before or after it.",
        "At sunrise the inspector opened the green cabinet and counted the brass keys. The lower drawer contained a folded list of labels, sorted by building and room. The last entry named the workshop beside the loading dock.",
    ),
    "copy:1": (
        "Copy these three lines exactly, including punctuation. Do not add a heading.",
        "The first train stops at the stone bridge before turning west.\nThe second train crosses the river and waits beside the warehouse.\nThe third train leaves the valley at noon and arrives before dusk.",
    ),
    "edit:0": (
        "Rewrite the following paragraph exactly, except replace every instance of copper with silver. No commentary.",
        "The copper plate rests on the top shelf of the storage cabinet. An inspector checks the copper plate before every afternoon shift and signs the sheet on the left. After the inspection, the copper plate returns to its original position.",
    ),
    "edit:1": (
        "Return the following paragraph with the single word amber changed to violet wherever it occurs. No commentary.",
        "The archive sends an amber notice when the second copy has arrived. The coordinator places the amber notice above the date on the delivery form. At closing time the clerk files the amber notice with the rest of the records.",
    ),
    "new:0": (
        "Answer in two or three plain English sentences without a preamble.",
        "Why can a shorter queue improve response time even if the workers operate at exactly the same speed?",
    ),
    "new:1": (
        "Answer in two or three plain English sentences without a preamble.",
        "How could a museum decide whether two unlabeled photographs show the same room at different times?",
    ),
}
ORDER = ("copy:0", "new:0", "edit:0", "new:1", "copy:1", "edit:1")


def need(x, msg):
    if not x:
        raise ValueError(msg)


def rendered(fn, key):
    system, user = CORPUS[key]
    text = fn(
        "apply-template",
        dict(messages=[dict(role="system", content=system), dict(role="user", content=user)]),
    )["prompt"]
    need(text.count(system) == text.count(user) == 1, "chat template missing content")
    ids = fn("tokenize", dict(content=text, add_special=False, parse_special=True))["tokens"]
    return dict(rendered=text, ids=ids, sha256=digest(ids))


def issue(fn, ids, eog, limit=GEN):
    t0 = time.perf_counter_ns()
    r = fn(
        "completion",
        dict(
            prompt=ids,
            n_predict=limit,
            cache_prompt=False,
            id_slot=0,
            temperature=0,
            seed=1513,
            stream=False,
            return_tokens=True,
        ),
    )
    wall = (time.perf_counter_ns() - t0) / 1e6
    t = r["timings"]
    have = "draft_n" in t
    need(have == ("draft_n_accepted" in t), "partial draft counters")
    tokens = r["tokens"]
    text = fn("detokenize", dict(tokens=[i for i in tokens if i not in eog]))["content"]
    return dict(
        ids=tokens,
        text=r["content"],
        detokenized=text,
        wall_ms=wall,
        prompt_ms=t["prompt_ms"],
        decode_ms=t["predicted_ms"],
        slot=r["id_slot"],
        evaluated=r["tokens_evaluated"],
        processed=t["prompt_n"],
        cached=t["cache_n"],
        predicted=t["predicted_n"],
        truncated=r["truncated"],
        stop=r["stop_type"],
        counters_present=have,
        drafted=t.get("draft_n", 0),
        accepted=t.get("draft_n_accepted", 0),
    )


def check_row(prompt, row, eog, mode):
    ids = prompt["ids"]
    r = row
    out = r["ids"]
    ints = ("slot", "evaluated", "processed", "cached", "predicted", "drafted", "accepted")
    need(
        all(type(r[k]) is int for k in ints)
        and type(r["counters_present"]) is bool
        and type(out) is list
        and all(type(t) is int and 0 <= t < V for t in out),
        "invalid integer/vector/counter",
    )
    need(
        type(ids) is list
        and 0 < len(ids) < 4096 - GEN
        and all(type(t) is int and 0 <= t < V for t in ids)
        and prompt["sha256"] == digest(ids),
        "prompt IDs",
    )
    need(
        r["slot"] == r["cached"] == 0
        and r["truncated"] is False
        and r["evaluated"] == r["processed"] == len(ids),
        "not a cold, intact prompt",
    )
    need(
        1 <= len(out) <= r["predicted"] <= GEN
        and (r["predicted"] == len(out) if mode == "none" else r["predicted"] - len(out) <= 8),
        "reported vs delivered tokens",
    )
    need(
        r["stop"] in ("eos", "limit")
        and ((out[-1] in eog) == (r["stop"] == "eos"))
        and not any(t in eog for t in out[:-1])
        and (r["stop"] != "limit" or r["predicted"] == GEN),
        "stop does not reconcile",
    )
    need(
        isinstance(r["text"], str) and r["text"] == r["detokenized"],
        "text and independent detokenization differ",
    )
    need(
        0 <= r["accepted"] <= min(r["drafted"], r["predicted"])
        and (r["drafted"] > 0) == r["counters_present"]
        and (mode != "none" or r["drafted"] == 0),
        "invalid draft metrics",
    )
    need(
        all(
            type(r[k]) in (float, int) and math.isfinite(r[k]) and r[k] >= 0
            for k in ("wall_ms", "decode_ms", "prompt_ms")
        )
        and r["wall_ms"] > 0
        and r["wall_ms"] + 3 >= r["prompt_ms"] + r["decode_ms"],
        "invalid timing",
    )


def validate_block(block, eog):
    i = block["block"]
    need(
        type(i) is int
        and 0 <= i < 4
        and block["mode"] == ARMS[i]
        and type(eog) is list
        and eog
        and all(type(v) is int and 0 <= v < V for v in eog),
        "invalid block or EOG",
    )
    keys = list(ORDER) if i < 2 else list(reversed(ORDER))
    need(
        [r["key"] for r in block["rows"]] == keys and set(block["inputs"]) == set(CORPUS),
        "missing/unbalanced cases",
    )
    seen = set()
    for rec in block["rows"]:
        k = rec["key"]
        p = block["inputs"][k]
        need(
            all(p["rendered"].count(v) == 1 for v in CORPUS[k]) and p["sha256"] not in seen,
            "unexpected or duplicate input",
        )
        seen.add(p["sha256"])
        check_row(p, rec["response"], eog, block["mode"])


def assess(blocks, eog):
    need(
        len(blocks) == 4 and [b["block"] for b in blocks] == list(range(4)),
        "four independent ABBA blocks required",
    )
    [validate_block(b, eog) for b in blocks]
    need(all(b["inputs"] == blocks[0]["inputs"] for b in blocks), "inputs differ between arms")
    cases = {}
    rows = [{r["key"]: r["response"] for r in b["rows"]} for b in blocks]
    for key in ORDER:
        rs = [r[key] for r in rows]
        paired = []
        identical = lambda x, y: (x["ids"], x["stop"], x["text"]) == (
            y["ids"],
            y["stop"],
            y["text"],
        )
        for a, b in [(0, 1), (3, 2)]:
            x, y = rs[a], rs[b]
            paired.append(
                dict(
                    saved_wall_fraction=1 - y["wall_ms"] / x["wall_ms"],
                    baseline_wall_ms=x["wall_ms"],
                    lookup_wall_ms=y["wall_ms"],
                    identical=identical(x, y),
                    baseline_delivered=len(x["ids"]),
                    lookup_delivered=len(y["ids"]),
                    lookup_predicted=y["predicted"],
                    drafted=y["drafted"],
                    accepted=y["accepted"],
                )
            )
        repeat = identical(rs[0], rs[3]) and identical(rs[1], rs[2])
        eligible = repeat and all(
            p["identical"]
            and p["lookup_delivered"] == p["lookup_predicted"]
            and p["accepted"] > 0
            and p["saved_wall_fraction"] >= 0.05
            for p in paired
        )
        cases[key] = dict(pairs=paired, repeatable_within_both_arms=repeat, passes=eligible)
    cohorts = {}
    for kind in ("copy", "edit", "new"):
        ks = [v for k, v in cases.items() if k.startswith(kind + ":")]
        cohorts[kind] = dict(
            cases_meeting_gate=sum(v["passes"] for v in ks),
            observed_cases=len(ks),
            both_cases_pass=all(v["passes"] for v in ks),
            order_median_saved_fraction=[
                median(v["pairs"][j]["saved_wall_fraction"] for v in ks) for j in range(2)
            ],
        )
    return dict(
        cases=cases,
        workloads=cohorts,
        all_six_pass=all(v["passes"] for v in cases.values()),
        scope="Two prompts per synthetic family, four separate native server launches, one warmup per launch. Cold prompts, natural EOS or 64 tokens. The comparison requires exact delivered IDs, stop/text, positive accepted draft count and >=5% lower wall time in both orders; no population-level or quality guarantee.",
    )


def prove_mode(pid, server, port, block):
    p = Path("/proc") / str(pid)
    need((p / "exe").samefile(server), "unexpected running binary")
    args = (p / "cmdline").read_bytes().decode().split("\0")
    mode = ARMS[block]
    opts = {
        "--spec-type": mode,
        "--draft-max": "8",
        "--draft-min": "1",
        "--spec-ngram-size-n": "4",
        "--spec-ngram-size-m": "8",
    }
    for k, v in opts.items():
        need(args.count(k) == 1 and args[args.index(k) + 1] == v, "wrong process settings")
    need(
        not any(
            a in args
            for a in (
                "-md",
                "--model-draft",
                "--lookup-cache-static",
                "--lookup-cache-dynamic",
                "-lcs",
                "-lcd",
            )
        ),
        "external draft not allowed",
    )
    slot = json.load(urlopen(f"http://127.0.0.1:{port}/slots", timeout=5))
    need(
        len(slot) == 1
        and slot[0]["id"] == 0
        and slot[0]["n_ctx"] == 4096
        and slot[0]["is_processing"] is False
        and slot[0]["speculative"] is (mode != "none"),
        "slot speculation not initialized as requested",
    )
    sid = hashlib.sha256(
        (
            Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            + ":"
            + str(pid)
            + ":"
            + (p / "stat").read_text().rsplit(")", 1)[1].split()[19]
        ).encode()
    ).hexdigest()
    return dict(
        block=block,
        mode=mode,
        slot_capable=slot[0]["speculative"],
        ngram_key=4,
        ngram_value=8,
        draft_min=1,
        draft_max=8,
        process_fingerprint=sid,
    )


def execute(args):
    rt = json.loads((args.out / "runtime.json").read_bytes())
    eog = rt["eog_token_ids"]
    fn = lambda k, d: post(args.port, k, d)
    g = prove_mode(args.pid, args.server, args.port, args.block)
    out = args.out
    mode = ARMS[args.block]
    (out / "lookup-setting.json").write_text(json.dumps(g, indent=2) + "\n")
    prompt = fn(
        "tokenize", dict(content="Please name one color.", add_special=False, parse_special=False)
    )["tokens"]
    issue(fn, prompt, eog, 3)
    ins = {k: rendered(fn, k) for k in ORDER}
    x = dict(block=args.block, mode=mode, inputs=ins, rows=[])
    path = out / "lookup-rows.json"
    tmp = out / "lookup.partial"
    need(not any(p.exists() for p in (path, tmp)), "refuse to replace rows")
    for k in ORDER if args.block < 2 else reversed(ORDER):
        row = issue(fn, ins[k]["ids"], eog)
        x["rows"].append(dict(key=k, response=row))
        t = tmp.with_suffix(".write")
        t.write_text(json.dumps(x, separators=(",", ":"), allow_nan=False) + "\n")
        t.replace(tmp)
        check_row(ins[k], row, eog, mode)
        print(k, len(row["ids"]), row["drafted"], row["accepted"], flush=True)
    validate_block(x, eog)
    tmp.replace(path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pid", required=True, type=int)
    p.add_argument("--server", type=Path, required=True)
    p.add_argument("--port", required=True, type=int)
    p.add_argument("--block", required=True, type=int, choices=range(4))
    p.add_argument("--out", required=True, type=Path)
    execute(p.parse_args())
