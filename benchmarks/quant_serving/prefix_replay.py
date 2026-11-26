"""Matched single-slot prefix reuse benchmark for the llama.cpp native completion API."""

import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

LENGTHS = (256, 2048)
NGEN = 24
CASES = (
    ("antimony", "What is the purpose of a write-ahead log?"),
    ("indigo", "Why do we use a checksum when storing records?"),
    ("cobalt", "When might it help to index a timestamp column?"),
    ("ochre", "How can a service detect duplicate messages?"),
)


def digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def post(port, path, data):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/{path}",
        json.dumps(data).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.load(r)


def complete(port, ids, reuse, n, *, slot=0):
    # Explicitly mask the EOG set printed by this checkpoint, identically for all arms.
    # A request-only ignore_eos flag did not keep the pinned server at fixed length.
    eog = json.loads(os.environ["SPARKFUSE_EOG_IDS"])
    if (
        not isinstance(eog, list)
        or not eog
        or any(type(t) is not int or t < 0 for t in eog)
        or sorted(set(eog)) != eog
    ):
        raise ValueError("invalid checkpoint EOG list")
    started = time.perf_counter()
    response = post(
        port,
        "completion",
        dict(
            prompt=ids,
            n_predict=n,
            cache_prompt=reuse,
            id_slot=slot,
            temperature=0,
            seed=1513,
            ignore_eos=True,
            logit_bias=[[t, False] for t in eog],
            return_tokens=True,
            stream=False,
        ),
    )
    ms = (time.perf_counter() - started) * 1000
    if any(t in eog for t in response["tokens"]):
        raise ValueError("backend returned an explicitly suppressed EOG token")
    t = response["timings"]
    return dict(
        ids=response["tokens"],
        stop=response["stop_type"],
        processed=t["prompt_n"],
        reused=t["cache_n"],
        predicted=t["predicted_n"],
        prompt_ms=t["prompt_ms"],
        decode_ms=t["predicted_ms"],
        wall_ms=ms,
    )


def common(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def run(port):
    def tokenize(s):
        return post(port, "tokenize", dict(content=s, add_special=False, parse_special=False))[
            "tokens"
        ]

    rows = []
    for li, length in enumerate(LENGTHS):
        for i, (tag, question) in enumerate(CASES):
            text = f"Archive {tag}: Each transaction has an owner, a timestamp and a status. "
            body = tokenize(text * 240)
            query = tokenize("\nQuestion: " + question + " Answer concisely:")
            seed = tokenize("\nMemo: record the previous revision and exit.")
            assert len(query) < length < len(body) and len(seed) >= 2
            trunk = body[: length - len(query)]
            target, other = trunk + query, trunk + seed
            prefix = common(target, other)
            order = ("cold", "reuse") if (li + i) % 2 == 0 else ("reuse", "cold")
            row = dict(
                length=length,
                case=tag,
                target_ids=target,
                primer_ids=other,
                target=digest(target),
                primer=digest(other),
                primer_length=len(other),
                shared=prefix,
                order=list(order),
            )
            for arm in order:
                # Both arms see the same preceding request in the same slot.
                complete(port, other, False, 1)
                row[arm] = complete(port, target, arm == "reuse", NGEN)
            rows.append(row)
            print(
                "measured",
                length,
                tag,
                row["cold"]["processed"],
                row["reuse"]["processed"],
                row["cold"]["ids"] == row["reuse"]["ids"],
                flush=True,
            )
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rows = run(a.port)
    a.output.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
