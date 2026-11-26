"""Separate visible text, sampled-ID coverage and native termination. Never fill a missing stream ID."""

import json
from pathlib import Path
import sys
from collections import Counter
from review_natural import review
from natural235 import SLOTS, need


def describe(row, witness, eog):
    ev = row["events"]
    end = row["end_ns"]
    n = row["predicted"]
    ids = witness["ids"]
    need(type(end) is int and end > 0 and type(n) is int and n > 0, "bad extent")
    lastj = 0
    lastt = 0
    seen = set()
    parts = []
    counts = []
    first = None
    last = None
    weighted = 0
    units = 0
    ids_match = True
    for t, j, tid, fragment in ev:
        need(
            type(t) is int
            and lastt <= t <= end
            and t > 0
            and type(j) is int
            and lastj < j <= n
            and type(tid) is int
            and isinstance(fragment, str),
            "bad event",
        )
        lastj = j
        lastt = t
        seen.add(j)
        ids_match &= j <= len(ids) and ids[j - 1] == tid
        parts.append(fragment)
        added = sum(not c.isspace() for c in fragment)
        units += added
        weighted += t * added
        if added:
            if first is None:
                first = t
            last = t
        counts.append((t, fragment))
    text = "".join(parts)
    missing = sorted(set(range(1, n + 1)) - seen)
    witness_match = (
        ids_match
        and n == len(ids)
        and text == witness["content"]
        and row["stop"] == witness["stop"]
    )
    if not witness_match:
        coverage = "witness-disagreement"
    elif not missing:
        coverage = "fully-delivered"
    elif missing == [n] and row["stop"] == "eos" and ids[-1] in eog:
        coverage = "only-final-eog-not-delivered"
    elif any(j < n for j in missing):
        coverage = "interior-ordinal-missing"
    else:
        coverage = "other-missing-ordinal"
    # Retrospective only: the client did not know that later text would stay equal to AMBER.
    target = None
    if row["case"].endswith(":key") and text.strip() == "AMBER":
        prefix = ""
        for t, s in counts:
            prefix += s
            if prefix.strip() == "AMBER":
                target = t
                break
        need(target is not None, "missing label boundary")
    return dict(
        case=row["case"],
        rep=row["rep"],
        native_stop=row["stop"],
        backend_predicted=n,
        delivered_ordinals=len(seen),
        missing_ordinals=missing,
        coverage=coverage,
        delivered_text_matches_witness=text == witness["content"],
        first_id_ns=ev[0][0] if ev else None,
        first_visible_ns=first,
        last_visible_ns=last,
        terminal_ns=end,
        tail_after_last_visible_ns=None if last is None else end - last,
        visible_nonspace_codepoints=units,
        codepoint_arrival_sum_ns=weighted,
        mean_visible_codepoint_arrival_ms=None if units == 0 else weighted / units / 1e6,
        retrospective_exact_label_ns=target,
    )


def build(lock, root, ref):
    full = review(lock, root, ref)
    need(
        (json.dumps(full, indent=2, allow_nan=False) + "\n").encode()
        == (root / "summary.json").read_bytes(),
        "not an accepted complete raw replay",
    )
    eog = json.loads((ref / "runtime.json").read_bytes())["eog_token_ids"]
    blocks = []
    classes = Counter()
    for i, tile in enumerate(SLOTS):
        raw = json.loads((root / f"block-{i}/rows.json").read_bytes())
        streams = []
        for row in raw["streams"]:
            v = describe(row, raw["witnesses"][row["case"]], eog)
            primary = full["comparison"]["block_observations"][i][row["case"]][row["rep"]]
            need(
                (v["coverage"] == "fully-delivered")
                == (primary["all_sampled_ids_observed"] and primary["delivery_matches_witness"])
                and v["terminal_ns"] / 1e6 == primary["terminal_ms"]
                and (None if v["first_visible_ns"] is None else v["first_visible_ns"] / 1e6)
                == primary["first_visible_ms"],
                "drift from primary receipt",
            )
            streams.append(v)
            classes[v["coverage"]] += 1
        blocks.append(dict(block=i, tile=tile, streams=streams))
    return dict(
        schema="sparkfuse.natural235.delivery.v1",
        timed_streams=sum(classes.values()),
        coverage_counts=dict(sorted(classes.items())),
        blocks=blocks,
        scope="Descriptive client reception of Unicode non-whitespace code points. Not graphemes, token decode time, a new serving gate or a safe early-stop rule. Missing sampled IDs remain missing.",
    )


if __name__ == "__main__":
    print(json.dumps(build(*map(Path, sys.argv[1:])), indent=2, allow_nan=False))
