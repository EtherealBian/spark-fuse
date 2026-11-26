"""Matched prefill microbatch experiment for an MXFP4 MoE; warm, cache-disabled requests."""

import argparse
import json
import math
from pathlib import Path
from statistics import median
from prefix_replay import complete, digest, post

BLOCKS = (128, 512, 512, 128)
QUESTIONS = (
    "How do checksums help detect an incomplete write?",
    "What does a retry budget prevent?",
)
LENGTHS = (256, 2048)
NGEN = 24


def measure(port, block):
    def tok(s):
        return post(port, "tokenize", dict(content=s, add_special=False, parse_special=False))[
            "tokens"
        ]

    complete(port, tok("A short warmup for this independent process."), False, 3)
    rows = []
    for size in LENGTHS:
        for i, q in enumerate(QUESTIONS):
            start = tok(
                (
                    "The storage journal records commit time and owner. "
                    if i == 0
                    else "The gateway sets a limit on retries for each operation. "
                )
                * 300
            )
            end = tok("\nQuestion: " + q + " Answer:")
            assert len(start) > size > len(end)
            ids = start[: size - len(end)] + end
            w = complete(port, ids, False, NGEN)
            rows.append(dict(size=size, case=i, input_ids=ids, input_hash=digest(ids), **w))
    return dict(block=block, ubatch=BLOCKS[block], rows=rows)


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def paired_changes(records):
    """Keep prompt identity and time direction; positive differences favor ubatch 512."""
    assert len(records) == 4
    pairs = []
    for which, a, b in (("A_then_B", 0, 1), ("B_then_A", 3, 2)):
        left, right = records[a], records[b]
        first = next(
            (
                i
                for i, (x, y) in enumerate(zip(left["ids"], right["ids"], strict=True), 1)
                if x != y
            ),
            None,
        )
        pairs.append(
            dict(
                order=which,
                first_divergent_token=first,
                saved_ms={k: left[k] - right[k] for k in ("prompt_ms", "decode_ms", "wall_ms")},
            )
        )
    return dict(
        input_sha256=records[0]["input_hash"],
        comparisons=pairs,
        wall_faster_in_both_orders=all(v["saved_ms"]["wall_ms"] > 0 for v in pairs),
        same_tokens_in_both_orders=all(v["first_divergent_token"] is None for v in pairs),
    )


def audit(blocks):
    require(len(blocks) == 4, "expected four independent server blocks")
    group = {}
    for i, b in enumerate(blocks):
        require(
            type(b["block"]) is int
            and b["block"] == i
            and type(b["ubatch"]) is int
            and b["ubatch"] == BLOCKS[i]
            and len(b["rows"]) == 4,
            "invalid block design",
        )
        for j, r in enumerate(b["rows"]):
            key = (LENGTHS[j // 2], j % 2)
            size, case = key
            require(
                all(type(r[k]) is int for k in ("size", "case", "reused", "processed", "predicted"))
                and (r["size"], r["case"]) == key
                and r["processed"] == size
                and r["reused"] == 0,
                "prompt accounting",
            )
            require(
                isinstance(r["input_ids"], list)
                and len(r["input_ids"]) == size
                and digest(r["input_ids"]) == r["input_hash"]
                and all(type(t) is int and 0 <= t < 201088 for t in r["input_ids"]),
                "input receipt",
            )
            require(
                r["predicted"] == NGEN
                and len(r["ids"]) == NGEN
                and r["stop"] == "limit"
                and all(type(t) is int and 0 <= t < 201088 for t in r["ids"]),
                "output receipt",
            )
            require(
                all(
                    type(r[k]) in (int, float) and math.isfinite(r[k]) and r[k] > 0
                    for k in ("prompt_ms", "decode_ms", "wall_ms")
                )
                and r["wall_ms"] + 3 >= r["prompt_ms"] + r["decode_ms"],
                "invalid timing",
            )
            group.setdefault(key, []).append(r)
    require(
        len({rs[0]["input_hash"] for rs in group.values()}) == 4, "expected four distinct prompts"
    )
    summary = {}
    for size in LENGTHS:
        exact = 0
        stable = 0
        for case in range(2):
            rs = group[(size, case)]
            require(len({r["input_hash"] for r in rs}) == 1, "unmatched inputs")
            exact += int(all(r["ids"] == rs[0]["ids"] for r in rs))
            stable += int(rs[0]["ids"] == rs[3]["ids"] and rs[1]["ids"] == rs[2]["ids"])
        require(
            group[(size, 0)][0]["input_hash"] != group[(size, 1)][0]["input_hash"],
            "distinct prompts required",
        )
        summary[str(size)] = dict(
            cases=2,
            all_four_blocks_identical=exact,
            within_setting_repeatable=stable,
            paired_order_changes=[paired_changes(group[(size, case)]) for case in range(2)],
        )
        for ub in (128, 512):
            records = [r for c in (0, 1) for j, r in enumerate(group[(size, c)]) if BLOCKS[j] == ub]
            summary[str(size)][str(ub)] = {
                k: median(r[k] for r in records) for k in ("prompt_ms", "decode_ms", "wall_ms")
            }
    return dict(
        schema=1,
        order=list(BLOCKS),
        generated_tokens=NGEN,
        cases=summary,
        all_tokens_exact=all(r["all_four_blocks_identical"] == 2 for r in summary.values()),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    m = p.add_subparsers(dest="mode", required=True)
    a = m.add_parser("measure")
    a.add_argument("--block", type=int, choices=range(4), required=True)
    a.add_argument("--port", type=int, required=True)
    a.add_argument("--output", type=Path, required=True)
    a = m.add_parser("audit")
    a.add_argument("dir", type=Path)
    a = p.parse_args()
    if a.mode == "measure":
        a.output.write_text(
            json.dumps(measure(a.port, a.block), separators=(",", ":"), allow_nan=False) + "\n"
        )
    else:
        print(
            json.dumps(
                audit([json.loads((a.dir / f"block-{i}.json").read_text()) for i in range(4)]),
                indent=2,
                allow_nan=False,
            )
        )
