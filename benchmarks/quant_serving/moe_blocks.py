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


def audit(blocks):
    require(len(blocks) == 4, "expected four independent server blocks")
    group = {}
    for i, b in enumerate(blocks):
        require(
            b["block"] == i and b["ubatch"] == BLOCKS[i] and len(b["rows"]) == 4,
            "invalid block design",
        )
        for j, r in enumerate(b["rows"]):
            key = (LENGTHS[j // 2], j % 2)
            size, case = key
            require(
                (r["size"], r["case"]) == key and r["processed"] == size and r["reused"] == 0,
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
            cases=2, all_four_blocks_identical=exact, within_setting_repeatable=stable
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
