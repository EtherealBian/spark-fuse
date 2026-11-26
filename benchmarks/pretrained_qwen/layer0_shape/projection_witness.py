"""Replay a fixed first-layer panel from segment commitments and a minimal BF16 witness.

Identical SHA-256 commitments of the recorded segments imply zero mismatches.
For unequal segments, compare every BF16 word at each sampled position. Never
infer a mismatch count from unequal hashes alone. No raw embedding table needed.
"""

import argparse
import hashlib
import json
import struct
import tempfile
from collections import defaultdict
from pathlib import Path

from fork_witness import materialize

INDEX = "eb767b629ae46189862a3a15f16d16a7981833b37f53726e9d53d57abd1bfa27"
PARENT = "6d46f39acbf2b71387f1ea1ea00191380f74fad7b8d999969dfafa27a014ac6d"
ORIGINAL = "0fd833ad333bfbff19241d04a16f42e0174893c42de1a96f79d0b5682eda0c88"
STAGES = {
    "embedding": 5120,
    "normalized": 5120,
    "cos": 128,
    "sin": 128,
    "key_linear": 1024,
    "value": 1024,
    "key_rotated": 1024,
}
ARMS = ("whole", "split", "padded")
BANK = (("key_rotated", 0), ("value", 1))


def need(ok, why):
    if not ok:
        raise ValueError(why)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def checked_index(path, pin):
    need(path.stat().st_size < 256 * 1024, "oversize index")
    data = path.read_bytes()
    need(sha(data) == pin, "unexpected measured index")
    return json.loads(data)


def panel(doc):
    cells = [(n, f, b) for n in (128, 2048) for f in (1, 2) for b in (0, 1)]
    schedule = [
        (c, t, a)
        for t in (0, 1)
        for c in (cells if t == 0 else cells[::-1])
        for a in (ARMS if t == 0 else ARMS[::-1])
    ]
    need(
        doc["schema"] == "sparkfuse.first-bank-probe.v1" and doc["parent_index_sha256"] == PARENT,
        "wrong experiment",
    )
    pos = 0
    runs = {}
    by_cell = defaultdict(set)
    for row, key in zip(doc["rows"], schedule, strict=True):
        c, t, a = key
        n = c[0]
        p = n * 3 // 4
        need(
            (tuple(row["cell"]), row["repeat"], row["arm"]) == key
            and row["projected_rows"] == {"whole": [n], "split": [p, n - p], "padded": [n, n]}[a],
            "unexpected position/shape",
        )
        need(list(row["stages"]) == list(STAGES), "stage order")
        runs[key] = {}
        for s, width in STAGES.items():
            v = row["stages"][s]
            need(v["offset"] == pos and v["bytes"] == width * 6, "noncontiguous segment")
            pos += width * 6
            runs[key][s] = v["sha256"]
            by_cell[c, s].add(v["sha256"])
    need(pos == 3907584, "wrong logical total")
    return cells, runs, by_cell


def parent_banks(root, parent_index, cells):
    oracle = {}
    payload = {}
    old = {(tuple(r["cell"]), r["repeat"], r["arm"]): r["kv"] for r in parent_index["records"]}
    for c in cells:
        for t in (0, 1):
            for a, pa in [("whole", "whole"), ("split", "inplace")]:
                key = (*c, t, pa)
                b = (root / ("-".join(map(str, key)) + "-kv.bf16")).read_bytes()
                m = old[c, t, pa]
                need(
                    len(b) == m["bytes"] == 786432 and sha(b) == m["sha256"],
                    "bad full-model oracle",
                )
                for s, i in BANK:
                    val = b[i * 6144 : (i + 1) * 6144]
                    h = sha(val)
                    oracle[c, t, a, s] = h
                    payload[h] = val
    return oracle, payload


def explicit(by_cell, known):
    out = {}
    for (_, stage), hashes in by_cell.items():
        if len(hashes) == 1 and stage not in dict(BANK):
            continue
        for h in hashes - known.keys():
            # A disclosure rule, not a fallback: a changed panel requires a new review.
            need(stage == "key_linear", "explicit samples would disclose an unreviewed stage")
            out[h] = 6 * STAGES[stage]
    need(len(out) == 8 and sum(out.values()) == 49152, "unexpected compact payload")
    return dict(sorted(out.items()))


def pack(raw, doc, needed, dst):
    need(
        raw.stat().st_size == 3907584 and not dst.exists(), "invalid source or existing destination"
    )
    b = raw.read_bytes()
    need(sha(b) == ORIGINAL, "original sample digest changed")
    blobs = {}
    for r in doc["rows"]:
        for m in r["stages"].values():
            seg = b[m["offset"] : m["offset"] + m["bytes"]]
            h = m["sha256"]
            need(sha(seg) == h, "bad original segment")
            if h in needed:
                blobs[h] = seg
    need(blobs.keys() == needed.keys(), "missing explicit segment")
    with dst.open("xb") as f:
        for h in needed:
            f.write(blobs[h])


def load_explicit(file, needed):
    need(file.stat().st_size == sum(needed.values()), "wrong compact length")
    out = {}
    with file.open("rb") as fp:
        for h, n in needed.items():
            data = fp.read(n)
            need(len(data) == n and sha(data) == h, "bad explicit segment")
            out[h] = data
        need(not fp.read(1), "trailing bytes")
    return out


def counts(h1, h2, width, data):
    if h1 == h2:
        return [0, 0, 0]
    a, b = data[h1], data[h2]
    need(len(a) == len(b) == 6 * width, "missing unequal segment")
    out = []
    for pos in range(3):
        lo, hi = 2 * width * pos, 2 * width * (pos + 1)
        out.append(
            sum(
                x != y
                for x, y in zip(
                    struct.iter_unpack("<H", a[lo:hi]),
                    struct.iter_unpack("<H", b[lo:hi]),
                    strict=True,
                )
            )
        )
    return out


def execute(report, parent, archive, compact, raw=None):
    doc = checked_index(report / "index.json", INDEX)
    pdoc = checked_index(parent / "index.json", PARENT)
    cells, runs, by_cell = panel(doc)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "full-model"
        materialize(parent, archive, p)
        oracle, data = parent_banks(p, pdoc, cells)
    needed = explicit(by_cell, data)
    if raw is not None:
        pack(raw, doc, needed, compact)
    data.update(load_explicit(compact, needed))
    observations = []
    oracle_ok = True
    stable = True
    for c in cells:
        rr = []
        same = all(runs[c, 0, a] == runs[c, 1, a] for a in ARMS)
        stable &= same
        for t in (0, 1):
            w = runs[c, t, "whole"]
            d = {}
            for a in ("whole", "split"):
                d[a] = {
                    s: counts(runs[c, t, a][s], oracle[c, t, a, s], STAGES[s], data)
                    for s, _ in BANK
                }
                oracle_ok &= not any(any(v) for v in d[a].values())

            def compare(a):
                return {
                    stage: counts(w[stage], runs[c, t, a][stage], width, data)
                    for stage, width in STAGES.items()
                }

            rr.append(
                dict(
                    full_model_bank_mismatches=d,
                    whole_vs_split=compare("split"),
                    whole_vs_padded=compare("padded"),
                )
            )
        observations.append(dict(cell=list(c), cross_order_bitwise=same, rounds=rr))
    return dict(
        schema="sparkfuse.first-bank.compact-replay.v1",
        index_sha256=INDEX,
        parent_index_sha256=PARENT,
        compact_sha256=sha(compact.read_bytes()),
        explicit_bytes=compact.stat().st_size,
        explicit_key_projection_segments=len(needed),
        same_commitment_cell_stages=sum(len(v) == 1 for v in by_cell.values()),
        rows=len(runs),
        full_model_first_bank_reproduced=oracle_ok,
        repeat_order_stable=stable,
        cells=observations,
        scope="Exact sampled mismatch counts: equal index commitments, independently reconstructed full-model K/V, and explicit raw BF16 for unequal pre-rotary keys. This compact proof neither reconstructs every original activation nor rehashes the monolithic original sample file on replay.",
    )


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("report", type=Path)
    a.add_argument("parent", type=Path)
    a.add_argument("archive", type=Path)
    a.add_argument("compact", type=Path)
    a.add_argument("--pack-original", type=Path)
    z = a.parse_args()
    print(
        json.dumps(
            execute(z.report, z.parent, z.archive, z.compact, z.pack_original),
            indent=2,
            allow_nan=False,
        )
    )
