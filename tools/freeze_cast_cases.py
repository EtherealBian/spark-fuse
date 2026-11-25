"""Freeze retained BF16 decomp failures and width-matched controls before the audit.

A compact catalog, not a rewrite of the append-only search. Preserve the original
worker JSON bytes by hash and its complete comparison; skip a missing worker or a
discovery error explicitly rather than guessing historical backend ownership.
"""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path


def freeze(folder, limit):
    lines = (folder / "index.jsonl").read_bytes().splitlines(keepends=True)[:limit]
    if len(lines) != limit or limit <= 0 or not lines[-1].endswith(b"\n"):
        raise ValueError("incomplete index prefix")
    errors, controls, excluded = [], [], []
    for number, line in enumerate(lines, 1):
        entry = json.loads(line)
        key = entry["id"]
        if not (key.startswith("decomp-") and key.endswith("-bfloat16")):
            continue
        bad = entry["status"] == "error"
        if not bad and not (entry["phase"] == "broad-search" and entry["status"] == "ok"):
            continue
        path = folder / "evidence" / (key + (".rejected.json" if bad else ".json"))
        if not path.is_file():
            if not bad:
                raise ValueError("missing control worker: " + key)
            excluded.append(dict(id=key, line=number, reason="no retained worker"))
            continue
        raw = path.read_bytes()
        w = json.loads(raw)
        c = w.get("comparison")
        if c is None:
            if not bad:
                raise ValueError("incomplete control")
            excluded.append(
                dict(id=key, line=number, reason=w["status"], worker_sha256=sha256(raw).hexdigest())
            )
            continue
        if w["job"]["id"] != key or w["tuning"]["selected"] != c["schedule"]:
            raise ValueError("worker identity/schedule mismatch")
        backend = {n: z["parity"] for n, z in c["backends"].items()}
        if bad and backend != {"eager": "pass", "sparkfuse": "pass", "inductor": "fail"}:
            excluded.append(
                dict(id=key, line=number, reason="other comparator outcome", parity=backend)
            )
            continue
        if not bad and set(backend.values()) != {"pass"}:
            raise ValueError("inconsistent successful worker")
        v = dict(
            id=key,
            line=number,
            cohort="retained_rejection" if bad else "control",
            job=w["job"],
            worker_sha256=sha256(raw).hexdigest(),
            original_source=w["provenance"],
            comparison=c,
        )
        (errors if bad else controls).append(v)
    widths = {z["job"]["width"] for z in errors}
    selected = []
    for z in controls:
        width = z["job"]["width"]
        if z["job"]["rows"] >= 16 and width in widths:
            selected.append(z)
            widths.remove(width)
        if len(selected) == 3:
            break
    return dict(
        index_lines=limit,
        index_prefix_sha256=sha256(b"".join(lines)).hexdigest(),
        selection="All retained decomp BF16 exclusively-Inductor comparison rejections; then first three successful broad-search workers with >=16 rows and distinct rejection-matched widths in index order.",
        excluded=excluded,
        cases=errors + selected,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("--lines", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = freeze(a.folder, a.lines)
    a.output.write_bytes(
        gzip.compress(
            (json.dumps(result, indent=1, sort_keys=True, allow_nan=False) + "\n").encode(), mtime=0
        )
    )
