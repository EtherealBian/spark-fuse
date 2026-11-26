"""Build a new compile_tuned SQLite catalog from a *trusted*, portable campaign bundle.

No PyTorch import, GPU, model deserialization or kernel compilation. Verifies the original
measurement records and every referenced source archive before publishing the new file.
"""

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile

if __package__:
    from .campaign_snapshot import encoded, verify_doc
else:
    from campaign_snapshot import encoded, verify_doc

HEX = re.compile("[0-9a-f]{64}")
LAYOUTS = {
    "block",
    "warp",
    "flat",
    "warp_xor",
    "warp_slots",
    "warp_bf16_guard",
    "block_welford",
    "compact_mean",
    "mean_tree",
    "mean_cohort",
    "mean_packed_tile",
}


def legal(s):
    return (
        type(s) is dict
        and set(s) == {"threads", "vec", "cache", "layout"}
        and type(s["threads"]) is int
        and s["threads"] in (32, 64, 128, 256, 512, 1024)
        and type(s["vec"]) is int
        and s["vec"] in (1, 2, 4, 8)
        and type(s["cache"]) is bool
        and isinstance(s["layout"], str)
        and s["layout"] in LAYOUTS
    )


def classify(records, compiler):
    if not isinstance(compiler, str) or not HEX.fullmatch(compiler):
        raise ValueError("expected full compiler digest")
    reasons = Counter()
    group = defaultdict(list)
    blocked = set()
    for e in records:
        idx = e["index"]
        w = e["worker"] or {}
        q = w.get("tuning") or {}
        key = q.get("key")
        source = w.get("provenance") or {}
        sha = source.get("source_tree_sha256")
        name = idx["id"]
        backends = w.get("comparison", {}).get("backends", {})
        if q.get("compiler") != compiler:
            reasons["other_or_missing_compiler"] += 1
            continue
        if not isinstance(key, str) or not HEX.fullmatch(key):
            raise ValueError("invalid graph/environment signature: " + name)
        reason = None
        if idx["status"] != "ok" or w.get("status") != "ok":
            reason = "failed_or_incomplete"
        elif (
            not isinstance(sha, str)
            or not HEX.fullmatch(sha)
            or source.get("archive") != f"sources/{sha}.tar.gz"
        ):
            reason = "no_archived_source"
        elif set(backends) != {"eager", "inductor", "sparkfuse"} or any(
            t.get("parity") != "pass" for t in backends.values()
        ):
            reason = "no_explicit_three_way_parity"
        elif not q.get("admitted"):
            reason = "default_retained"
        elif not legal(q["selected"]):
            raise ValueError("unsupported schedule descriptor: " + name)
        if reason:
            reasons[reason] += 1
            blocked.add(key)
            continue
        group[key].append(dict(case=name, selected=q["selected"], source=sha))
    rows = {}
    for key, items in sorted(group.items()):
        if key in blocked or len({encoded(i["selected"]) for i in items}) != 1:
            reasons["ambiguous_or_revoked"] += len(items)
            continue
        rows[key] = dict(
            key=key,
            compiler=compiler,
            selected=items[0]["selected"],
            evidence=sorted(
                ({"id": i["case"], "source": i["source"]} for i in items), key=lambda q: q["id"]
            ),
        )
    reasons["imported_records"] = sum(len(v["evidence"]) for v in rows.values())
    assert sum(reasons.values()) == len(records)
    return rows, dict(sorted(reasons.items()))


def import_bundle(bundle, dest, compiler):
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("refusing to replace a catalog")
    with gzip.open(bundle, "rt", encoding="utf8") as f:
        doc = json.load(f)
    verified = verify_doc(doc, bundle.parent / "sources")
    rows, reasons = classify(doc["records"], compiler)
    if not rows:
        raise ValueError("no independently qualified, admitted schedules for this exact compiler")
    info = dict(
        format="sparkfuse.measured-catalog.v1",
        compiler=compiler,
        records_sha256=doc["records_sha256"],
        bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
        unique_keys=len(rows),
        counts=reasons,
        verified=verified,
    )
    with tempfile.TemporaryDirectory(dir=dest.parent) as folder:
        temp = Path(folder) / "catalog.sqlite"
        with closing(sqlite3.connect(temp)) as db, db:
            db.execute("CREATE TABLE schedules(key TEXT PRIMARY KEY,record TEXT NOT NULL)")
            db.executemany(
                "INSERT INTO schedules VALUES(?,?)",
                [(key, encoded(value).decode()) for key, value in rows.items()],
            )
            db.execute("CREATE TABLE provenance(info TEXT NOT NULL)")
            db.execute("INSERT INTO provenance VALUES(?)", (encoded(info).decode(),))
            assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        # Same-filesystem atomic create, including a racing writer; never replace a user's DB.
        os.link(temp, dest)
    return info


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--compiler",
        required=True,
        help="exact backend_id() SHA-256; never match on a GPU model name alone",
    )
    a = p.parse_args()
    print(json.dumps(import_bundle(a.bundle, a.out, a.compiler), indent=2))
