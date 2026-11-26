"""Audit a prefix preparation experiment against its measured single-slot source."""

import argparse
import hashlib
import json
from pathlib import Path

from audit_bundle import ROOT, attest, check
from audit_replay import TAGS, audit as audit_reference
from catalog_prefix import audit as audit_prepared


def source_files(directory):
    # Support the published layout and an independently rerun PANEL=prefix session.
    prefix = "prefix-" if (directory / "prefix-rows.json").is_file() else ""
    return {key: directory / f"{prefix}{key}.json" for key in ("rows", "checkpoint", "runtime")}


def load_inputs(directory, source, *, include_rows):
    fingerprints = {}

    def read(key, path):
        data = path.read_bytes()
        fingerprints[key] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    files = source_files(source)
    old = read("reference_rows", files["rows"])
    audit_reference(old)
    check(
        [(r["length"], r["case"]) for r in old] == [(n, c) for n in (256, 2048) for c in TAGS],
        "reference row order changes seeded arm order",
    )
    lock = ROOT / "qwen72b-q4km.lock.json"
    base = None
    for role, where in (
        ("reference", files),
        ("prepared", {k: directory / f"{k}.json" for k in ("checkpoint", "runtime")}),
    ):
        cp = read(role + "_checkpoint", where["checkpoint"])
        rt = read(role + "_runtime", where["runtime"])
        sig = attest(lock, cp, rt, 81, 1, 128)
        if base is None:
            base = sig
        check(sig == base, "different binaries, driver or EOG mask across runs")
    forbidden = set(base["eog_token_ids"])
    check(
        all(forbidden.isdisjoint(row[a]["ids"]) for row in old for a in ("cold", "reuse")),
        "reference includes suppressed EOG",
    )
    if not include_rows:
        return files["rows"]
    raw = read("prepared_rows", directory / "rows.json")
    panel = audit_prepared(raw, old)
    check(
        all(
            forbidden.isdisjoint(arm[stage]["ids"])
            for row in raw
            for arm in row["arms"].values()
            for stage in ("prime", "request")
        ),
        "prepared run includes suppressed EOG",
    )
    return dict(
        schema=1,
        model=cp["model"],
        revision=cp["revision"],
        format=cp["format"],
        verified_weight_bytes=cp["total_bytes"],
        runtime=base,
        evidence_sha256=fingerprints,
        panel=panel,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("source", "audit"))
    p.add_argument("directory", type=Path)
    p.add_argument("reference", type=Path)
    a = p.parse_args()
    result = load_inputs(a.directory, a.reference, include_rows=(a.mode == "audit"))
    print(result if a.mode == "source" else json.dumps(result, indent=2, allow_nan=False))
