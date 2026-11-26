"""Reconcile the four live timing streams with raw rows; keep the excluded launch separate."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from audit_tile235 import check
from measure235 import audit, require
from tile235 import TILES


def encoded(x):
    return (json.dumps(x, indent=2, allow_nan=False) + "\n").encode()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def observations(folder):
    raw = (folder / "rows.json").read_bytes()
    runtime = (folder / "runtime.json").read_bytes()
    rows = json.loads(raw)
    rt = json.loads(runtime)
    timing = encoded(audit(rows, rt["eog_token_ids"]))
    return rows, rt, timing, {"rows.json": sha(raw), "runtime.json": sha(runtime)}


def reconcile_block(folder, idx):
    rows, rt, offline, hashes = observations(folder)
    live = (folder / "timing.json").read_bytes()
    require(live == offline, "live stdout differs from offline timing bytes")
    require(rt["context"]["n_ubatch"] == TILES[idx], "wrong timing block")
    hashes["timing.json"] = sha(live)
    return rows, hashes


def excluded(folder, inputs):
    rows, rt, clean, hashes = observations(folder)
    mixed = (folder / "stdout.txt").read_bytes()
    expected = b"".join(
        f"case {r['rep']} {r['case']} {r['processed']} {r['predicted']}\n".encode()
        for r in rows["rows"]
    )
    require(
        rows["inputs"] == inputs and rt["context"]["n_ubatch"] == 128, "excluded workload drift"
    )
    require(mixed == expected + clean and mixed != clean, "excluded stdout cause not reproduced")
    return dict(
        reason="progress_on_machine_readable_stdout",
        completed_requests=8,
        included_in_comparison=False,
        inputs_sha256={**hashes, "stdout.txt": sha(mixed)},
    )


def proof(lock, root, ref, omit=None):
    primary = encoded(check(lock, root, ref))
    require((root / "summary.json").read_bytes() == primary, "primary summary did not replay")
    hashes = {}
    panels = []
    for i, u in enumerate(TILES):
        raw, h = reconcile_block(root / f"block-{i}", i)
        hashes[f"block-{i}"] = h
        panels.append(raw)
    keys = list(panels[0]["inputs"])
    case_medians = {}
    for key in keys:
        case_medians[key] = {}
        for ub in sorted(set(TILES)):
            cohort = [
                row
                for panel, tile in zip(panels, TILES, strict=True)
                if tile == ub
                for row in panel["rows"]
                if row["case"] == key
            ]
            require(len(cohort) == 4, "wrong descriptive median cohort")
            case_medians[key][str(ub)] = {
                k: median(row[k] for row in cohort) for k in ("prompt_ms", "decode_ms", "wall_ms")
            }
    return dict(
        schema=1,
        primary_sha256=sha(primary),
        included_requests=32,
        timing_streams=hashes,
        descriptive_case_median_ms=case_medians,
        excluded_attempt=None if omit is None else excluded(omit, panels[0]["inputs"]),
        scope="Absolute times pool four observations per prompt and arm for description only. The decision uses paired order-specific gates in the primary summary.",
    )


if __name__ == "__main__":
    import sys

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("lock", type=Path)
    p.add_argument("root", type=Path)
    p.add_argument("baseline", type=Path)
    p.add_argument(
        "--excluded",
        type=Path,
        help="Optional original rejected stdout and source observations; never counted as a timed launch",
    )
    a = p.parse_args()
    sys.stdout.buffer.write(encoded(proof(a.lock, a.root, a.baseline, a.excluded)))
