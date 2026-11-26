"""Bind natural-termination/first-visible-text measurements to the original verified 235B checkpoint."""

import hashlib, json, re, sys
from pathlib import Path
from audit235 import rebuild, verify_inventory, admission
from audit_bundle import attest
from physical_guard import accepted
from natural235 import SLOTS, compare, audit, need, verify_inputs, ORACLE_SHA256


def read(p):
    return json.loads(p.read_bytes())


def review(lock, root, ref):
    base = rebuild(lock, ref)
    need(read(ref / "summary.json") == base, "public baseline failed")
    blocks = []
    seen = set()
    digests = {}
    guards = {}

    def load(d, name):
        path = d / name
        raw = path.read_bytes()
        digests[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    for i, tile in enumerate(SLOTS):
        d = root / f"block-{i}"
        bud = load(d, "header-budget.json")
        verify_inventory(lock, bud)
        rt = load(d, "runtime.json")
        cp = load(d, "checkpoint.json")
        sig = attest(lock, cp, rt, 95, 1, tile)
        need(
            sig == base["runtime"]
            and cp == read(ref / "checkpoint.json")
            and bud == read(ref / "header-budget.json"),
            "wrong weights, device, or backend",
        )
        tr = load(d, "admission-transfer.json")
        admission(tr, bud, "transfer")
        need(tr == read(ref / "admission-transfer.json"), "wrong transfer")
        admission(load(d, "admission.json"), bud, "launch")
        g = load(d, "memory-guard.json")
        accepted(g)
        guards[str(i)] = g["phases"]
        tag = load(d, "tile-setting.json")
        fp = tag["process_fingerprint"]
        need(
            type(tag["block"]) is int
            and (tag["block"], tag["tile"]) == (i, tile)
            and isinstance(fp, str)
            and re.fullmatch("[0-9a-f]{64}", fp)
            and fp not in seen,
            "reused process or wrong arm",
        )
        seen.add(fp)
        raw = load(d, "rows.json")
        verify_inputs(raw["inputs"], sig["eog_token_ids"])
        load(d, "timing.json")
        expect = json.dumps(audit(raw, sig["eog_token_ids"]), indent=2, allow_nan=False) + "\n"
        need(
            (d / "timing.json").read_bytes() == expect.encode(), "block timing does not byte-replay"
        )
        blocks.append(raw)
    return dict(
        schema="sparkfuse.natural235.v1",
        frozen_input_contract_sha256=ORACLE_SHA256,
        tile_sequence=list(SLOTS),
        measured_stream_requests=32,
        independent_nonstream_witnesses=16,
        excluded_warmups=4,
        reference_summary_sha256=hashlib.sha256((ref / "summary.json").read_bytes()).hexdigest(),
        evidence_sha256=digests,
        system_memory_samples=guards,
        comparison=compare(blocks, base["runtime"]["eog_token_ids"]),
    )


if __name__ == "__main__":
    print(json.dumps(review(*map(Path, sys.argv[1:])), indent=2, allow_nan=False))
