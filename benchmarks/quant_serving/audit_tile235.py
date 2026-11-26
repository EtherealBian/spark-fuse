"""Verify each launch against the independent published baseline before comparing microbatch sizes."""

import hashlib
import json
import sys
from pathlib import Path
from audit235 import admission, verify_inventory, rebuild
from audit_bundle import attest
from physical_guard import accepted
from tile235 import TILES, compare, require


def check(lock, root, ref):
    base = rebuild(lock, ref)
    require(
        base == json.loads((ref / "summary.json").read_bytes()), "public reference did not replay"
    )
    hashed = {}
    blocks = []
    guards = {}

    def read(d, n):
        b = (d / n).read_bytes()
        hashed[str((d / n).relative_to(root))] = hashlib.sha256(b).hexdigest()
        return json.loads(b)

    for i, u in enumerate(TILES):
        d = root / f"block-{i}"
        bud = read(d, "header-budget.json")
        verify_inventory(lock, bud)
        cp = read(d, "checkpoint.json")
        rt = read(d, "runtime.json")
        sig = attest(lock, cp, rt, 95, 1, u)
        require(
            sig == base["runtime"]
            and cp == json.loads((ref / "checkpoint.json").read_bytes())
            and bud == json.loads((ref / "header-budget.json").read_bytes()),
            "baseline weight/backend changed",
        )
        ad = read(d, "admission-transfer.json")
        admission(ad, bud, "transfer")
        require(
            ad == json.loads((ref / "admission-transfer.json").read_bytes()),
            "wrong original transfer admission",
        )
        admission(read(d, "admission.json"), bud, "launch")
        g = read(d, "memory-guard.json")
        accepted(g)
        guards[str(i)] = g["phases"]
        blocks.append((read(d, "tile-setting.json"), read(d, "rows.json")))
    s = compare(blocks, json.loads((ref / "rows.json").read_bytes()), sig["eog_token_ids"])
    return dict(
        schema=1,
        verified_weight_bytes=base["verified_weight_bytes"],
        reference_summary_sha256=hashlib.sha256((ref / "summary.json").read_bytes()).hexdigest(),
        tile_sequence=list(TILES),
        requests=32,
        warmups_excluded=4,
        comparison=s,
        system_memory_samples=guards,
        inputs_sha256=hashed,
    )


if __name__ == "__main__":
    print(json.dumps(check(*map(Path, sys.argv[1:])), indent=2, allow_nan=False))
