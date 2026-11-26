"""Bind all four real launches to the accepted 235B backend and its full weights before comparison."""

import argparse
import hashlib
import json
from pathlib import Path
import re
from audit235 import admission, verify_inventory, rebuild
from audit_bundle import attest
from physical_guard import accepted
from lookup235 import assess, ARMS, need


def review(lock, root, reference):
    base = rebuild(lock, reference)
    need(
        base == json.loads((reference / "summary.json").read_bytes()),
        "reference does not reproduce",
    )
    need(base["verified_weight_bytes"] == 85691001472, "incomplete 235B baseline")
    digests = {}
    panels = []
    pids = set()
    guards = {}

    def read(path):
        value = path.read_bytes()
        digests[str(path.relative_to(root))] = hashlib.sha256(value).hexdigest()
        return json.loads(value)

    for i, mode in enumerate(ARMS):
        d = root / ("block-" + str(i))
        bud = read(d / "header-budget.json")
        verify_inventory(lock, bud)
        need(
            bud == json.loads((reference / "header-budget.json").read_bytes()), "inventory differs"
        )
        cp = read(d / "checkpoint.json")
        rt = read(d / "runtime.json")
        sig = attest(lock, cp, rt, 95, 1, 128)
        need(
            sig == base["runtime"]
            and cp == json.loads((reference / "checkpoint.json").read_bytes()),
            "runtime or weights drift",
        )
        admission(read(d / "admission-transfer.json"), bud, "transfer")
        need(
            (d / "admission-transfer.json").read_bytes()
            == (reference / "admission-transfer.json").read_bytes(),
            "wrong original transfer admission",
        )
        admission(read(d / "admission.json"), bud, "launch")
        guard = read(d / "memory-guard.json")
        accepted(guard)
        guards[str(i)] = guard["phases"]
        g = read(d / "lookup-setting.json")
        fp = g.pop("process_fingerprint")
        need(
            re.fullmatch("[0-9a-f]{64}", fp)
            and fp not in pids
            and g
            == dict(
                block=i,
                mode=mode,
                slot_capable=(mode != "none"),
                ngram_key=4,
                ngram_value=8,
                draft_min=1,
                draft_max=8,
            )
            and type(g["slot_capable"]) is bool
            and type(g["block"]) is int,
            "invalid mode, slot or reused process",
        )
        pids.add(fp)
        b = read(d / "lookup-rows.json")
        need(b["block"] == i, "mismatched block")
        panels.append(b)
    return dict(
        schema=1,
        reference_summary_sha256=hashlib.sha256(
            (reference / "summary.json").read_bytes()
        ).hexdigest(),
        runtime=base["runtime"],
        verified_weight_bytes=base["verified_weight_bytes"],
        launches=len(pids),
        target_requests=sum(len(x["rows"]) for x in panels),
        comparison=assess(panels, base["runtime"]["eog_token_ids"]),
        memory_sample_phases=guards,
        evidence_sha256=digests,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("lock", type=Path)
    p.add_argument("run", type=Path)
    p.add_argument("reference", type=Path)
    a = p.parse_args()
    print(json.dumps(review(a.lock, a.run, a.reference), indent=2, allow_nan=False))
