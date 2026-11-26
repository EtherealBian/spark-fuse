"""Attest a complete dual-slot streaming run; performance is derived only from raw client clocks."""

import argparse
import hashlib
import json
from pathlib import Path
from audit_bundle import ROOT, attest
from stream_wave import audit


def rebuild(directory):
    sha = {}

    def read(name):
        raw = (directory / name).read_bytes()
        sha[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    cp = read("checkpoint.json")
    rt = read("runtime.json")
    signature = attest(ROOT / "qwen72b-q4km.lock.json", cp, rt, 81, 2, 128)
    out = audit(read("rows.json"), signature["eog_token_ids"])
    return dict(
        schema=1,
        model=cp["model"],
        revision=cp["revision"],
        verified_weight_bytes=cp["total_bytes"],
        runtime=signature,
        evidence_sha256=sha,
        panel=out,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    a = p.parse_args()
    print(json.dumps(rebuild(a.directory), indent=2, allow_nan=False))
