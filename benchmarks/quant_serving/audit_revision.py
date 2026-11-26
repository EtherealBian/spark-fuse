"""Bind an edited-manifest experiment to an independently rebuilt 235B baseline."""

import argparse
import hashlib
import json
from pathlib import Path
from audit235 import rebuild
from revision_canary import audit, need


def report(lock, directory, reference):
    need(not directory.samefile(reference), "reference must be an independent baseline")
    base = rebuild(lock, reference)
    receipt = (reference / "summary.json").read_bytes()
    need(base == json.loads(receipt), "stored baseline summary does not match raw evidence")
    from audit235 import admission, verify_inventory
    from audit_bundle import attest
    from physical_guard import accepted

    hashes = {}

    def read(name):
        raw = (directory / name).read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    budget = read("header-budget.json")
    verify_inventory(lock, budget)
    cp = read("checkpoint.json")
    rt = read("runtime.json")
    signature = attest(lock, cp, rt, 95, 1, 128)
    mem = read("memory-guard.json")
    accepted(mem)
    for phase, name in [("transfer", "admission-transfer.json"), ("launch", "admission.json")]:
        admission(read(name), budget, phase)
    for key in ("checkpoint.json", "header-budget.json", "admission-transfer.json"):
        need(
            hashes[key] == base["evidence_sha256"][key],
            "not the exact original transfer/weights/inventory",
        )
    panel = audit(read("rows.json"), signature["eog_token_ids"])
    weight = sum(
        t["reported_mib"]
        for t in rt["backend_buffers"]
        if t["kind"] == "model" and t["device"] == "CUDA0"
    )
    need(weight == base["backend_cuda_model_mib"], "different CUDA model buffer")
    own = dict(
        schema=1,
        model=cp["model"],
        revision=cp["revision"],
        verified_weight_bytes=cp["total_bytes"],
        runtime=signature,
        backend_cuda_model_mib=weight,
        system_memory_observations=mem,
        serving=panel,
        evidence_sha256=hashes,
        scope=base["scope"],
    )
    for key in ("model", "revision", "verified_weight_bytes", "runtime"):
        need(base[key] == own[key], "checkpoint, driver, device, EOG or serving binary drift")
    for key in ("checkpoint.json", "header-budget.json"):
        need(base["evidence_sha256"][key] == own["evidence_sha256"][key], "split evidence drift")
    need(
        base["evidence_sha256"]["rows.json"] != own["evidence_sha256"]["rows.json"],
        "same workload evidence",
    )
    own.update(
        panel="edited-manifest-cache-canary",
        reference_summary_sha256=hashlib.sha256(receipt).hexdigest(),
        scope="Single DGX Spark 235B Q2_K; eight counterfactual manifest lookups in a separately launched backend. Compare a cold target to reuse after its older revision. Not a general model-quality, BF16-equivalence or production latency claim.",
    )
    return own


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("lock", "directory", "reference"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    print(json.dumps(report(a.lock, a.directory, a.reference), indent=2, allow_nan=False))
