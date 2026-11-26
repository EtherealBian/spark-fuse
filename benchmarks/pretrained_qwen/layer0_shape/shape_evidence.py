"""Audit the recorded projection experiment or materialize its fixed full-model reference."""

import argparse
import json
from pathlib import Path

from fork_witness import materialize
from physical_guard import accepted
from projection_witness import INDEX, PARENT, checked_index, execute, need, sha

PARENT_SUMMARY = "42cd5e78ddb83ce8df1b6eb36321a5fc64c5a887033f0298f75013e456ea91c7"
SUMMARY = "49b37e6ceddbb1bbfee830ceb8318c5db1f23623fcf7142b9dca3e50bcb3dd5d"
ADMISSION = "4b52c8e60c37dbf1b9fe17e641f261918de8d79294fc851f297c98c472cca81d"
GUARD = "98fc2e06f1b0ff1de1ce8de3d2e340913409465ac8b08951108dd2367a297eaf"


def verified_parent(parent, archive, dest):
    checked_index(parent / "index.json", PARENT)
    summary = (parent / "summary.json").read_bytes()
    need(sha(summary) == PARENT_SUMMARY, "unexpected full-model summary")
    materialize(parent, archive, dest)
    (dest / "summary.json").write_bytes(summary)


def audit(report, parent, archive, lock):
    source = Path(__file__).resolve().parent
    doc = checked_index(report / "index.json", INDEX)
    need(
        doc["sources"]
        == {
            n: sha((source / n).read_bytes())
            for n in ("kvshape.py", "collect_shape.py", "review_shape.py")
        },
        "measured sources differ",
    )
    ad = checked_index(report / "admission.json", ADMISSION)
    need(
        ad["admitted"] is True
        and ad["available_bytes"] >= ad["required_bytes"] == 84 * 2**30
        and ad["free_disk_bytes"] >= ad["required_disk_bytes"] == 2 * 2**30,
        "not admitted",
    )
    accepted(checked_index(report / "memory-guard.json", GUARD))
    full = checked_index(parent / "index.json", PARENT)
    m = doc["meta"]
    spec = json.loads(lock.read_bytes())
    expected = {}
    for name in ("config.json", "model.safetensors.index.json", "model-00001-of-00017.safetensors"):
        x = spec["files"][name]
        k = "git_blob" if "git_blob" in x else "sha256"
        expected[name] = {k: x[k], "bytes": x["size"]}
    need(
        m["verified_files"] == expected
        and m["revision"] == spec["revision"]
        and all(
            m[k] == full["meta"][k]
            for k in ("model", "revision", "device", "torch", "cuda", "transformers")
        ),
        "checkpoint or runtime mismatch",
    )
    need(
        m["inv_freq_dtype"] == "torch.float32" and m["selected_embedding_rows"] == 46,
        "unexpected probe configuration",
    )
    summary = checked_index(report / "summary.json", SUMMARY)
    out = execute(report, parent, archive, report / "projection-witness.bf16", None)
    for k in (
        "rows",
        "index_sha256",
        "repeat_order_stable",
        "full_model_first_bank_reproduced",
        "cells",
    ):
        need(out[k] == summary[k], "raw and compact audits differ: " + k)
    return out


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    sub = a.add_subparsers(dest="command", required=True)
    r = sub.add_parser("replay")
    p = sub.add_parser("materialize-parent")
    for n in ("report", "parent", "archive", "lock"):
        r.add_argument(n, type=Path)
    for n in ("parent", "archive", "destination"):
        p.add_argument(n, type=Path)
    z = a.parse_args()
    if z.command == "replay":
        print(json.dumps(audit(z.report, z.parent, z.archive, z.lock), indent=2, allow_nan=False))
    else:
        verified_parent(z.parent, z.archive, z.destination)
