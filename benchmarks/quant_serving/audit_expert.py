"""Verify a published dense/MoE ledger without weights. Optional local-directory replay is stricter."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from expert_ledger import BATCHES, inspect, need

MODELS = {
    "72": ("qwen72b-q4km.lock.json", "qwen72b/catalog", "runtime.json", "qwen2", 80, 0, 0),
    "120": (
        "gpt-oss-120b-mxfp4.lock.json",
        "gpt-oss-120b",
        "runtime-0.json",
        "gpt-oss",
        36,
        128,
        4,
    ),
    "235": ("qwen235b-q2k.lock.json", "qwen235b", "runtime.json", "qwen3moe", 94, 128, 8),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(p):
    return json.loads(p.read_bytes())


def pos(v):
    return type(v) is int and v > 0


def check_one(key, cfg, report_dir, receipt, src, reports, weights=None):
    name, sub, rt, arch, layers, n, k = cfg
    lp = src / name
    pin = load(lp)
    cp = reports / sub / "checkpoint.json"
    run = reports / sub / rt
    a = load(report_dir / (key + ".json"))
    c = load(cp)
    backend = load(run)
    need(
        a["schema"] == "sparkfuse.gguf.expert-ledger.v1"
        and all(a[q] == pin[q] for q in ("model", "revision", "format"))
        and a["architecture"] == arch
        and a["layers"] == layers,
        "wrong model identity",
    )
    need(
        receipt["lock_sha256"] == digest(lp) == c["lock_sha256"]
        and receipt["runtime_receipt_sha256"] == digest(run)
        and receipt["prior_full_checksum_receipt_sha256"] == digest(cp)
        and receipt["ledger_sha256"] == digest(report_dir / (key + ".json")),
        "input receipt no longer matches",
    )
    need(
        c["files"] == pin["files"]
        and c["total_bytes"] == a["source_object_bytes"]
        and c["revision"] == pin["revision"],
        "different full-checksum manifest",
    )
    need(
        receipt["abi_sha256"] == backend["runtime_lib_sha256"]["libggml-base.so.0.9.7"],
        "different measured GGML ABI",
    )
    need(len(a["shards"]) == len(pin["files"]), "incomplete shard list")
    total = packed = tensors = 0
    for x, s in zip(pin["files"], a["shards"], strict=True):
        need(
            (s["name"], s["object_bytes"], s["declared_object_sha256"])
            == (x["name"], x["bytes"], x["sha256"]),
            "shard lock differs",
        )
        need(
            pos(s["tensors"])
            and pos(s["aligned_directory_bytes"])
            and type(s["padding_or_trailer_bytes"]) is int
            and s["padding_or_trailer_bytes"] >= 0
            and re.fullmatch("[0-9a-f]{64}", s["directory_sha256"]),
            "bad shard dimensions or header digest",
        )
        payload = x["bytes"] - s["aligned_directory_bytes"] - s["padding_or_trailer_bytes"]
        need(payload > 0, "empty shard")
        total += x["bytes"]
        packed += payload
        tensors += s["tensors"]
    need(
        (total, packed, tensors)
        == (a["source_object_bytes"], a["packed_tensor_bytes"], a["tensor_count"])
        and (packed, tensors)
        == (receipt["independent_packed_tensor_bytes"], receipt["independent_reader_tensors"]),
        "independent reader, shard totals or tensor count differ",
    )
    for tab in (a["packed_bytes_by_type"], a["packed_bytes_by_group"]):
        need(
            tab and all(pos(v) for v in tab.values()) and sum(tab.values()) == packed,
            "bad byte accounting",
        )
    routed = a["packed_bytes_by_group"].get("routed_expert", 0)
    bounds = a["routed_expert_union_bounds"]
    need(
        (a["experts_per_layer"] or 0, a["selected_per_token_per_layer"] or 0) == (n, k),
        "wrong expert geometry",
    )
    if n:
        need(
            routed > 0
            and routed % n == 0
            and bounds
            == {
                str(b): dict(min_bytes=(routed // n) * k, max_bytes=(routed // n) * min(n, b * k))
                for b in BATCHES
            },
            "invalid disjoint/shared expert bounds",
        )
    else:
        need(routed == 0 and not bounds, "dense control received routed experts")
    if weights is not None:
        need(inspect(pin, weights) == a, "local GGUF directories differ from published ledger")
    return dict(
        architecture=arch,
        tensors=tensors,
        object_bytes=total,
        packed_bytes=packed,
        routed_bytes=routed,
        one_token_expert_bytes=bounds.get("1", {}).get("min_bytes", 0),
    )


def audit(folder, src, reports, local=None):
    v = load(folder / "verified.json")
    need(
        v["schema"] == 1
        and set(v["models"]) == set(MODELS)
        and set(v["readers"]) == {"gguf_reader.py", "constants.py", "quants.py"}
        and all(re.fullmatch("[0-9a-f]{64}", h) for h in v["readers"].values()),
        "incomplete independent-reader receipt",
    )
    local = local or {}
    need(set(local).issubset(MODELS), "unknown weight directory")
    return {
        k: check_one(k, cfg, folder, v["models"][k], src, reports, local.get(k))
        for k, cfg in MODELS.items()
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reports", type=Path, help="repository reports directory")
    ap.add_argument("--weights72", type=Path)
    ap.add_argument("--weights120", type=Path)
    ap.add_argument("--weights235", type=Path)
    args = ap.parse_args()
    src = Path(__file__).resolve().parent
    loc = {
        k: getattr(args, "weights" + k) for k in MODELS if getattr(args, "weights" + k) is not None
    }
    print(
        json.dumps(
            audit(args.reports / "expert-ledger", src, args.reports, loc), indent=2, allow_nan=False
        )
    )
