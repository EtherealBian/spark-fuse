"""Rebuild cross-panel evidence from immutable inputs; never read an existing summary."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from audit_replay import audit as audit_single
from audit_routes import audit as audit_routes
from moe_blocks import audit as audit_moe
from runtime_receipt import PIN

ROOT = Path(__file__).resolve().parent
HEX = re.compile("[0-9a-f]{64}")


def check(ok, msg):
    if not ok:
        raise ValueError(msg)


def attest(lock, checkpoint, runtime, layers, slots, ub):
    raw = lock.read_bytes()
    m = json.loads(raw)
    entries = checkpoint["files"]
    check(
        checkpoint["schema"] == 1
        and checkpoint["lock_sha256"] == hashlib.sha256(raw).hexdigest()
        and all(checkpoint[k] == m[k] for k in ("model", "revision", "format", "files"))
        and checkpoint["total_bytes"] == sum(e["bytes"] for e in entries),
        "checkpoint receipt differs from published lock",
    )
    check(
        runtime["context"]
        == dict(n_seq_max=slots, n_ctx=slots * 4096, n_ctx_seq=4096, n_batch=512, n_ubatch=ub)
        and runtime["llama_cpp_revision"] == PIN
        and runtime["device"] == "NVIDIA GB10"
        and runtime["gpu_layers_including_output"] == layers,
        "wrong execution context",
    )
    check(
        bool(HEX.fullmatch(runtime["server_sha256"]))
        and re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", runtime["driver"]),
        "bad binary or driver identifier",
    )
    eog = runtime.get("eog_token_ids")
    check(
        isinstance(eog, list)
        and bool(eog)
        and all(type(v) is int and 0 <= v < (152064 if layers == 81 else 201088) for v in eog)
        and sorted(set(eog)) == eog,
        "invalid EOG receipt",
    )
    if layers == 81:
        check(
            eog == [128247, 151643, 151645, 151662, 151663, 151664],
            "unexpected pinned Qwen EOG set",
        )
    deps = runtime["runtime_lib_sha256"]
    check(
        all(HEX.fullmatch(v) for v in deps.values())
        and all(
            any(k.startswith(p) for k in deps)
            for p in ("libggml-base.so", "libggml-cuda.so", "libllama.so")
        ),
        "incomplete mapped runtime manifest",
    )
    memory = runtime["backend_buffers"]
    check(
        bool(memory)
        and all(
            x["kind"] in ("model", "KV", "compute")
            and type(x["reported_mib"]) in (float, int)
            and math.isfinite(x["reported_mib"])
            and x["reported_mib"] >= 0
            for x in memory
        ),
        "bad buffer snapshot",
    )
    check(
        any(
            x["device"] == "CUDA0" and x["kind"] == "model" and x["reported_mib"] > 0
            for x in memory
        ),
        "missing CUDA weight buffer",
    )
    return {
        k: runtime[k]
        for k in (
            "llama_cpp_revision",
            "server_sha256",
            "runtime_lib_sha256",
            "driver",
            "eog_token_ids",
            "device",
            "gpu_layers_including_output",
        )
    }


def excluded_pilot(pilot, reference, eog):
    """Validate the archived early-stop counterexample, never mix it into latency estimates."""
    check(
        pilot["schema"] == 1
        and pilot["attempted_pairs"] == 8
        and pilot["explicit_eog_mask"] is False
        and pilot["request_ignore_eos"] is True
        and len(pilot["rejected_pairs"]) == 1,
        "invalid pilot exclusion receipt",
    )
    row = pilot["rejected_pairs"][0]
    peers = [r for r in reference if r["case"] == row["case"] and r["length"] == row["length"]]
    check(
        len(peers) == 1
        and all(
            row[k] == peers[0][k]
            for k in ("target", "primer", "shared", "target_ids", "primer_ids", "order")
        ),
        "pilot is not the same matched workload",
    )
    cold, warm = row["cold"], row["reuse"]
    for arm, reused in ((cold, 0), (warm, row["shared"])):
        check(
            arm["reused"] == reused
            and arm["processed"] == row["length"] - reused
            and len(arm["ids"]) == arm["predicted"]
            and all(type(t) is int and 0 <= t < 152064 for t in arm["ids"]),
            "pilot counter or token receipt mismatch",
        )
    check(
        cold["stop"] == "limit"
        and cold["predicted"] == 24
        and warm["stop"] == "eos"
        and 0 < warm["predicted"] < 24
        and warm["ids"][-1] in eog,
        "pilot did not evidence the stated early EOS",
    )
    return dict(
        case=row["case"],
        input_tokens=row["length"],
        cold_tokens=cold["predicted"],
        reuse_tokens=warm["predicted"],
        premature_terminal_id=warm["ids"][-1],
        included_in_timings=False,
    )


def bundle(kind, d):
    hashes = {}
    exclusions = {}

    def read(name):
        b = (d / name).read_bytes()
        hashes[name] = hashlib.sha256(b).hexdigest()
        return json.loads(b)

    if kind == "72b":
        lock = ROOT / "qwen72b-q4km.lock.json"
        cp = read("prefix-checkpoint.json")
        cp2 = read("routing-checkpoint.json")
        r = read("prefix-runtime.json")
        r2 = read("routing-runtime.json")
        base = attest(lock, cp, r, 81, 1, 128)
        check(
            cp == cp2 and base == attest(lock, cp2, r2, 81, 2, 128),
            "unmatched checkpoints or binaries across panels",
        )
        prefix = read("prefix-rows.json")
        traces = read("routing-rows.json")
        exclusions["request_only_eos"] = excluded_pilot(
            read("pilot-exclusion.json"), prefix, base["eog_token_ids"]
        )
        single = audit_single(prefix)
        routes = audit_routes(traces, read("routing-inputs.json"))
        outputs = [row[a]["ids"] for row in prefix for a in ("cold", "reuse")] + [
            row["ids"] for t in traces for row in t["rows"]
        ]
        panels = dict(prefix=single, routing=routes)
    elif kind == "120b":
        lock = ROOT / "gpt-oss-120b-mxfp4.lock.json"
        cp = read("checkpoint.json")
        signatures = []
        blocks = []
        for i, ub in enumerate((128, 512, 512, 128)):
            signatures.append(attest(lock, cp, read(f"runtime-{i}.json"), 37, 1, ub))
            blocks.append(read(f"block-{i}.json"))
        base = signatures[0]
        check(all(s == base for s in signatures), "unmatched runtimes")
        panels = dict(microbatch=audit_moe(blocks))
        outputs = [r["ids"] for block in blocks for r in block["rows"]]
    else:
        raise ValueError("unknown panel")
    forbidden = set(base["eog_token_ids"])
    check(
        all(forbidden.isdisjoint(ids) for ids in outputs), "fixed-length stream includes masked EOG"
    )
    return dict(
        schema=1,
        model=cp["model"],
        revision=cp["revision"],
        format=cp["format"],
        verified_weight_bytes=cp["total_bytes"],
        runtime=base,
        evidence_sha256=dict(sorted(hashes.items())),
        panels=panels,
        excluded=exclusions,
        all_tokens_exact=all(p["all_tokens_exact"] for p in panels.values()),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("72b", "120b"))
    p.add_argument("directory", type=Path)
    a = p.parse_args()
    print(json.dumps(bundle(a.kind, a.directory), indent=2, allow_nan=False))
