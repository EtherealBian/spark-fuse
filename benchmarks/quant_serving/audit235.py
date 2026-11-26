"""Rebuild the single-Spark 235B feasibility receipt from individual observations."""

import argparse
import hashlib
import json
from pathlib import Path
from audit_bundle import attest
from measure235 import audit, require
from physical_guard import accepted
from shard_headers import inspect

GIB = 2**30


def inventory(lock, directory):
    config = json.loads(lock.read_bytes())
    headers = []
    for entry in config["files"]:
        name = entry["name"]
        require(Path(name).name == name and name.endswith(".gguf"), "bad GGUF file")
        with (directory / name).open("rb") as f:
            headers.append(f.read(8 * 2**20))
    return inspect(config, headers)


def admission(receipt, budget, phase):
    obj = budget["source_object_bytes"]
    kv = budget["uniform_dense_fp16_kv_bytes"]
    required = obj + kv + 22 * GIB
    diskneed = 16 * GIB + (obj if phase == "transfer" else 0)
    require(
        all(
            type(receipt[k]) is int and receipt[k] >= 0
            for k in (
                "available_ram",
                "required_ram",
                "free_disk",
                "required_disk",
                "reserved_ram",
                "headroom_after_object_and_kv",
            )
        )
        and receipt["approved"] is True
        and receipt["phase"] == phase
        and receipt["required_ram"] == required
        and receipt["required_disk"] == diskneed
        and receipt["reserved_ram"] == 22 * GIB
        and receipt["available_ram"] >= required
        and receipt["free_disk"] >= diskneed
        and receipt["headroom_after_object_and_kv"] == receipt["available_ram"] - obj - kv,
        "resource admission cannot be reconciled",
    )


def verify_inventory(lock, budget):
    m = json.loads(lock.read_bytes())
    require(
        m["revision"] == "437d6915c5c512869e4cf8b16b840bbe3fb172bc"
        and m["format"] == "Q2_K"
        and m["model"] == "unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF"
        and budget["architecture"] == "qwen3moe"
        and all(budget[k] == m[k] for k in ("model", "revision", "format"))
        and len(m["files"]) == len(budget["shards"]) == 2
        and (
            budget["layers"],
            budget["total_experts"],
            budget["active_experts"],
            budget["context_per_slot"],
            budget["slots"],
        )
        == (94, 128, 8, 4096, 1),
        "incorrect 235B architecture or split",
    )
    # Bind the public split identities, not just their aggregate size. The native
    # GGUF directory digests can be re-created with `inventory` from verified files.
    identity = [
        (
            "00001",
            49906751904,
            "7cb5dfd9d11f6297477f359439ad906eb92d62fc96b1783871159e7204d427ff",
            "fb336776f31ee8e364d3880ef60b1301b20e0b4e5d6594a82656e3b0308bba0c",
        ),
        (
            "00002",
            35784249568,
            "73801de2867057497a7fb6d0f0350ef7832b98f25c264f3a9909313af785672d",
            "56458562c8d8b025dd1a22858e6d24ddd0e019392a3de6dbe542ae9fcfa68de1",
        ),
    ]
    for shard, entry, (part, size, digest, header) in zip(
        budget["shards"], m["files"], identity, strict=True
    ):
        require(
            entry
            == dict(
                name=f"Qwen3-235B-A22B-Instruct-2507-Q2_K-{part}-of-00002.gguf",
                bytes=size,
                sha256=digest,
            )
            and shard["directory_sha256"] == header,
            "unrecognized shard/directory digest",
        )
        require(
            shard["name"] == entry["name"] and shard["source_object_bytes"] == entry["bytes"],
            "shard budget mismatch",
        )
    g = budget["packed_tensor_bytes_by_type"]
    require(
        set(g) == {"f32", "q2_K", "q3_K", "q4_K", "q6_K"}
        and all(type(x) is int and x > 0 for x in g.values())
        and sum(g.values()) == budget["packed_tensor_bytes"]
        and sum(s["tensor_data_start"] for s in budget["shards"])
        == budget["aligned_directory_bytes"]
        and budget["tensor_padding_or_trailer_bytes"] >= 0
        and budget["packed_tensor_bytes"]
        + budget["aligned_directory_bytes"]
        + budget["tensor_padding_or_trailer_bytes"]
        == budget["source_object_bytes"]
        == sum(s["bytes"] for s in m["files"])
        == 85691001472
        and budget["uniform_dense_fp16_kv_bytes"] == 2 * 4096 * 94 * 4 * (128 + 128)
        and budget["tensors"] == sum(x["tensors"] for x in budget["shards"]) == 1131
        and budget["logical_gguf_elements"] == 235093634560,
        "235B byte inventory mismatch",
    )


def rebuild(lock, directory):
    hashes = {}

    def read(n):
        b = (directory / n).read_bytes()
        hashes[n] = hashlib.sha256(b).hexdigest()
        return json.loads(b)

    budget = read("header-budget.json")
    verify_inventory(lock, budget)
    cp = read("checkpoint.json")
    rt = read("runtime.json")
    sig = attest(lock, cp, rt, 95, 1, 128)
    require(all(t < 152064 for t in sig["eog_token_ids"]), "unexpected EOG for this vocabulary")
    mem = read("memory-guard.json")
    accepted(mem)
    for phase, name in [("transfer", "admission-transfer.json"), ("launch", "admission.json")]:
        admission(read(name), budget, phase)
    panel = audit(read("rows.json"), sig["eog_token_ids"])
    weight = sum(
        t["reported_mib"]
        for t in rt["backend_buffers"]
        if t["kind"] == "model" and t["device"] == "CUDA0"
    )
    return dict(
        schema=1,
        model=cp["model"],
        revision=cp["revision"],
        verified_weight_bytes=cp["total_bytes"],
        runtime=sig,
        backend_cuda_model_mib=weight,
        system_memory_observations=mem,
        serving=panel,
        evidence_sha256=hashes,
        scope="Pinned 2-bit mixed MoE placement and fixed-length greedy repeatability only. System memory minima are one-second samples, not GPU allocation peaks. No comparison to unquantized quality or another model.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["inventory", "audit"])
    p.add_argument("lock", type=Path)
    p.add_argument("directory", type=Path)
    a = p.parse_args()
    print(
        json.dumps(
            (inventory if a.mode == "inventory" else rebuild)(a.lock, a.directory),
            indent=2,
            allow_nan=False,
        )
    )
