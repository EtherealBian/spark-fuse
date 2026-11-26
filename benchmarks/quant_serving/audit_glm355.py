"""GLM-4.5 conditional admission and independently replayable single-Spark feasibility panel."""

import argparse, hashlib, json, shutil
from pathlib import Path
from audit_bundle import attest
from physical_guard import accepted
from glm_directory import inspect, part
from measure_glm355 import audit, require

GIB = 2**30
LOCK = "09707d77c9b250e9efae0fd5e8b6b44f9c2c74f96db0cca691b71bc28e68fa4c"
BUDGET = "394727c7c7ac97cf9ad3454a19bb2d4914deb563e671f24fa79eb73916acfb21"
BASE = "reports/qwen235b/runtime.json"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def serialize(obj):
    return (json.dumps(obj, indent=2, allow_nan=False) + "\n").encode()


def read_inputs(lock, budget):
    lm = lock.read_bytes()
    r = budget.read_bytes()
    require(sha(lm) == LOCK and sha(r) == BUDGET, "directory or checkpoint lock changed")
    return json.loads(lm), json.loads(r)


def inventory(lock, weights):
    require(sha(lock.read_bytes()) == LOCK, "wrong file lock")
    l = json.loads(lock.read_bytes())
    hdr = []
    items = {}
    for i, f in enumerate(l["files"]):
        n = f["name"]
        require(
            Path(n).name == n
            and not (weights / n).is_symlink()
            and (weights / n).stat().st_size == f["bytes"],
            "unsafe/missing shard",
        )
        with (weights / n).open("rb") as fp:
            raw = fp.read((16 if i == 0 else 1) * 2**20)
        hdr.append(raw)
        meta, t, *_ = part(raw, f["bytes"])
        items.update(t)
        if i == 0:
            require(
                meta["glm4moe.nextn_predict_layers"] == 1
                and meta["glm4moe.leading_dense_block_count"] == 3,
                "unsupported architecture",
            )
    require(
        all(items[k][0] == (5120, 151552) for k in ("token_embd.weight", "output.weight")),
        "wrong vocabulary",
    )
    b = inspect(l, hdr)
    require(
        sha(serialize(b)) == BUDGET,
        "directory lengths, hashes, shapes or byte ledger do not match pinned data",
    )
    return b


def assess(phase, b, available, disk):
    require(
        phase in ("launch", "transfer")
        and all(type(x) is int and x >= 0 for x in (available, disk)),
        "invalid system sample",
    )
    obj = b["source_object_bytes"]
    kv = b["uniform_dense_fp16_kv_bytes"]
    need = obj + kv + 25 * GIB
    space = 16 * GIB + (obj if phase == "transfer" else 0)
    return dict(
        phase=phase,
        available_ram=available,
        free_disk=disk,
        required_ram=need,
        required_disk=space,
        reserved_ram=25 * GIB,
        approved=available >= need and disk >= space,
    )


def admit(phase, b, d):
    require(d.is_dir() and not d.is_symlink(), "missing weights filesystem")
    s = [
        x.split()
        for x in Path("/proc/meminfo").read_text().splitlines()
        if x.startswith("MemAvailable:")
    ]
    require(len(s) == 1 and s[0][2] == "kB", "memory sample missing")
    return assess(phase, b, int(s[0][1]) * 1024, shutil.disk_usage(d).free)


def gate(phase, b, r):
    require(
        type(r["approved"]) is bool
        and r["approved"] is True
        and all(
            type(r[k]) is int
            for k in ("available_ram", "free_disk", "required_ram", "required_disk", "reserved_ram")
        )
        and r == assess(phase, b, r["available_ram"], r["free_disk"]),
        "missing measured headroom",
    )


def rebuild(lock, d, baseline):
    dig = {}

    def read(n):
        p = d / n
        require(
            p.is_file() and not p.is_symlink() and p.stat().st_size <= 16 * 2**20, "invalid receipt"
        )
        s = p.read_bytes()
        dig[n] = sha(s)
        return json.loads(s)

    b = read("header-budget.json")
    read_inputs(lock, d / "header-budget.json")
    cp = read("checkpoint.json")
    r = read("runtime.json")
    sig = attest(lock, cp, r, 94, 1, 128)
    base = json.loads(baseline.read_bytes())
    require(
        all(
            sig[n] == base[n]
            for n in (
                "device",
                "llama_cpp_revision",
                "server_sha256",
                "runtime_lib_sha256",
                "driver",
            )
        )
        and all(t < 151552 for t in sig["eog_token_ids"]),
        "backend or model EOG mismatch",
    )
    mem = read("memory-guard.json")
    accepted(mem)
    gate("transfer", b, read("admission-transfer.json"))
    gate("launch", b, read("admission.json"))
    v = audit(read("rows.json"), sig["eog_token_ids"])
    eq = sum(q["repeated_greedy_exact"] for q in v["cases"].values())
    weight = sum(
        x["reported_mib"]
        for x in r["backend_buffers"]
        if x["device"] == "CUDA0" and x["kind"] == "model"
    )
    return dict(
        schema="sparkfuse.glm45-feasibility.v1",
        model=cp["model"],
        revision=cp["revision"],
        verified_weight_bytes=cp["total_bytes"],
        backend_placement_layers_including_output=94,
        header_blocks_including_nextn=93,
        header_nextn_predict_layers=1,
        backend_cuda_model_mib=weight,
        reference_runtime_sha256=sha(baseline.read_bytes()),
        runtime=sig,
        system_memory_observations=mem,
        serving=v,
        matching_repeat_pairs=eq,
        tested_repeatability_gate=eq == 4,
        evidence_sha256=dig,
        scope="Pinned mixed UD-IQ1_S on one GB10. Counterbalanced fixed-length cold prompts; all-layer placement is a backend receipt, not a claim that the NextN/MTP layer ran. No natural-language accuracy or equivalence to higher precision; backend model buffer is not peak memory.",
    )


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("mode", choices=("audit", "transfer", "launch", "inventory"))
    a.add_argument("lock", type=Path)
    a.add_argument("directory", type=Path)
    a.add_argument("--budget", type=Path)
    a.add_argument("--baseline", type=Path, default=Path(BASE))
    u = a.parse_args()
    v = (
        rebuild(u.lock, u.directory, u.baseline)
        if u.mode == "audit"
        else inventory(u.lock, u.directory)
        if u.mode == "inventory"
        else admit(u.mode, read_inputs(u.lock, u.budget)[1], u.directory)
    )
    print(serialize(v).decode(), end="")
    raise SystemExit(0 if v.get("approved", True) else 75)
