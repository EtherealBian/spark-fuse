"""CPU verifier. Oracle agreement and padding are independent outcomes, not acceptance filters."""

import hashlib
import json
import sys
from pathlib import Path
import numpy as np

STAGES = {
    "embedding": 5120,
    "normalized": 5120,
    "cos": 128,
    "sin": 128,
    "key_linear": 1024,
    "value": 1024,
    "key_rotated": 1024,
}
ARMS = ("whole", "split", "padded")
IH = "6d46f39acbf2b71387f1ea1ea00191380f74fad7b8d999969dfafa27a014ac6d"
SH = "42cd5e78ddb83ce8df1b6eb36321a5fc64c5a887033f0298f75013e456ea91c7"


def need(c, m):
    if not c:
        raise ValueError(m)


def digest(b):
    return hashlib.sha256(b).hexdigest()


def read(p):
    return json.loads(p.read_bytes())


def segments(b, pos, metadata):
    r = {}
    need(list(metadata) == list(STAGES), "stage membership/order")
    for k, n in STAGES.items():
        size = 3 * n * 2
        s = b[pos : pos + size]
        need(
            len(s) == size and metadata[k] == dict(offset=pos, bytes=size, sha256=digest(s)),
            "bad offset/extent/hash",
        )
        x = np.frombuffer(s, dtype="<u2").reshape(3, n)
        need(not np.any((x & 0x7F80) == 0x7F80), "nonfinite")
        r[k] = x
        pos += size
    return pos, r


def comparison(x, y):
    return {k: (x[k] != y[k]).sum(1).tolist() for k in STAGES}


def oracle(p, k, t, arm, record):
    n, f, br = k
    filename = f"{n}-{f}-{br}-{t}-{arm}-kv.bf16"
    b = (p / filename).read_bytes()
    need(
        len(b) == record["kv"]["bytes"] == 786432 and digest(b) == record["kv"]["sha256"],
        "altered parent KV",
    )
    x = np.frombuffer(b, dtype="<u2").reshape(128, 3, 8, 128)
    return {"key_rotated": x[0].reshape(3, 1024), "value": x[1].reshape(3, 1024)}


def audit(p, parent, lock):
    from physical_guard import accepted

    accepted(read(p / "memory-guard.json"))
    ad = read(p / "admission.json")
    need(
        ad["admitted"] is True
        and ad["available_bytes"] >= ad["required_bytes"] == 84 * 2**30
        and ad["free_disk_bytes"] >= ad["required_disk_bytes"] == 2 * 2**30,
        "admission",
    )
    need(
        digest((parent / "index.json").read_bytes()) == IH
        and digest((parent / "summary.json").read_bytes()) == SH,
        "parent changed",
    )
    pr = read(parent / "index.json")
    old = {(tuple(z["cell"]), z["repeat"], z["arm"]): z for z in pr["records"]}
    doc = read(p / "index.json")
    m = doc["meta"]
    need(
        doc["schema"] == "sparkfuse.first-bank-probe.v1"
        and doc["parent_index_sha256"] == IH
        and all(
            m[k] == pr["meta"][k]
            for k in ("model", "revision", "device", "torch", "cuda", "transformers")
        )
        and m["inv_freq_dtype"] == "torch.float32"
        and m["selected_embedding_rows"] > 0,
        "model/runtime",
    )
    need(
        doc["sources"]
        == {
            n: digest(Path(__file__).with_name(n).read_bytes())
            for n in ("kvshape.py", "collect_shape.py", "review_shape.py")
        },
        "source drift",
    )
    spec = read(lock)
    files = ("model.safetensors.index.json", "config.json", "model-00001-of-00017.safetensors")
    need(
        m["verified_files"]
        == {
            n: {
                ("git_blob" if "git_blob" in spec["files"][n] else "sha256"): spec["files"][n].get(
                    "git_blob", spec["files"][n].get("sha256")
                ),
                "bytes": spec["files"][n]["size"],
            }
            for n in files
        }
        and m["revision"] == spec["revision"],
        "input shard/config/index",
    )
    keys = [(n, f, b) for n in (128, 2048) for f in (1, 2) for b in (0, 1)]
    expected = [
        (k, t, a)
        for t in (0, 1)
        for k in (keys if not t else keys[::-1])
        for a in (ARMS if not t else ARMS[::-1])
    ]
    b = (p / "samples.bf16").read_bytes()
    pos = 0
    data = {}
    need(len(doc["rows"]) == len(expected), "incomplete panel")
    for row, (k, t, a) in zip(doc["rows"], expected, strict=True):
        n = k[0]
        j = n * 3 // 4
        need(
            (tuple(row["cell"]), row["repeat"], row["arm"]) == (k, t, a)
            and row["projected_rows"] == {"whole": [n], "split": [j, n - j], "padded": [n, n]}[a],
            "row order/computation",
        )
        pos, data[k, t, a] = segments(b, pos, row["stages"])
    need(pos == len(b), "trailing bytes")
    reports = []
    for k in keys:
        rr = []
        for t in (0, 1):
            whole, split, padded = [data[k, t, a] for a in ARMS]
            checks = {}
            for name, alias, v in [("whole", "whole", whole), ("split", "inplace", split)]:
                real = oracle(parent, k, t, alias, old[k, t, alias])
                checks[name] = {n: (v[n] != real[n]).sum(1).tolist() for n in real}
            rr.append(
                dict(
                    full_model_bank_mismatches=checks,
                    whole_vs_split=comparison(whole, split),
                    whole_vs_padded=comparison(whole, padded),
                )
            )
        stable = all(
            all(np.array_equal(data[k, 0, a][s], data[k, 1, a][s]) for s in STAGES) for a in ARMS
        )
        reports.append(dict(cell=list(k), cross_order_bitwise=stable, rounds=rr))
    matches = all(
        not any(d)
        for c in reports
        for r in c["rounds"]
        for vals in r["full_model_bank_mismatches"].values()
        for d in vals.values()
    )
    stable = all(r["cross_order_bitwise"] for r in reports)
    return dict(
        schema="sparkfuse.first-bank-diagnosis.v1",
        rows=len(expected),
        index_sha256=digest((p / "index.json").read_bytes()),
        samples_sha256=digest(b),
        full_model_first_bank_reproduced=matches,
        repeat_order_stable=stable,
        cells=reports,
        scope="Block zero only. Counts compare three selected positions per stage; padded subcalls project extra dummy rows without writing cache. No speed, full-logit equivalence, specific GEMM dispatch or end-to-end repair claim.",
    )


if __name__ == "__main__":
    print(json.dumps(audit(*map(Path, sys.argv[1:])), indent=2, allow_nan=False))
