"""Pure-stdlib offline replay of paired 32B prefix snapshot waves from raw measurements."""

import hashlib
import json
import math
import re
from pathlib import Path
from statistics import median

ARMS = ("cold", "snapshot")
LENS = (128, 2048)
STEPS = 8
V = 152064
EAGER = "c271a47061af2aced3c41520e7a3b0c772e9706a55387b3786949099b3c978d3"
RESIDENT = "246c0657ed97aabd8d2d55faf01f95786a73cc8ecf64c8ad200b59ad49cf12c4"
SOURCES = ("prefix_snapshot.py", "audit_fork32.py", "measure_fork32.py")
BYTES_PER_POSITION = 2 * 64 * 8 * 128 * 2


def need(x, msg):
    if not x:
        raise ValueError(msg)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    return {k: sha(Path(__file__).with_name(k)) for k in SOURCES}


def refs(reports):
    result = []
    for k, h in [("eager", EAGER), ("resident", RESIDENT)]:
        f = reports / ("qwen32b-" + k + ".json")
        need(sha(f) == h, "32B reference drift")
        result.append(json.loads(f.read_bytes()))
    return result


def original(o, n, i):
    a, b = [
        next(
            z for z in next(c for c in x["cases"] if c["length"] == n)["checks"] if z["prompt"] == i
        )
        for x in o
    ]
    need(
        a["input_ids"] == b["input_ids"]
        and a["free"] == b["free"] == a["reference"] == b["reference"],
        "oracles disagree",
    )
    return tuple(a["input_ids"]), a["reference"][:STEPS]


def workload(o, n, i):
    p = n * 3 // 4
    a, g = original(o, n, i)
    b, _ = original(o, n, 3 - i)
    alt = a[:p] + b[p:]
    need(len(a) == len(alt) == n and a != alt, "branches must differ")
    return (a, alt), g


def order(j):
    return (ARMS if j in (0, 3) else ARMS[::-1]), ((0, 1) if j < 2 else (1, 0))


def timing(v):
    return type(v) in (float, int) and math.isfinite(v) and v > 0


def toks(v):
    return type(v) is list and len(v) == STEPS and all(type(i) is int and 0 <= i < V for i in v)


def case(c, o, n, i):
    targets, gold = workload(o, n, i)
    p = n * 3 // 4
    need(
        type(c["length"]) is int
        and type(c["family"]) is int
        and (c["length"], c["family"]) == (n, i)
        and type(c["prefix"]) is int
        and c["prefix"] == p
        and c["inputs"] == [list(x) for x in targets]
        and all(type(t) is int for x in c["inputs"] for t in x),
        "different input or prefix",
    )
    need(
        c["snapshot_bytes"] == p * BYTES_PER_POSITION
        and type(c["snapshot_bytes"]) is int
        and c["capacity"] == n + 32,
        "KV accounting/capacity",
    )
    probes = c["diagnostics"]
    need(
        [(d["arm"], d["branch"]) for d in probes] == [(a, k) for a in ARMS for k in (0, 1)],
        "missing/duplicate diagnostics",
    )
    idx = {}
    exact = []
    ids_match = []
    for d in probes:
        need(
            type(d["branch"]) is int
            and toks(d["tokens"])
            and d["shape"] == [STEPS, V]
            and d["dtype"] == "torch.bfloat16"
            and d["finite"] is True
            and re.fullmatch("[a-f0-9]{64}", d["logits_sha256"]),
            "invalid trajectory evidence",
        )
        idx[d["arm"], d["branch"]] = d
    need(idx["cold", 0]["tokens"] == gold, "fresh unedited branch diverges from published 32B")
    for k in (0, 1):
        a, b = [idx[x, k] for x in ARMS]
        ids_match.append(a["tokens"] == b["tokens"])
        exact.append(a["logits_sha256"] == b["logits_sha256"])
        need(not exact[-1] or ids_match[-1], "identical logits but nonidentical argmax")
    need(len(c["pairs"]) == 4, "four crossed order pairs required")
    waves = {a: [] for a in ARMS}
    branch = {a: [] for a in ARMS}
    prep = []
    savings = [[], []]
    branch_orders = [[], []]
    trajectory_stable = True
    for j, row in enumerate(c["pairs"]):
        arms, branches = order(j)
        need(
            type(row["pair"]) is int
            and row["pair"] == j
            and tuple(w["arm"] for w in row["waves"]) == arms,
            "unbalanced arms",
        )
        times = {}
        for w in row["waves"]:
            a = w["arm"]
            ts = w["targets"]
            h = w["prepare_ms"]
            elapsed = w["wave_ms"]
            need(
                timing(elapsed)
                and len(ts) == 2
                and tuple(t["branch"] for t in ts) == branches
                and all(type(t["branch"]) is int for t in ts),
                "bad branch order",
            )
            need(
                (a == "cold" and type(h) is int and h == 0) or (a == "snapshot" and timing(h)),
                "bad prep charge",
            )
            for t in ts:
                need(timing(t["wall_ms"]) and toks(t["ids"]), "invalid target")
                trajectory_stable &= t["ids"] == idx[a, t["branch"]]["tokens"]
                branch[a].append(t["wall_ms"])
            need(elapsed >= h + sum(t["wall_ms"] for t in ts), "wave timer excludes charged work")
            times[a] = elapsed
            waves[a].append(elapsed)
            if a == "snapshot":
                prep.append(h)
        gain = 1 - times["snapshot"] / times["cold"]
        savings[int(arms[0] != "cold")].append(gain)
        branch_orders[int(branches[0] != 0)].append(gain)
    first = {k: median(v) for k, v in zip(ARMS, savings)}
    bo = {k: median(v) for k, v in zip(("base_first", "edited_first"), branch_orders)}
    speed = min(*first.values(), *bo.values()) >= 0.03
    return dict(
        length=n,
        family=i,
        prefix_tokens=p,
        owned_kv_bytes=p * BYTES_PER_POSITION,
        matching_token_trajectories=sum(ids_match),
        byte_identical_full_logit_trajectories=sum(exact),
        trajectories_checked=2,
        all_timed_rows_agree_with_own_diagnostic=trajectory_stable,
        median_wave_ms={a: median(v) for a, v in waves.items()},
        median_prepared_target_ms={a: median(v) for a, v in branch.items()},
        median_snapshot_prepare_ms=median(prep),
        paired_wave_savings_by_first=first,
        paired_wave_savings_by_branch_order=bo,
        passes_timing_only=speed,
        passes_strict_gate=all(exact) and all(ids_match) and trajectory_stable and speed,
    )


def summarize(raw, o):
    need(
        raw["schema"] == "sparkfuse.fork32.v1"
        and raw["oracles"] == [EAGER, RESIDENT]
        and raw["experiment_source_sha256"] == sources(),
        "unbound experiment",
    )
    m = raw["meta"]
    old = o[1]
    need(
        m["all_cuda_bf16"] is True
        and all(
            m[k] == old[k]
            for k in (
                "model",
                "revision",
                "parameters",
                "device",
                "torch",
                "transformers",
                "cuda",
                "accelerate",
                "attention",
                "dtype",
                "layers",
                "model_storage_before",
                "model_storage_after",
                "gate_source_sha256",
                "source_sha256",
            )
        ),
        "not the accepted 32B resident backend",
    )
    need(
        raw["generated"] == STEPS and type(raw["generated"]) is int and len(raw["cases"]) == 4,
        "fixed 8-token budget or cases missing",
    )
    cases = [
        case(c, o, n, i)
        for c, (n, i) in zip(raw["cases"], [(n, i) for n in LENS for i in (1, 2)], strict=True)
    ]
    return dict(
        schema="sparkfuse.fork32.summary.v1",
        cases=cases,
        timed_targets=64,
        timed_snapshot_preparations=16,
        diagnostic_trajectories=16,
        all_strict_gates=all(c["passes_strict_gate"] for c in cases),
        boundary="Batch-one, two related branches per wave, each returns eight host-visible greedy IDs; the snapshot arm pays prefix preparation, allocation, restore and reset. Device-resident BF16 KV storage is not a DRAM traffic counter or multi-tenant SLA.",
    )


def bundle(run, reports):
    from physical_guard import accepted

    a = json.loads((run / "admission.json").read_bytes())
    m = json.loads((run / "memory-guard.json").read_bytes())
    accepted(m)
    need(
        a["admitted"] is True
        and all(
            type(a[k]) is int
            for k in ("available_bytes", "required_bytes", "free_disk_bytes", "required_disk_bytes")
        )
        and a["available_bytes"] >= a["required_bytes"] == 84 * 2**30
        and a["free_disk_bytes"] >= a["required_disk_bytes"] == 2 * 2**30,
        "invalid live admission",
    )
    raw = json.loads((run / "raw.json").read_bytes())
    res = summarize(raw, refs(reports))
    need(
        (json.dumps(res, indent=2, allow_nan=False) + "\n").encode()
        == (run / "timing.json").read_bytes(),
        "timing not reproduced",
    )
    return dict(
        schema="sparkfuse.fork32.bundle.v1",
        measurement=res,
        memory_guard=m,
        evidence_sha256={
            k: sha(run / k)
            for k in ("raw.json", "timing.json", "admission.json", "memory-guard.json")
        },
    )


if __name__ == "__main__":
    import sys

    mode, path, reports = sys.argv[1:]
    need(mode in ("timing", "bundle"), "expected timing or bundle")
    p = Path(path)
    r = Path(reports)
    value = bundle(p, r) if mode == "bundle" else summarize(json.loads(p.read_bytes()), refs(r))
    print(json.dumps(value, indent=2, allow_nan=False))
