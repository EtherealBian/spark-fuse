"""Independently re-read BF16 bytes. Report first changed step/bank, never infer a kernel cause."""

import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np

ARMS = ("whole", "inplace", "restored")
RAW = "ac68452353a75ac21cf03ae7d864153cdb77d0bf578658a8d426d9e4626c0c48"
SUMMARY = "3890e2f7fc3cab824ea738ac94cd4fb3f1441fe3951d79fb0cdc5c96cb71784c"


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def fileid(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ident(r):
    return {"raw": fileid(r / "raw.json"), "summary": fileid(r / "summary.json")}


def source_panel(r):
    need(ident(r) == dict(raw=RAW, summary=SUMMARY), "prior panel is not pinned")
    p = {}
    d = json.loads((r / "raw.json").read_bytes())
    for c in d["cases"]:
        for k, ids in enumerate(c["inputs"]):
            a, b = [z for z in c["diagnostics"] if z["branch"] == k]
            p[c["length"], c["family"], k] = dict(ids=ids, whole=a, restored=b)
    return p


def filebase(key, rep, arm):
    return "-".join(map(str, (*key, rep, arm)))


def save(p, v):
    q = p.with_name(p.name + ".partial")
    q.write_text(json.dumps(v, indent=2, allow_nan=False) + "\n")
    os.replace(q, p)


def asfloat(x):
    return (x.astype(np.uint32) << 16).view(np.float32)


def read(root, name, meta, shape):
    b = (root / name).read_bytes()
    need(
        meta == dict(shape=list(shape), bytes=len(b), sha256=hashlib.sha256(b).hexdigest())
        and len(b) == np.prod(shape) * 2,
        "incorrect binary extent/digest",
    )
    x = np.frombuffer(b, dtype="<u2").reshape(shape)
    need(np.isfinite(asfloat(x)).all(), "nonfinite binary")
    return x


def compare(a, b, kv_a, kv_b):
    af, bf = asfloat(a), asfloat(b)
    cnt = np.count_nonzero(a != b, axis=1)
    ka = (kv_a != kv_b).reshape(kv_a.shape[:2] + (-1,)).sum(-1)
    diff = np.flatnonzero(ka.sum(1))
    tok = np.argmax(af, axis=1)
    tok_b = np.argmax(bf, axis=1)
    div = np.flatnonzero(tok != tok_b)
    j = int(div[0]) if len(div) else None
    gap = None
    if j is not None:
        x, y = int(tok[j]), int(tok_b[j])
        gap = dict(
            reference_id=x,
            candidate_id=y,
            reference_margin_between_two=float(af[j, x] - af[j, y]),
            candidate_margin_between_two=float(bf[j, y] - bf[j, x]),
        )
    return dict(
        different_logit_values_by_step=cnt.tolist(),
        max_abs_by_step=np.max(np.abs(af - bf), axis=1).tolist(),
        first_different_greedy_step=None if j is None else j + 1,
        at_first_changed_argmax=gap,
        changed_sampled_kv_scalars=int(ka.sum()),
        first_changed_sampled_kv_bank=None
        if not len(diff)
        else dict(
            layer_0_based=int(diff[0]) // 2,
            bank=["key", "value"][int(diff[0]) % 2],
            counts_at_last_prefix_first_suffix_last_input=ka[diff[0]].tolist(),
        ),
    )


def audit(out, ref):
    from physical_guard import accepted

    accepted(json.loads((out / "memory-guard.json").read_bytes()))
    a = json.loads((out / "admission.json").read_bytes())
    need(
        a["admitted"] is True
        and a["available_bytes"] >= a["required_bytes"] == 84 * 2**30
        and a["free_disk_bytes"] >= a["required_disk_bytes"] == 2 * 2**30,
        "bad admission",
    )
    p = source_panel(ref)
    v = json.loads((out / "index.json").read_bytes())
    need(
        v["schema"] == "sparkfuse.fork.attribution.v1"
        and v["reference"] == ident(ref)
        and v["sources"]
        == {
            k: fileid(Path(__file__).with_name(k))
            for k in ["attribution_review.py", "attribute.py"]
        },
        "metadata mismatch",
    )
    old = json.loads((ref / "raw.json").read_bytes())["meta"]
    need(
        set(v["meta"]) == set("model revision parameters device torch cuda transformers".split())
        and all(v["meta"][k] == old[k] for k in v["meta"]),
        "wrong model/runtime",
    )
    expected = []
    for n in (128, 2048):
        cells = [k for k in p if k[0] == n]
        for rep in (0, 1):
            for k in cells if not rep else cells[::-1]:
                expected.extend((k, rep, arm) for arm in (ARMS if not rep else ARMS[::-1]))
    need(
        [(tuple(x["cell"]), x["repeat"], x["arm"]) for x in v["records"]] == expected,
        "incomplete or unbalanced counterfactuals",
    )
    wanted = {
        filebase(k, i, arm) + "-" + kind + ".bf16"
        for k, i, arm in expected
        for kind in ("logits", "kv")
    }
    need({q.name for q in out.glob("*.bf16")} == wanted, "binary membership differs")
    records = {t: row for t, row in zip(expected, v["records"])}
    results = []
    for k, original in p.items():
        rounds = []
        prior = {}
        stable = True
        for rep in (0, 1):
            arrays = {}
            tokens = {}
            for arm in ARMS:
                row = records[k, rep, arm]
                base = filebase(k, rep, arm)
                logit_rows = read(out, base + "-logits.bf16", row["logits"], (8, 152064))
                kv = read(out, base + "-kv.bf16", row["kv"], (128, 3, 8, 128))
                arrays[arm] = (logit_rows, kv)
                tokens[arm] = asfloat(logit_rows).argmax(1).tolist()
                dig = (row["logits"]["sha256"], row["kv"]["sha256"])
                stable &= rep == 0 or prior[arm] == dig
                prior[arm] = dig
            cold, snap = original["whole"], original["restored"]
            changed = next(
                (i for i, (a, b) in enumerate(zip(cold["tokens"], snap["tokens"])) if a != b), 7
            )
            lc, kc = arrays["whole"]
            li, ki = arrays["inplace"]
            lr, kr = arrays["restored"]
            s = dict(
                whole_reproduced=prior["whole"][0] == cold["logits_sha256"]
                and tokens["whole"] == cold["tokens"],
                restored_free_path_reproduced_through_common_input=tokens["restored"][: changed + 1]
                == snap["tokens"][: changed + 1],
                restored_digest_when_all_forced_inputs_match=(
                    prior["restored"][0] == snap["logits_sha256"]
                )
                if cold["tokens"][:-1] == snap["tokens"][:-1]
                else None,
                whole_vs_inplace=compare(lc, li, kc, ki),
                inplace_vs_restored=compare(li, lr, ki, kr),
            )
            rounds.append(s)
        results.append(dict(cell=list(k), cross_order_bitwise_repeat=bool(stable), rounds=rounds))
    return dict(
        schema="sparkfuse.fork.attribution.summary.v1",
        records=len(records),
        index_sha256=fileid(out / "index.json"),
        reference=ident(ref),
        cells=results,
        scope="Forced whole-prefill greedy token at every decode step, not free generation. All vocabulary entries compared. Physical KV checks only three selected positions per bank at end of prefill. Diagnostic I/O excluded from any performance claim; observations do not identify a particular GEMM or attention kernel.",
    )


if __name__ == "__main__":
    print(json.dumps(audit(*map(Path, sys.argv[1:])), indent=2, allow_nan=False))
