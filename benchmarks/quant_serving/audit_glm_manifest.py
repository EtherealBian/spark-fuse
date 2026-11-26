"""Compare decisions on the same manifest messages. Never compare token IDs between tokenizers."""

import argparse, json
from pathlib import Path
from audit_glm355 import BASE, attest, accepted, gate, read_inputs, rebuild, require, sha
from glm_manifest import audit, make, tasks, label

QFILES = {
    "summary.json": "aa4c302ee6beb43fcaadb3bbddf8ab658d0ce4bd92e6fcf3f5dc4c7145045612",
    "rows.json": "9f5b479e9d541d94fef05bdd2cbdbd4c8f4a54bcc76ddcbdbe4fb5a1b51c405b",
    "runtime.json": "2b396205c9d65dfc5ac469b29a2f863401289e0b4b57a9baa36cebc93046affd",
}


def compare(q, g):
    keys = {label(*v) for v in tasks()}
    require(
        set(q["cases"]) == set(g["cases"]) == keys and q["pairs"] == g["pairs"] == 8,
        "not the same manifest cases",
    )
    out = {}
    for arm in ["cold", "reuse"]:
        joint = {
            a: {b: 0 for b in ["correct", "stale", "other"]} for a in ["correct", "stale", "other"]
        }
        details = []
        for n, p, v in tasks():
            k = label(n, p, v)
            x = q["cases"][k]
            y = g["cases"][k]
            _, target = make(n, p, v)
            _, old = make(n, p, 1 - v)
            require(
                all((t["expected"], t["stale"]) == (target, old) for t in (x, y)),
                "manifest answer identities differ",
            )
            a = x["arms"][arm]
            b = y["arms"][arm]
            joint[a["verdict"]][b["verdict"]] += 1
            details.append(
                {
                    "case": k,
                    "qwen_verdict": a["verdict"],
                    "glm_verdict": b["verdict"],
                    "qwen_eos": a["natural_eos"],
                    "glm_eos": b["natural_eos"],
                }
            )
        out[arm] = dict(qwen_row_glm_column=joint, paired_verdicts=details)
    return out


def evaluate(lock, d, prior, qref):
    def rd(dir, n):
        p = dir / n
        require(
            p.is_file() and not p.is_symlink() and p.stat().st_size <= 16 * 2**20, "invalid receipt"
        )
        return p.read_bytes()

    needq = {k: rd(qref, k) for k in QFILES}
    require(all(sha(v) == QFILES[k] for k, v in needq.items()), "unrecognized Qwen reference")
    qr = json.loads(needq["summary.json"])
    qraw = json.loads(
        needq["rows.json"]
    )  # Audit with the Qwen vocabulary. The original 235B raw and summary are pinned separately.
    from revision_canary import audit as qwen_audit

    q = qwen_audit(qraw, json.loads(needq["runtime.json"])["eog_token_ids"])
    require(q == qr["serving"], "Qwen semantic results do not reproduce")
    shas = {}

    def read(n):
        b = rd(d, n)
        shas[n] = sha(b)
        return json.loads(b)

    gb = rebuild(lock, prior, Path(BASE))
    require(
        gb == json.loads(rd(prior, "summary.json")) and gb["tested_repeatability_gate"],
        "large-model baseline not qualified",
    )
    lc, b = read_inputs(lock, d / "header-budget.json")
    read("header-budget.json")
    cp = read("checkpoint.json")
    rt = read("runtime.json")
    sig = attest(lock, cp, rt, 94, 1, 128)
    require(
        sig == gb["runtime"] and rd(d, "checkpoint.json") == rd(prior, "checkpoint.json"),
        "changed binary or weights",
    )
    accepted(read("memory-guard.json"))
    gate("launch", b, read("admission.json"))
    gate("transfer", b, read("admission-transfer.json"))
    require(
        rd(d, "admission-transfer.json") == rd(prior, "admission-transfer.json"), "transfer changed"
    )
    raw = read("rows.json")
    g = audit(raw, sig["eog_token_ids"])
    require(g == read("timing.json"), "saved semantic scorer drift")
    for n, p, v in tasks():
        k = label(n, p, v)
        msg, _ = make(n, p, v)
        for corpus in (qraw, raw):
            require(
                all(corpus["inputs"][k]["rendered"].count(m["content"]) == 1 for m in msg),
                "native template lost shared message body",
            )
    return dict(
        schema="sparkfuse.cross-manifest.v1",
        glm_reference_sha256=sha(rd(prior, "summary.json")),
        qwen_reference_sha256=QFILES["summary.json"],
        glm_evidence_sha256=shas,
        glm_serving=g,
        matched_tasks=compare(q, g),
        scope="Eight paired synthetic manifest revisions. Same messages and strict whole-answer scoring, native tokenizer/template per model, natural EOS or 12 generated tokens. No stripping reasoning or punctuation; capped or unformatted answers may score other. Do not compare cross-tokenizer IDs, general quality, model scale or precision in isolation.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    [p.add_argument(k, type=Path) for k in ("lock", "directory", "baseline", "qwen")]
    a = p.parse_args()
    print(json.dumps(evaluate(a.lock, a.directory, a.baseline, a.qwen), indent=2, allow_nan=False))
