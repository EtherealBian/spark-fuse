"""Qualify isolated and end-to-end feedback evidence separately. No model, GPU or dependencies."""

import argparse
import hashlib
import json
from pathlib import Path

import audit_trials
import loop_audit32
import native_audit32
from physical_guard import accepted

GIB = 2**30


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(b):
    return hashlib.sha256(b).hexdigest()


def evidence(directory, name):
    p = directory / name
    need(p.is_file() and not p.is_symlink() and p.stat().st_size <= 16 * 1024**2, "invalid " + name)
    return p.read_bytes()


def decoded(data):
    def unique(pairs):
        ans = {}
        for k, v in pairs:
            need(k not in ans, "duplicate JSON key")
            ans[k] = v
        return ans

    def reject(token):
        raise ValueError("non-finite JSON value " + token)

    return json.loads(data, object_pairs_hook=unique, parse_constant=reject)


def verified_summary(directory, rebuilt):
    blob = evidence(directory, "summary.json")
    want = (json.dumps(rebuilt, indent=2, allow_nan=False) + "\n").encode()
    need(blob == want, "published component summary differs from raw reconstruction")
    return digest(blob)


def guard(admission, report):
    required = {
        "available_bytes",
        "required_bytes",
        "free_disk_bytes",
        "required_disk_bytes",
        "admitted",
    }
    need(
        set(admission) == required
        and admission["admitted"] is True
        and all(type(admission[k]) is int for k in required - {"admitted"}),
        "invalid admission record",
    )
    need(
        admission["available_bytes"] >= admission["required_bytes"] == 84 * GIB
        and admission["free_disk_bytes"] >= admission["required_disk_bytes"] == 2 * GIB,
        "original headroom not available",
    )
    accepted(report)


def component(directory, module, oracles):
    blobs = {n: evidence(directory, n) for n in ("raw.json", "admission.json", "memory-guard.json")}
    raw = decoded(blobs["raw.json"])
    guard(decoded(blobs["admission.json"]), decoded(blobs["memory-guard.json"]))
    s = module.summarize(raw, oracles)
    dig = {k: digest(b) for k, b in blobs.items()}
    dig["summary.json"] = verified_summary(directory, s)
    return (
        raw,
        s,
        dict(
            evidence_sha256=dig,
            logits_checks=s["full_logit_checks"],
            timed_requests=s["timed_requests"],
            passes_matched_request_gate=s["recommend_for_tested_geometry"],
            cases=s["cases"],
        ),
    )


def qualify(micro, loop, native):
    need(
        type(micro) is bool and type(loop) is bool and (native is None or type(native) is bool),
        "invalid gate flag",
    )
    # These two whole-model studies do not share the same paired baseline. Do not multiply speedups.
    return {
        "isolated_feedback_gate": micro,
        "recurrent_vs_queued": "passed" if loop else "did_not_meet_gate",
        "native_vs_torch_graph": "not_evaluated_here"
        if native is None
        else "passed"
        if native
        else "did_not_meet_gate",
        "native_vs_queued": "not_directly_measured",
        "recommend_default_change": False,
    }


def report(reports, micro, loop, native=None):
    oracles = loop_audit32.refs(reports)
    mraw = evidence(micro, "raw.json")
    m = decoded(mraw)
    ms = audit_trials.audit(m)
    mh = verified_summary(micro, ms)
    loop_raw, ls, ld = component(loop, loop_audit32, oracles)
    nflag = None
    need(
        (m["device"], m["torch"], m["cuda"])
        == tuple(loop_raw["meta"][k] for k in ("device", "torch", "cuda")),
        "micro and model runtimes differ",
    )
    parts = {
        "feedback_only": dict(
            raw_sha256=digest(mraw),
            summary_sha256=mh,
            pairs=ms["matched_pairs"],
            units="CUDA event microseconds per isolated feedback operation",
            cases=ms["cases"],
        ),
        "queued_vs_recurrent": ld,
    }
    if native is not None:
        n, ns, nd = component(native, native_audit32, oracles)
        need(
            n["meta"] == loop_raw["meta"] and n["oracle_sha256"] == loop_raw["oracle_sha256"],
            "whole-model runs use different reference or provenance",
        )
        need(
            all(
                n["experiment_source_sha256"][p] == m["source"][p]
                for p in ("vocab_feedback.py", "vocab_feedback.cu")
            ),
            "isolated and whole-model kernels differ",
        )
        nflag = ns["recommend_for_tested_geometry"]
        parts["torch_graph_vs_native_graph"] = nd
    return dict(
        schema="sparkfuse.feedback-stack.v1",
        device=loop_raw["meta"]["device"],
        checkpoint_revision=loop_raw["meta"]["revision"],
        oracle_sha256=loop_raw["oracle_sha256"],
        qualification=qualify(
            ms["eligible_for_model_experiment"], ls["recommend_for_tested_geometry"], nflag
        ),
        components=parts,
        scope="Whole requests start before prefill and end at 32 host-visible IDs. Isolated operator evidence is not a substitute for a full request. Comparisons have different baselines; never multiply their reported speedups. No general serving or default recommendation.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", required=True, type=Path)
    parser.add_argument("--micro", required=True, type=Path)
    parser.add_argument("--loop", required=True, type=Path)
    parser.add_argument(
        "--native",
        type=Path,
        help="Omit only to report the native full model as not evaluated in this bundle",
    )
    a = parser.parse_args()
    print(json.dumps(report(a.reports, a.micro, a.loop, a.native), indent=2, allow_nan=False))
