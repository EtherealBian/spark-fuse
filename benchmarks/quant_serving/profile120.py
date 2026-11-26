"""Choose a startup microbatch for a specified, measured, serial request mix. Never reconfigure a live server."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from audit_bundle import ROOT, attest, bundle
from moe_blocks import audit

ORDERS = (("128_then_512", 0, 1), ("512_then_128", 3, 2))


def strict_mix(pairs):
    mix = {}
    for key, value in pairs:
        if key in mix:
            raise ValueError("duplicate workload key")
        mix[key] = value
    return mix


def choose(blocks, mix=None, *, min_gain=0.03, max_regression=0.02):
    """Pure policy. Production callers use from_evidence() to bind the live runtime receipts."""
    verdict = audit(blocks)
    for fraction in (min_gain, max_regression):
        if (
            type(fraction) not in (int, float)
            or not math.isfinite(fraction)
            or not 0 <= fraction < 1
        ):
            raise ValueError("policy thresholds must be fractions in [0,1)")
    rows = [{r["input_hash"]: r for r in b["rows"]} for b in blocks]
    mix = {k: 1 for k in rows[0]} if mix is None else mix
    if (
        not isinstance(mix, dict)
        or not mix
        or any(
            not isinstance(k, str)
            or len(k) != 64
            or any(ch not in "0123456789abcdef" for ch in k)
            or type(v) is not int
            or not 1 <= v <= 1000000
            for k, v in mix.items()
        )
    ):
        raise ValueError("expected measured input SHA-256 to integer request-count mapping")
    mix = dict(sorted(mix.items()))
    budget = sum(mix.values())
    unknown = sorted(set(mix) - set(rows[0]))
    result = dict(
        schema=1,
        baseline_ubatch=128,
        candidate_ubatch=512,
        selected_ubatch=128,
        min_weighted_wall_gain=min_gain,
        max_single_request_wall_regression=max_regression,
        request_counts=mix,
        total_requests=budget,
        scope="Offline startup choice for this measured prompt-ID mix, pinned backend, warm single slot, cold prompt KV, 24 fixed output tokens. Weighted totals are projections, not traffic replay or a quality/SLO guarantee.",
        orders={},
        gates={},
    )
    if unknown:
        return dict(
            result,
            decision="keep",
            reason="unmeasured prompt: no extrapolation from input length",
            unknown_sha256=unknown,
        )
    stable = {k: all(row[k]["ids"] == rows[0][k]["ids"] for row in rows[1:]) for k in mix}
    all_safe = True
    enough = True
    for label, ai, bi in ORDERS:
        a, b = rows[ai], rows[bi]
        wall_a = sum(mix[k] * a[k]["wall_ms"] for k in mix)
        wall_b = sum(mix[k] * b[k]["wall_ms"] for k in mix)
        fraction = (wall_a - wall_b) / wall_a
        offenders = [k for k in mix if b[k]["wall_ms"] > a[k]["wall_ms"] * (1 + max_regression)]
        all_safe &= not offenders
        enough &= wall_b < wall_a and fraction >= min_gain
        result["orders"][label] = dict(
            projected_128_ms=wall_a,
            projected_512_ms=wall_b,
            projected_fraction_saved=fraction,
            individually_over_budget_sha256=offenders,
        )
    result["gates"] = dict(
        all_selected_prompts_have_identical_tokens_in_all_four_blocks=all(stable.values()),
        positive_gain_and_margin_in_both_orders=bool(enough),
        each_selected_prompt_within_regression_budget_in_both_orders=bool(all_safe),
    )
    result["different_token_sha256"] = [k for k in mix if not stable[k]]
    accept = all(result["gates"].values())
    result.update(
        selected_ubatch=512 if accept else 128,
        decision="select" if accept else "keep",
        reason="all gates passed" if accept else "one or more measured gates failed",
        all_measured_prompts_exact=verdict["all_tokens_exact"],
    )
    return result


def measured_cohorts(blocks, **thresholds):
    """Separate deployments for the actual two recorded short / long requests, not general length-based routing."""
    groups = {}
    for row in blocks[0]["rows"]:
        groups.setdefault(str(row["size"]), {})[row["input_hash"]] = 1
    return {size: choose(blocks, mix, **thresholds) for size, mix in groups.items()}


def verify_target(signature, target):
    """Require separate full-weight and live-process receipts, not a model name or an editable label."""
    provenance = {}

    def read(name):
        raw = (target / name).read_bytes()
        provenance[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    cp = read("checkpoint.json")
    rt = read("runtime.json")
    ub = rt["context"]["n_ubatch"]
    if type(ub) is not int or ub not in (128, 512):
        raise ValueError("unmeasured target microbatch")
    if attest(ROOT / "gpt-oss-120b-mxfp4.lock.json", cp, rt, 37, 1, ub) != signature:
        raise ValueError(
            "target runtime differs from the measured deployment; do not apply this profile"
        )
    return dict(verified_current_ubatch=ub, evidence_sha256=provenance)


def from_evidence(directory, mix=None, *, target=None, **thresholds):
    base = bundle("120b", directory)
    blocks = []
    for i in range(4):
        name = f"block-{i}.json"
        data = (directory / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != base["evidence_sha256"][name]:
            raise ValueError("benchmark evidence changed after verification")
        blocks.append(json.loads(data))
    decision = choose(blocks, mix, **thresholds)
    decision.update(
        model=base["model"],
        revision=base["revision"],
        runtime=base["runtime"],
        evidence_sha256=base["evidence_sha256"],
    )
    if mix is None:
        decision["separate_known_prompt_cohorts"] = measured_cohorts(blocks, **thresholds)
    decision["target_receipts"] = (
        verify_target(base["runtime"], target) if target is not None else None
    )
    return decision


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument(
        "--mix",
        type=Path,
        help="JSON mapping of exact input-token SHA-256 to workload counts (default: all four equally)",
    )
    p.add_argument(
        "--target",
        type=Path,
        help="Optional target checkpoint.json + runtime.json directory. Refuse mismatched weights, launch geometry or runtime signature.",
    )
    p.add_argument("--min-gain", type=float, default=0.03)
    p.add_argument("--max-case-regression", type=float, default=0.02)
    a = p.parse_args()
    mix = None if a.mix is None else json.loads(a.mix.read_bytes(), object_pairs_hook=strict_mix)
    print(
        json.dumps(
            from_evidence(
                a.directory,
                mix,
                target=a.target,
                min_gain=a.min_gain,
                max_regression=a.max_case_regression,
            ),
            indent=2,
            allow_nan=False,
        )
    )
