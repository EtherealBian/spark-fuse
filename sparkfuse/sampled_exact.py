"""Tune one dispatch key across all its call sites under a sampled exact-output constraint."""

from dataclasses import asdict
import random
from statistics import median

import torch

from .measure import compare, device_time
from .partition import ordered_collectives
from .runtime import compile_export
from .tuning import backend_id, candidates, signature, trial_context


def _shortlist(plan, default, budget, rng):
    pool = candidates(plan)
    rng.shuffle(pool)
    anchor = ordered_collectives(plan)
    family = [s for s in pool if anchor is not None and s.layout == anchor.layout]
    outside = [s for s in pool if s not in family]
    # Interleave families: numerical equivalence is measured, not inferred from the layout's name.
    order = [anchor, default]
    for i in range(max(len(family), len(outside))):
        if i < len(family):
            order.append(family[i])
        if i < len(outside):
            order.append(outside[i])
    return list(dict.fromkeys(s for s in order if s is not None))[:budget]


def _sequence(calls):
    def run():
        for call in calls:
            call()

    return run


@torch.inference_mode()
def tune_shared_exact(ep, sites, *, budget, seed):
    """sites maps each invocation name to three (actual inputs, raw ATen oracle outputs) pairs.

    The third reference is not consulted until selection is final. Times represent one execution of
    the independent region calls in sequence, not the whole model or Python dispatch latency.
    """
    if budget < 1 or not sites or any(len(items) != 3 for items in sites.values()):
        raise ValueError("need a positive budget and three requests at every call site")
    device = next(iter(sites.values()))[0][0][0].device
    default = compile_export(ep, device)
    p = default.plan
    rng = random.Random(seed)
    order = _shortlist(p, default.schedule, budget, rng)
    exact = tuple(range(len(p.outputs)))
    gates, trials, usable = [], [], {}
    for s in order:
        k = default if s == default.schedule else compile_export(ep, device, s)
        train_calls = []
        phase, name = None, None
        try:
            for idx, phase in enumerate(("train", "holdout")):
                for name, requests in sites.items():
                    args, ref = requests[idx]
                    launch = k.prepare(*args)
                    with trial_context(f"sampled exact {phase} at {name}", k):
                        compare(launch(), ref, exact=exact)
                    if idx == 0:
                        train_calls.append(launch)
        except AssertionError as e:
            gates.append(
                dict(
                    schedule=asdict(s),
                    kernel_cache_id=k.path.parent.name,
                    phase=phase,
                    site=name,
                    error=str(e)[:1200],
                )
            )
            continue
        train = _sequence(train_calls)
        with trial_context("grouped discovery timing", k):
            tm = device_time(train, groups=7, unroll=12, replay=4)
        trials.append(dict(schedule=asdict(s), kernel_cache_id=k.path.parent.name, device=tm))
        # Do not retain preallocated outputs for every measured candidate.
        usable[s] = (k, tm["median_us"])
    if not usable:
        raise AssertionError(
            f"no candidate passed sampled exact parity for {list(sites)}: {gates[:1]}"
        )
    anchor = next(
        iter(usable)
    )  # First eligible, not necessarily the unconditional compiler default.
    winner = min(usable, key=lambda s: usable[s][1])
    pairs, ratios = [], []
    if winner != anchor:
        held = {
            s: _sequence([usable[s][0].prepare(*requests[1][0]) for requests in sites.values()])
            for s in (anchor, winner)
        }
        for i in range(5):
            pair = {}
            labels = [("anchor", anchor), ("candidate", winner)]
            for label, s in reversed(labels) if i % 2 else labels:
                k, launch = usable[s][0], held[s]
                with trial_context(f"grouped holdout {i} {label}", k):
                    pair[label] = device_time(launch, groups=7, unroll=16, replay=6)["median_us"]
            pairs.append(pair)
        ratios = [w["anchor"] / w["candidate"] for w in pairs]
    admitted = bool(ratios and median(ratios) >= 1.03 and sum(x > 1.0 for x in ratios) >= 4)
    selected = winner if admitted else anchor
    k = usable[selected][0]
    audits = {}
    for name, requests in sites.items():
        args, ref = requests[2]
        with trial_context(f"sampled exact independent audit at {name}", k):
            audits[name] = compare(k(*args), ref, exact=exact)
    return dict(
        key=signature(p, device),
        compiler=backend_id(),
        seed=seed,
        precision_gate="all boundary outputs exact on these requests",
        sites=list(sites),
        selected=asdict(selected),
        default=asdict(default.schedule),
        anchor=asdict(anchor),
        candidate=asdict(winner),
        admitted=admitted,
        heldout_speedup_vs_anchor=median(ratios) if ratios else None,
        paired_holdout_us=pairs,
        trials=trials,
        invalid=gates,
        third_input_errors=audits,
    )
