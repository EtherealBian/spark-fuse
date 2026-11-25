"""Measured schedule dispatch. No dataset names or hard-coded benchmark winners."""

from contextlib import contextmanager
from dataclasses import asdict
from functools import lru_cache
import hashlib, json, random, sqlite3, statistics
from pathlib import Path
import torch
from .ir import lower, Unsupported
from .runtime import compile_export, toolchain
from .codegen import Schedule
from .measure import compare, device_time


@lru_cache(None)
def backend_id():
    h = hashlib.sha256()
    for f in (
        "ir.py",
        "codegen.py",
        "vector.py",
        "compact_mean.py",
        "mean_tree.py",
        "mean_cohort.py",
        "softmax.py",
        "layernorm.py",
        "runtime.py",
        "tuning.py",
        "measure.py",
    ):
        h.update(Path(__file__).with_name(f).read_bytes())
    return h.hexdigest()


def signature(p, device):
    prop = torch.cuda.get_device_properties(device)
    stamp = dict(
        ir=p.explain(),
        backend=backend_id(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        compiler=toolchain()[1],
        chip=prop.name,
        cc=[prop.major, prop.minor],
        sm=prop.multi_processor_count,
    )
    return hashlib.sha256(json.dumps(stamp, sort_keys=True).encode()).hexdigest()


def connect(path):
    db = sqlite3.connect(path, timeout=5)
    db.execute("CREATE TABLE IF NOT EXISTS schedules (key TEXT PRIMARY KEY, record TEXT NOT NULL)")
    return db


def compile_tuned(ep, path, device="cuda"):
    """Cache miss (including compiler version mismatch) uses the safe default."""
    key = signature(lower(ep), device)
    schedule = None
    if Path(path).is_file():
        from contextlib import closing

        with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
            row = db.execute("SELECT record FROM schedules WHERE key=?", (key,)).fetchone()
        if row:
            schedule = Schedule(**json.loads(row[0])["selected"])
    return compile_export(ep, device, schedule)


def compact_eligible(p):
    return (
        p.red is not None
        and p.vals[p.red].op == "mean"
        and p.width > 1
        and all(p.vals[i].shape == (*p.domain[:-1], 1) for i in p.outputs)
    )


def candidates(p):
    from .mean_tree import cohort_eligible, eligible, sensitive

    ordered = (
        [Schedule(t, "mean_tree", cache, 4) for t in (128, 256, 512) for cache in (False, True)]
        if eligible(p)
        else []
    )
    if cohort_eligible(p):
        ordered += [
            Schedule(t, "mean_cohort", cache, 4)
            for t, cache in ((128, False), (128, True), (256, True), (512, True))
        ]
    if sensitive(p):
        return ordered
    choices = list(ordered)
    layouts = ["block", "warp"] if p.red is not None else ["flat", "warp", "block"]
    if p.red is not None and p.vals[p.red].op == "softmax" and 32 <= p.width <= 1024:
        layouts.append("warp_xor")
    caches = (
        [True, False]
        if p.red is not None and any(p.vals[i].shape == p.domain for i in p.outputs)
        else [False]
    )
    for layout in layouts:
        for threads in (32, 64, 128, 256, 512):
            if layout in ("warp", "warp_xor") and threads == 32:
                continue
            for vec in (1, 2, 4, 8):
                for cache in caches:
                    s = Schedule(threads, layout, cache, vec)
                    try:
                        s.validate(p)
                    except Unsupported:
                        continue
                    if cache and (p.width + 31) // (32 if layout == "warp" else threads) > 160:
                        continue
                    choices.append(s)
    if compact_eligible(p):
        for threads in (128, 256, 512, 1024):
            for vec in (1, 2, 4, 8):
                for layout in ("compact_mean", "block") if threads == 1024 else ("compact_mean",):
                    sched = Schedule(threads, layout, False, vec)
                    try:
                        sched.validate(p)
                    except Unsupported:
                        continue
                    choices.append(sched)
    if p.red is not None and p.vals[p.red].op == "layer_norm" and p.width % 4 == 0:
        for cache in caches:
            if not cache or p.width <= 128 * 160:
                choices.append(Schedule(128, "block_welford", cache, 4))
    return choices


@contextmanager
def trial_context(phase, kernel):
    """Annotate the original error without touching the possibly faulted device."""
    try:
        yield
    except Exception as exc:
        exc.add_note(
            f"SparkFuse {phase}; schedule={asdict(kernel.schedule)}; kernel_cache_id={kernel.path.parent.name}"
        )
        raise


@torch.inference_mode()
def tune(
    ep,
    args,
    reference,
    held_args,
    held_reference,
    path,
    *,
    budget=16,
    seed=17,
    exact=(),
    store=True,
):
    base = compile_export(ep, args[0].device)
    p = base.plan
    rng = random.Random(seed)
    candidates_ = candidates(p)
    rng.shuffle(candidates_)
    # Cover dissimilar layouts before filling the shortlist randomly.
    front = [
        Schedule(256, "flat", False, 4),
        Schedule(256, "block", True, 4),
        Schedule(128, "warp", True, 4),
        Schedule(512, "block", False, 4),
        Schedule(64, "warp", False, 1),
    ]
    if compact_eligible(p):
        # Test a conventional 1024-thread reduction as well: wider blocks alone can help.
        pack = max(s.vec for s in candidates_ if s.layout == "compact_mean")
        front.extend(
            [
                Schedule(1024, "block", False, pack),
                Schedule(1024, "compact_mean", False, pack),
                Schedule(512, "compact_mean", False, min(pack, 4)),
            ]
        )
    ranked = [base.schedule] + [s for s in front if s in candidates_] + candidates_
    ordered = list(dict.fromkeys(ranked))[:budget]
    measured = []
    compiled = {}
    invalid = []
    for sched in ordered:
        k = base if sched == base.schedule else compile_export(ep, args[0].device, sched)
        fn = k.prepare(*args)
        try:
            with trial_context("discovery parity", k):
                compare(fn(), reference, exact)
        except AssertionError as e:
            if sched == base.schedule:
                raise
            invalid.append(
                {
                    "schedule": asdict(sched),
                    "kernel_cache_id": k.path.parent.name,
                    "error": str(e)[:800],
                }
            )
            continue
        with trial_context("discovery timing", k):
            timing = device_time(fn, groups=7, unroll=12, replay=4)
        measured.append((timing["median_us"], sched, timing))
        compiled[sched] = k
    measured.sort(key=lambda x: x[0])
    winner = measured[0][1]
    basefn = base.prepare(*held_args)
    winfn = compiled[winner].prepare(*held_args)
    with trial_context("holdout candidate parity", compiled[winner]):
        errors = compare(winfn(), held_reference, exact)
    with trial_context("holdout default parity", base):
        compare(basefn(), held_reference, exact)
    pairs = []
    for i in range(5):
        order = ["default", "candidate"]
        if i % 2:
            order.reverse()
        pair = {}
        for name in order:
            with trial_context(
                f"holdout {i} {name} timing", base if name == "default" else compiled[winner]
            ):
                pair[name] = device_time(
                    basefn if name == "default" else winfn, groups=7, unroll=16, replay=6
                )["median_us"]
        pairs.append(pair)
    ratios = [a["default"] / a["candidate"] for a in pairs]
    admit = (
        winner != base.schedule
        and statistics.median(ratios) >= 1.03
        and sum(r > 1.0 for r in ratios) >= 4
    )
    rec = {
        "key": signature(p, args[0].device),
        "compiler": backend_id(),
        "seed": seed,
        "default": asdict(base.schedule),
        "selected": asdict(winner if admit else base.schedule),
        "candidate": asdict(winner),
        "admitted": admit,
        "heldout_speedup_vs_default": statistics.median(ratios),
        "paired_holdout_us": pairs,
        "heldout_errors": errors,
        "trials": [
            {"schedule": asdict(s), "kernel_cache_id": compiled[s].path.parent.name, "device": tim}
            for us, s, tim in measured
        ],
        "invalid": invalid,
    }
    if store:
        from contextlib import closing

        with closing(connect(path)) as db, db:
            db.execute(
                "INSERT OR REPLACE INTO schedules VALUES (?,?)", (rec["key"], json.dumps(rec))
            )
    return rec
