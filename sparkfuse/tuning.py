"""Measured schedule dispatch. No dataset names or hard-coded benchmark winners."""

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
        "softmax.py",
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


def candidates(p):
    choices = []
    layouts = ["block", "warp"] if p.red is not None else ["flat", "warp", "block"]
    caches = (
        [True, False]
        if p.red is not None and any(p.vals[i].shape == p.domain for i in p.outputs)
        else [False]
    )
    for layout in layouts:
        for threads in (32, 64, 128, 256, 512):
            if layout == "warp" and threads == 32:
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
    return choices


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
    ranked = [base.schedule] + [s for s in front if s in candidates_] + candidates_
    ordered = list(dict.fromkeys(ranked))[:budget]
    measured = []
    compiled = {}
    invalid = []
    for sched in ordered:
        k = base if sched == base.schedule else compile_export(ep, args[0].device, sched)
        fn = k.prepare(*args)
        try:
            compare(fn(), reference, exact)
        except AssertionError as e:
            if sched == base.schedule:
                raise
            invalid.append({"schedule": asdict(sched), "error": str(e)[:800]})
            continue
        timing = device_time(fn, groups=7, unroll=12, replay=4)
        measured.append((timing["median_us"], sched, timing))
        compiled[sched] = k
    measured.sort(key=lambda x: x[0])
    winner = measured[0][1]
    basefn = base.prepare(*held_args)
    winfn = compiled[winner].prepare(*held_args)
    errors = compare(winfn(), held_reference, exact)
    compare(basefn(), held_reference, exact)
    pairs = []
    for i in range(5):
        order = ["default", "candidate"]
        if i % 2:
            order.reverse()
        pair = {}
        for name in order:
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
        "trials": [{"schedule": asdict(s), "device": tim} for us, s, tim in measured],
        "invalid": invalid,
    }
    if store:
        from contextlib import closing

        with closing(connect(path)) as db, db:
            db.execute(
                "INSERT OR REPLACE INTO schedules VALUES (?,?)", (rec["key"], json.dumps(rec))
            )
    return rec
