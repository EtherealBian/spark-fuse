"""Re-map the same narrow FP32 mean tree; hold out inputs and distinguish fixed vs tuned layouts."""

import argparse
import json
from pathlib import Path
import random
from statistics import median
import torch
from examples.graphs import case
from mean_tree_audit import Boundaries, compare_bits
from provenance import capture
from sparkfuse import Schedule, compile_export
from sparkfuse.measure import compare, device_time, host_time
from sparkfuse.tuning import backend_id, candidates


def shift(x, n):
    return torch.cat([x.new_zeros(n), x.flatten()])[n:].view_as(x) if n else x


@torch.inference_mode()
def run():
    rng = random.Random(46411)
    source = capture(Path(__file__).resolve().parents[1] / ".runtime/mean-cohort-audit/sources")
    result = dict(
        provenance=source,
        compiler=backend_id(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),
        bench="Static single-output decomposed residual RMSNorm; seven interleaved windows per held-out request. Includes independent discovery for each family and fixed defaults.",
        cases=[],
    )
    for i, (n, h, dt) in enumerate(
        [
            (17, 8160, torch.float16),
            (32, 1536, torch.bfloat16),
            (128, 4096, torch.bfloat16),
            (31, 1024, torch.float16),
            (63, 4100, torch.bfloat16),
        ]
    ):
        seed = 97003 + 23 * i
        m, a = case("decomp", n, h, dt, seed)
        ref = m(*a)
        ep = torch.export.export(m, a)
        emit = compile_export(ep)
        grid = [s for s in candidates(emit.plan) if s.layout in ("mean_tree", "mean_cohort")]
        rng.shuffle(grid)
        builds = {}
        trials = []
        for s in grid:
            k = compile_export(ep, schedule=s)
            builds[s] = k
            fn = k.prepare(*a)
            assert compare_bits([fn()], [ref])[0]["bit_mismatches"] == 0
            trials.append(
                dict(
                    schedule=vars(s),
                    kernel=k.path.parent.name,
                    device=device_time(fn, groups=7, unroll=8, replay=4),
                )
            )
        families = {
            f: Schedule(
                **min(
                    (x for x in trials if x["schedule"]["layout"] == f),
                    key=lambda x: x["device"]["median_us"],
                )["schedule"]
            )
            for f in ("mean_tree", "mean_cohort")
        }
        schedules = {
            "tree_default": Schedule(512, "mean_tree", False, 4),
            "cohort_default": emit.schedule,
            "tree_fastest": families["mean_tree"],
            "cohort_fastest": families["mean_cohort"],
        }
        kernels = {k: builds[s] for k, s in schedules.items()}
        tapped = Boundaries()
        tep = torch.export.export(tapped, a)
        observers = {s: compile_export(tep, schedule=s) for s in set(schedules.values())}
        ind = torch.compile(m, backend="inductor", fullgraph=True, dynamic=False)
        r = dict(
            rows=n,
            width=h,
            dtype=str(dt),
            seed=seed,
            trials=trials,
            schedules={k: vars(s) for k, s in schedules.items()},
            heldout=[],
        )
        for offsets in [(0, 0, 0), (1, 2, 3)]:
            # Same values at both alignments, never used in the layout search.
            _, v = case("decomp", n, h, dt, seed + 101)
            v = tuple(shift(x, o) for x, o in zip(v, offsets))
            ref = m(*v)
            taps = tapped(*v)
            bits = {key: compare_bits(observers[s](*v), taps) for key, s in schedules.items()}
            assert all(b["bit_mismatches"] == 0 for stats in bits.values() for b in stats)
            users = {
                "eager": lambda: m(*v),
                "inductor": lambda: ind(*v),
                **{key: (lambda k=k: k(*v)) for key, k in kernels.items()},
            }
            dev = {**users, **{key: k.prepare(*v) for key, k in kernels.items()}}
            checks = {}
            for key, fn in users.items():
                y = fn()
                quality = compare_bits([y], [ref])
                if key in schedules:
                    assert quality[0]["bit_mismatches"] == 0
                try:
                    checks[key] = {"errors": compare(y, ref), "passed": True, "bits": quality}
                except AssertionError as e:
                    if key != "inductor":
                        raise
                    checks[key] = {"passed": False, "diagnostic": str(e), "bits": quality}
            keys = [k for k, q in checks.items() if q["passed"]]
            windows = []
            for _ in range(7):
                order = rng.sample(keys, len(keys))
                windows.append(
                    {
                        "order": order,
                        "device": {
                            k: device_time(dev[k], groups=7, unroll=8, replay=4) for k in order
                        },
                    }
                )
            med = {k: median(x["device"][k]["median_us"] for x in windows) for k in keys}
            r["heldout"].append(
                dict(
                    seed=seed + 101,
                    offsets=offsets,
                    tapped=bits,
                    checks=checks,
                    windows=windows,
                    median_us=med,
                    host_sync={k: host_time(users[k], 31) for k in keys},
                )
            )
            print(n, h, offsets, med, flush=True)
        result["cases"].append(r)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    data = run()
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
