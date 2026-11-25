"""Vary ONLY vector width at fixed scalar-reference layouts. All timings retained, even losses."""

import argparse, json, random
from pathlib import Path
from statistics import median
import torch
from examples.graphs import case
from sparkfuse import Schedule, compile_export
from sparkfuse.measure import compare, device_time
from sparkfuse.tuning import backend_id


@torch.inference_mode()
def run():
    specs = [
        ("scores", 1, 97, "bfloat16", 128, "warp", True),
        ("scores2", 47, 97, "bfloat16", 128, "warp", True),
        ("scores2", 257, 769, "float32", 256, "block", True),
        ("scores", 512, 1024, "bfloat16", 128, "warp", True),
        ("scores2", 128, 4096, "float16", 128, "block", False),
        ("scores", 12, 8192, "bfloat16", 256, "block", True),
    ]
    res = []
    for ix, (name, r, h, dtype, threads, layout, cache) in enumerate(specs):
        dt = getattr(torch, dtype)
        seed = 9400 + ix
        m, a = case(name, r, h, dt, seed)
        _, b = case(name, r, h, dt, seed + 101)
        ep = torch.export.export(m, a)
        ref = m(*b)
        exact = (1,) if name == "scores2" else ()
        timings = {}
        bound = {}
        errors = {}
        schedules = {}
        for vec in [1, 2, 4] if dt == torch.float32 else [1, 2, 4, 8]:
            sc = Schedule(threads, layout, cache, vec)
            schedules[vec] = vars(sc)
            k = compile_export(ep, schedule=sc)
            compare(k(*a), m(*a), exact=exact)
            f = k.prepare(*b)
            errors[vec] = compare(f(), ref, exact=exact)
            bound[vec] = f
            timings[vec] = []
        rng = random.Random(seed)
        orders = []
        for _ in range(7):
            order = list(bound)
            rng.shuffle(order)
            orders.append(order)
            for v in order:
                timings[v].append(device_time(bound[v], groups=7, unroll=16, replay=6)["median_us"])
        ratios = {v: median(t0 / t1 for t0, t1 in zip(timings[1], xs)) for v, xs in timings.items()}
        res.append(
            dict(
                case=name,
                shape=[r, h],
                dtype=dtype,
                seed=seed,
                holdout_seed=seed + 101,
                compiler=backend_id(),
                gpu=torch.cuda.get_device_name(),
                schedules=schedules,
                orders=orders,
                window_medians_us=timings,
                paired_scalar_over_candidate=ratios,
                errors=errors,
            )
        )
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(), indent=2, allow_nan=False) + "\n")
