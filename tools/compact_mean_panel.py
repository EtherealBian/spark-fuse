"""Paired compact-output reduction ablation, including ordinary 1024-thread blocks."""

import argparse
import json
from pathlib import Path
import random
from statistics import median
import torch
from examples.graphs import case
from provenance import capture
from sparkfuse import Schedule, compile_export
from sparkfuse.measure import compare, device_time, host_time
from sparkfuse.tuning import backend_id


@torch.inference_mode()
def panel(rows, h, dt):
    rng = random.Random(28391)
    seed = 87303
    m, a = case("mean", rows, h, dt, seed=seed)
    ep = torch.export.export(m, a)
    ref = m(*a)
    fam = ("block", "compact_mean")
    grid = [
        Schedule(t, layout, False, v)
        for layout in fam
        for t in (128, 256, 512, 1024)
        for v in ((2, 4) if dt == torch.float32 else (4, 8))
    ]
    rng.shuffle(grid)
    trials = []
    kernels = {}
    for s in grid:
        k = compile_export(ep, schedule=s)
        fn = k.prepare(*a)
        errors = compare(fn(), ref)
        time = device_time(fn, groups=7, unroll=12, replay=4)
        kernels[s] = k
        trials.append(
            dict(schedule=vars(s), kernel_id=k.path.parent.name, discovery=time, errors=errors)
        )
    winners = {
        v: Schedule(
            **min(
                (x for x in trials if x["schedule"]["layout"] == v),
                key=lambda x: x["discovery"]["median_us"],
            )["schedule"]
        )
        for v in fam
    }
    ind = torch.compile(m, backend="inductor", fullgraph=True, dynamic=False)
    results = []
    for idx, storage_offset in enumerate((0, 1)):
        _, args = case("mean", rows, h, dt, seed=seed + 101 + idx)
        x = args[0]
        x = torch.cat([x.new_zeros(1), x.flatten()])[1:].view_as(x) if storage_offset else x
        expected = m(x)
        fns = {
            "eager": lambda: m(x),
            "inductor": lambda: ind(x),
            **{layout: kernels[s].prepare(x) for layout, s in winners.items()},
        }
        normal = {
            "eager": fns["eager"],
            "inductor": fns["inductor"],
            **{layout: (lambda k=kernels[s]: k(x)) for layout, s in winners.items()},
        }
        checks = {}
        rejected = {}
        for key, fn in fns.items():
            try:
                checks[key] = compare(fn(), expected)
            except AssertionError as e:
                if key != "inductor":
                    raise
                rejected[key] = str(e)
        keys = list(checks)
        rounds = []
        for _ in range(7):
            order = keys.copy()
            rng.shuffle(order)
            rounds.append(
                {
                    "order": order,
                    "samples": {
                        key: device_time(fns[key], groups=7, unroll=16, replay=6) for key in order
                    },
                }
            )
        clocks = {key: median(w["samples"][key]["median_us"] for w in rounds) for key in keys}
        ratios = [
            w["samples"]["block"]["median_us"] / w["samples"]["compact_mean"]["median_us"]
            for w in rounds
        ]
        host = {k: host_time(normal[k], 31) for k in keys}
        item = dict(
            offset=storage_offset,
            seed=seed + 101 + idx,
            errors=checks,
            rejected=rejected,
            device_windows=rounds,
            device_us=clocks,
            ordinary_host_sync_us=host,
            paired_block_over_compact=median(ratios),
            compact_wins=sum(v > 1 for v in ratios),
        )
        results.append(item)
        print(
            rows,
            h,
            dt,
            storage_offset,
            {k: round(v, 3) for k, v in clocks.items()},
            item["compact_wins"],
            flush=True,
        )
    return dict(
        rows=rows,
        width=h,
        dtype=str(dt),
        train_seed=seed,
        fx=str(ep.graph),
        trials=trials,
        selected={k: vars(v) for k, v in winners.items()},
        heldout=results,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    d = dict(
        provenance=capture(
            Path(__file__).resolve().parents[1] / ".runtime/compact-mean-panel/sources"
        ),
        compiler=backend_id(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),
        method="Discover on one request, fastest of eight settings separately in each family; both see 128/256/512/1024 threads and the same packs. Freeze both winners. Two independent held-out requests, offset 0 and 1 element; eager, Inductor and both frozen kernels in seven randomized warm CUDA Graph windows each. Normal synchronized host calls measured separately. Kernel winners in this panel are not written to the tuner database.",
        cases=[],
    )
    for r, h, dt in (
        (31, 14336, torch.bfloat16),
        (32, 1536, torch.bfloat16),
        (47, 769, torch.float32),
        (128, 4096, torch.float16),
    ):
        d["cases"].append(panel(r, h, dt))
    t = a.output.with_suffix(".tmp")
    t.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    t.replace(a.output)
