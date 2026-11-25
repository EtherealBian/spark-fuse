"""Shuffled paired timing windows on identical data; profile outside both timers."""

import random
from statistics import median

from kernel_trace import kernel_census
from sparkfuse.measure import device_time, host_time


def measure(paths, seed):
    rng = random.Random(seed)
    windows = []
    for _ in range(7):
        order = list(paths)
        rng.shuffle(order)
        w = dict(order=order, device={}, host={})
        for name in order:
            f = paths[name]
            w["device"][name] = device_time(f, groups=7, unroll=3, replay=4)
            w["host"][name] = host_time(f, samples=15)
        windows.append(w)
    return dict(
        windows=windows,
        median_us={
            m: {n: median(median(w[m][n]["samples_us"]) for w in windows) for n in paths}
            for m in ("device", "host")
        },
        profiles={n: kernel_census(f) for n, f in paths.items()},
    )
