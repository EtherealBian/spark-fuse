"""Device timings use an unrolled CUDA graph. Host timings include normal dispatch."""

from statistics import median
import time
import torch
from torch.utils._pytree import tree_flatten


def distribution(xs):
    vals = sorted(xs)
    return {
        "median_us": median(vals),
        "p10_us": vals[int(0.1 * (len(vals) - 1))],
        "p90_us": vals[int(0.9 * (len(vals) - 1))],
        "samples_us": xs,
    }


def device_time(fn, groups=9, unroll=16, replay=4):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(unroll):
            ret = fn()
    for _ in range(3):
        g.replay()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(groups)
    ]
    for a, b in events:
        a.record()
        for _ in range(replay):
            g.replay()
        b.record()
    torch.cuda.synchronize()
    return distribution([a.elapsed_time(b) * 1000 / (replay * unroll) for a, b in events])


def host_time(fn, samples=31):
    times = []
    torch.cuda.synchronize()
    for _ in range(samples):
        t = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter_ns() - t) / 1000)
    return distribution(times)


def compare(out, ref, exact=()):
    a, sa = tree_flatten(out)
    b, sb = tree_flatten(ref)
    if sa != sb:
        raise AssertionError("output tree mismatch")
    result = []
    for k, (x, y) in enumerate(zip(a, b)):
        # Near one: <= 2 ULP for low precision; a stricter comparison for exposed residual.
        r, t = {
            torch.bfloat16: (0.012, 0.0002),
            torch.float16: (0.0018, 3e-5),
            torch.float32: (2e-5, 2e-6),
        }[y.dtype]
        if k in exact:
            r = t = 0
        torch.testing.assert_close(x, y, rtol=r, atol=t, equal_nan=True)
        delta = (x.float() - y.float()).abs()
        result.append(
            {
                "max_abs": delta.max().item(),
                "mean_abs": delta.mean().item(),
                "unequal": (x != y).sum().item(),
                "rtol": r,
                "atol": t,
            }
        )
    return result
