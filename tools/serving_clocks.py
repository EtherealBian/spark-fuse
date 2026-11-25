"""Shared single-device setup and ordinary fixed/rotating/queued request clocks."""

import random
from statistics import median
import time

import torch


def build(f):
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter_ns()
    result = f()
    torch.cuda.synchronize()
    return result, dict(
        wall_ms=(time.perf_counter_ns() - start) / 1e6,
        retained_allocator_delta_bytes=torch.cuda.memory_allocated() - base,
        peak_allocator_increase_bytes=torch.cuda.max_memory_allocated() - base,
    )


def clocks(paths, requests, seed):
    """Fixed-address and rotating latency; eight distinct outstanding results in queued mode."""
    rng = random.Random(seed)
    stream = torch.cuda.current_stream()
    windows = []
    for w in range(7):
        order = sorted(paths)
        rng.shuffle(order)
        ws = {}
        for name in order:
            f = paths[name]
            samples = {}
            for mode in ("rotating", "fixed") if w % 2 else ("fixed", "rotating"):
                sync = []
                submit = []
                for j in range(16):
                    req = requests[(j + w) % 8 if mode == "rotating" else 0]
                    stream.synchronize()
                    start = time.perf_counter_ns()
                    out = f(*req)
                    sent = time.perf_counter_ns()
                    stream.synchronize()
                    end = time.perf_counter_ns()
                    sync.append((end - start) / 1000)
                    submit.append((sent - start) / 1000)
                    del out
                samples[mode + "_sync_us"] = sync
                samples[mode + "_submit_us"] = submit
            queued = []
            for k in range(5):
                rs = [requests[(k + w + j) % 8] for j in range(8)]
                stream.synchronize()
                start = time.perf_counter_ns()
                out = [f(*r) for r in rs]
                stream.synchronize()
                end = time.perf_counter_ns()
                queued.append((end - start) / 8000)
                del out
            samples["queued8_us_per_forward"] = queued
            ws[name] = samples
        windows.append({"order": order, "samples": ws})
    metrics = list(samples)
    summary = {
        n: {k: median(median(w["samples"][n][k]) for w in windows) for k in metrics} for n in paths
    }
    return dict(seed=seed, windows=windows, median_us=summary)
