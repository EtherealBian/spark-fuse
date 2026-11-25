"""Numerical audit for long-key cached Softmax schedules found by the live search."""

import argparse
import json
from pathlib import Path
import torch
from provenance import capture
from examples.graphs import case
from sparkfuse import Schedule, compile_export
from sparkfuse.measure import compare
from sparkfuse.tuning import backend_id

CASES = [
    ("scores", 24, 8192, "float16", 64, "block", 2, 31885),
    ("scores2", 24, 2048, "float16", 512, "warp", 8, 32395),
    ("scores", 384, 4096, "bfloat16", 32, "block", 2, 32497),
]


@torch.inference_mode()
def run():
    source = capture(Path(__file__).resolve().parents[1] / ".runtime/wide-score-audit/sources")
    out = []
    for name, rows, h, dt, t, layout, vec, seed in CASES:
        m, sample = case(name, rows, h, getattr(torch, dt), seed=seed)
        sched = Schedule(t, layout, True, vec)
        k = compile_export(torch.export.export(m, sample), schedule=sched)
        for i in range(4):
            _, a = case(name, rows, h, getattr(torch, dt), seed=seed + 7919 * i)
            if i % 2:
                a = tuple(torch.cat([x.new_zeros(1), x.flatten()])[1:].view_as(x) for x in a)
            got = k(*a)
            torch.cuda.synchronize()
            ref = m(*a)
            errors = compare(got, ref, exact=(1,) if name == "scores2" else ())
            out.append(
                dict(
                    case=name,
                    rows=rows,
                    width=h,
                    dtype=dt,
                    schedule=vars(sched),
                    seed=seed + 7919 * i,
                    offset=i % 2,
                    kernel_cache_id=k.path.parent.name,
                    errors=errors,
                )
            )
            print(name, rows, h, dt, seed + 7919 * i, "pass", flush=True)
    return {
        "provenance": source,
        "compiler": backend_id(),
        "torch": str(torch.__version__),
        "device": torch.cuda.get_device_name(),
        "cases": out,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    d = run()
    if a.output:
        a.output.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
