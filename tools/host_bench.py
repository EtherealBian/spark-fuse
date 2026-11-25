"""Compare host frontends with identical CUDA code, matched inputs and fresh outputs."""

import argparse, json, time
from pathlib import Path
import torch
from examples.graphs import case
from sparkfuse import compile_export, Schedule
from sparkfuse.measure import compare, device_time, host_time, distribution
from sparkfuse.native import native_dispatch


@torch.inference_mode()
def run(name, rows, h, dtype):
    m, a = case(name, rows, h, dtype, seed=284)
    ep = torch.export.export(m, a)
    p = compile_export(
        ep,
        schedule=Schedule(512, "block", False, 4)
        if name == "resnorm2"
        else Schedule(256, "flat", False, 4),
    )
    native = native_dispatch(p)
    ind = torch.compile(m, fullgraph=True, dynamic=False)
    ref = m(*a)
    fs = {
        "eager": lambda: m(*a),
        "inductor": lambda: ind(*a),
        "python_reference": lambda: p(*a),
        "native_dispatch": lambda: native(*a),
    }
    rec = {}
    for key, fn in fs.items():
        e = compare(fn(), ref, exact=(1,) if name == "resnorm2" else ())
        rec[key] = {"errors": e, "device": device_time(fn), "host_windows": []}
    # Fixed sequence plus reversal; all four backends allocate normal, non-shared outputs.
    for i in range(6):
        labels = list(fs)
        if i % 2:
            labels.reverse()
        for key in labels:
            rec[key]["host_windows"].append(host_time(fs[key], 31)["median_us"])
    for v in rec.values():
        v["host_window_medians"] = distribution(v["host_windows"])
    return {
        "shape": [rows, h],
        "case": name,
        "dtype": str(dtype),
        "target": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "schedule": vars(p.schedule),
        "backends": rec,
        "same_kernel_cache_key": p.path.parent.name,
        "timing": "fresh outputs, synchronized host call. Median of six windows of 31 calls, alternating traversal.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    a.output.write_text(
        json.dumps(
            [run("resnorm2", 32, 1536, torch.bfloat16), run("gate", 32, 8960, torch.bfloat16)],
            indent=2,
        )
        + "\n"
    )
