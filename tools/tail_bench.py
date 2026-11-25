"""Fixed predeclared scalar/packed comparison for odd token widths; includes losses."""

import argparse, json
from pathlib import Path
from statistics import median
import torch
from examples.graphs import case
from sparkfuse import Schedule, compile_export
from sparkfuse.measure import compare, device_time


@torch.inference_mode()
def run():
    results = []
    for n, t, v in [("gate", 128, 4), ("resnorm2", 256, 4)]:
        layout = "flat" if n == "gate" else "block"
        h = 1537 if n == "gate" else 769
        for rows in (1, 47, 257, 1024):
            m, a = case(n, rows, h, torch.bfloat16, seed=9893)
            ep = torch.export.export(m, a)
            ref = m(*a)
            r = {
                "case": n,
                "shape": [rows, h],
                "dtype": "bfloat16",
                "seed": 9893,
                "schedule": {"threads": t, "layout": layout, "cache": False},
                "rounds_us": [],
            }
            fn = []
            for vec in (1, v):
                k = compile_export(ep, schedule=Schedule(t, layout, False, vec))
                bound = k.prepare(*a)
                compare(bound(), ref, exact=(1,) if n == "resnorm2" else ())
                fn.append(bound)
            for trial in range(5):
                sample = {}
                for idx in [1, 0] if trial % 2 else [0, 1]:
                    sample[["scalar", "packed"][idx]] = device_time(fn[idx], groups=7)["median_us"]
                r["rounds_us"].append(sample)
            r["scalar_over_packed"] = median(w["scalar"] / w["packed"] for w in r["rounds_us"])
            results.append(r)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(), indent=2) + "\n")
