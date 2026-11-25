"""Three-way holdout for one preselected packet layout plus *static* instruction evidence."""

import argparse, json, re, subprocess
from collections import Counter
from pathlib import Path
import torch
from bench import bench
from examples.graphs import case
from sparkfuse import compile_export, Schedule
from sparkfuse.tuning import backend_id


@torch.inference_mode()
def run():
    name = "scores"
    rows = 12
    width = 8192
    dt = torch.bfloat16
    seed = 12572
    m, a = case(name, rows, width, dt, seed)
    ep = torch.export.export(m, a)
    out = {}
    for v in (1, 8):
        sc = Schedule(256, "block", True, v)
        k = compile_export(ep, schedule=sc)
        dis = subprocess.check_output(
            ["cuobjdump", "--dump-sass", str(k.path.with_name("kernel.so"))], text=True
        )
        # Instruction counts in the executable, not a dynamic instruction trace.
        ops = Counter(
            re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?((?:LDG|STG)(?:\.[A-Z0-9_]+)*)\b", dis)
        )
        assert ops, "could not identify global memory instructions"
        out[str(v)] = {
            "static_global_ops": dict(ops),
            "kernel_source": k.source,
            "kernel_cache_id": k.path.parent.name,
            "benchmark": bench(name, rows, width, dt, seed, True, sc),
        }
    return dict(
        compiler=backend_id(),
        seed=seed,
        comparison=out,
        interpretation="The scalar and packed layouts differ only in vector width. Device replay reuses addresses; static SASS counts are not dynamic memory transactions.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = run()
    a.output.write_text(json.dumps(r, indent=2, allow_nan=False) + "\n")
    if any(
        b["parity"] != "pass"
        for v in r["comparison"].values()
        for b in v["benchmark"]["backends"].values()
    ):
        raise SystemExit(2)
