import argparse, json, time
from pathlib import Path
import torch
from examples.graphs import case
from sparkfuse.tuning import tune


@torch.inference_mode()
def run(name, rows, width, dtype, db, budget, seed):
    m, args = case(name, rows, width, dtype, seed)
    _, held = case(name, rows, width, dtype, seed + 1)
    ep = torch.export.export(m, args)
    r = tune(
        ep,
        args,
        m(*args),
        held,
        m(*held),
        db,
        budget=budget,
        seed=seed,
        exact=(1,) if name == "resnorm2" else (),
    )
    r.update(case=name, rows=rows, width=width, dtype=str(dtype))
    return r


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="resnorm2")
    p.add_argument("--rows", type=int, default=32)
    p.add_argument("--width", type=int, default=1536)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--budget", type=int, default=12)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = run(a.case, a.rows, a.width, getattr(torch, a.dtype), a.db, a.budget, a.seed)
    a.output.write_text(json.dumps(r, indent=2) + "\n")
