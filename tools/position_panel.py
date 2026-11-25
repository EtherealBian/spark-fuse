"""Three-backend, three-input validation of positional-bias outer broadcasts. Serialize externally."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
from statistics import median
import torch
from examples.position_bias import CausalPositionPrior, inputs
from sparkfuse import Schedule, compile_export
from sparkfuse.measure import compare, device_time
from sparkfuse.tuning import backend_id, tune


@torch.inference_mode()
def run(shape):
    seeds = (1901, 1999, 2273)
    a, b, c = (inputs(*shape, "cuda", v, offset=i % 2) for i, v in enumerate(seeds))
    m = CausalPositionPrior()
    ep = torch.export.export(m, a)
    rec = tune(ep, a, m(*a), b, m(*b), None, budget=12, seed=75, exact=(1,), store=False)
    chosen = compile_export(ep, schedule=Schedule(**rec["selected"]))
    base = compile_export(ep)
    ind = torch.compile(m, backend="inductor", fullgraph=True)
    raw = {
        "shape_HQK": shape,
        "seeds": seeds,
        "compiler": backend_id(),
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(),
        "fx": str(ep.graph),
        "plan": chosen.plan.explain(),
        "tuning": rec,
        "schedule": asdict(chosen.schedule),
        "backends": {},
        "interleaved_order": [],
        "kernel_id": chosen.path.parent.name,
    }
    fns = {
        "eager": lambda: m(*c),
        "inductor": lambda: ind(*c),
        "default": base.prepare(*c),
        "selected": chosen.prepare(*c),
    }
    ref = m(*c)
    for k, fn in fns.items():
        try:
            errs = compare(fn(), ref, exact=(1,))
            raw["backends"][k] = {"passed": True, "errors": errs, "windows": []}
        except AssertionError as e:
            raw["backends"][k] = {"passed": False, "error": str(e)}
    rng = random.Random(3417)
    for _ in range(7):
        order = [k for k, v in raw["backends"].items() if v["passed"]]
        rng.shuffle(order)
        raw["interleaved_order"].append(order)
        for k in order:
            raw["backends"][k]["windows"].append(device_time(fns[k], groups=7, unroll=12, replay=4))
    for val in raw["backends"].values():
        if val["passed"]:
            val["device_us"] = median(x["median_us"] for x in val["windows"])
    return raw


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    results = []
    for shape in ((8, 1, 513), (8, 97, 129), (12, 128, 1024)):
        result = run(shape)
        results.append(result)
        print(
            shape,
            {k: v.get("device_us", v["passed"]) for k, v in result["backends"].items()},
            flush=True,
        )
        doc = dict(
            method="Causal positional prior in FP32, random positive head slopes, absolute positions, causal mask. Seven randomized, hot CUDA Graph replay windows on a third request, not full-model or host latency.",
            expected=3,
            results=results,
        )
        tmp = a.output.with_suffix(".pending")
        tmp.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
        tmp.replace(a.output)
    if not all(b["passed"] for r in results for b in r["backends"].values()):
        raise SystemExit(2)
