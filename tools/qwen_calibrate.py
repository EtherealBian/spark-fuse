"""Calibration using materialized intermediate tensors from real Qwen2Attention requests."""

import argparse, json, random
from pathlib import Path
from statistics import median
import torch
from qwen_scores import fixture
from sparkfuse.partition import partition_export
from sparkfuse.calibration import calibrate_export
from sparkfuse.measure import compare, device_time, host_time


@torch.inference_mode()
def run(db, batch, seq, budget):
    mod, inputs = fixture(batch, seq)
    seeds = [3201, 3202, 3203, 3204]
    a, b, c, final = [inputs(s) for s in seeds]
    ep = torch.export.export(mod, a)
    base, _ = partition_export(ep, host_backend="native")
    tuned, report = calibrate_export(ep, a, b, c, db, budget=budget, seed=649)
    paths = {"eager": mod, "default_mixed": base, "tuned_mixed": tuned}
    ref = mod(*final)
    rows = {}
    for label, fn in paths.items():
        rows[label] = {"fourth_input_errors": compare(fn(*final), ref), "windows": []}
    rng = random.Random(407)
    orders = []
    for _ in range(7):
        order = list(paths)
        rng.shuffle(order)
        orders.append(order)
        for key in order:
            f = lambda: paths[key](*final)
            rows[key]["windows"].append(
                {"device": device_time(f, groups=7), "host": host_time(f, samples=21)}
            )
    for item in rows.values():
        for kind in ("device", "host"):
            item[kind + "_median_us"] = median(x[kind]["median_us"] for x in item["windows"])
    return dict(
        batch=batch,
        seq=seq,
        torch=str(torch.__version__),
        dtype="bfloat16",
        device=torch.cuda.get_device_name(),
        seeds=seeds,
        calibration=report,
        validation=rows,
        order=orders,
        scope="One real Qwen2Attention module with random weights; eager attention only. Region selections optimize held-out device replay; whole-module gains are not assumed.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seq", type=int, default=97)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--budget", type=int, default=14)
    a = p.parse_args()
    a.output.write_text(
        json.dumps(run(a.db, a.batch, a.seq, a.budget), indent=2, allow_nan=False) + "\n"
    )
