"""Enumerate schedules on fresh addresses; fail visibly on parity/launch errors. No speed selection."""

import argparse, json
from dataclasses import asdict
from pathlib import Path
import torch
from examples.graphs import case
from sparkfuse import compile_export, lower
from sparkfuse.tuning import backend_id, candidates
from sparkfuse.measure import compare


@torch.inference_mode()
def run(dt, width, output):
    m, a = case("layernorm2", 14, width, dt, 633)
    _, b = case("layernorm2", 14, width, dt, 241)
    b = tuple(
        torch.cat((x.new_zeros(i + 1), x.flatten()))[i + 1 :].view_as(x) for i, x in enumerate(b)
    )
    sets = (a, b)
    refs = [m(*x) for x in sets]
    ep = torch.export.export(m, a)
    configs = candidates(lower(ep))
    doc = dict(
        compiler=backend_id(),
        dt=str(dt),
        width=width,
        rows=14,
        seeds=[633, 241],
        element_offsets=[[0] * 4, [1, 2, 3, 4]],
        expected=len(configs),
        results=[],
        passed=False,
    )

    def save():
        tmp = output.with_suffix(".pending")
        tmp.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
        tmp.replace(output)

    for s in configs:
        k = compile_export(ep, schedule=s)
        record = dict(schedule=asdict(s), kernel=k.path.parent.name, status="running")
        doc["results"].append(record)
        save()
        try:
            record["errors"] = [compare(k(*x), ref, exact=(1,)) for x, ref in zip(sets, refs)]
            record["status"] = "pass"
        except Exception as e:
            record.update(status="fail", error=str(e))
            save()
            raise
    doc["passed"] = True
    save()
    print("Passed schedules:", len(configs), "on two requests", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    p.add_argument("--width", type=int, default=768)
    a = p.parse_args()
    run(getattr(torch, a.dtype), a.width, a.output)
