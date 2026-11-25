"""One isolated discovery, held-out validation, then a third seed for cross-backend comparison."""

import argparse, json, time
from pathlib import Path
import torch
from bench import bench
from tune_once import run
from sparkfuse import Schedule

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("request")
    p.add_argument("output", type=Path)
    p.add_argument("db", type=Path)
    a = p.parse_args()
    j = json.loads(a.request)
    t = time.time()
    dt = getattr(torch, j["dtype"])
    rec = run(j["case"], j["rows"], j["width"], dt, a.db, j["budget"], j["seed"])
    result = {"job": j, "tuning": rec, "status": "ok"}
    result["comparison"] = bench(
        j["case"],
        j["rows"],
        j["width"],
        dt,
        j["seed"] + 2000,
        do_profile=j.get("profile", False),
        sched=Schedule(**rec["selected"]),
    )
    result["elapsed_s"] = time.time() - t
    a.output.write_text(json.dumps(result, allow_nan=False) + "\n")
