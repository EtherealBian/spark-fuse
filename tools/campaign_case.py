"""One isolated discovery, held-out validation, then a third seed for cross-backend comparison."""

import argparse, json, time
import fcntl
from pathlib import Path
import torch
from provenance import capture
from bench import bench
from tune_once import run
from sparkfuse import Schedule

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("request")
    p.add_argument("output", type=Path)
    p.add_argument("db", type=Path)
    a = p.parse_args()
    gpu_lock = (a.db.parent.parent / "gpu-execution.lock").open("a")
    fcntl.flock(gpu_lock, fcntl.LOCK_EX)
    j = json.loads(a.request)
    src = capture(a.db.parent / "sources")
    t = time.time()
    dt = getattr(torch, j["dtype"])
    rec = run(j["case"], j["rows"], j["width"], dt, a.db, j["budget"], j["seed"])
    result = {"job": j, "tuning": rec, "provenance": src, "status": "ok"}
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
    failures = [k for k, v in result["comparison"]["backends"].items() if v["parity"] != "pass"]
    if "sparkfuse" in failures:
        import sqlite3

        with sqlite3.connect(a.db) as db:
            db.execute("DELETE FROM schedules WHERE key=?", (rec["key"],))
    if failures:
        result["status"] = "parity-failure"
        path = a.output.with_suffix(".rejected.json")
        path.write_text(json.dumps(result, allow_nan=False) + "\n")
        raise SystemExit(
            "parity failed for " + ", ".join(failures) + "; retained report: " + path.name
        )
    a.output.write_text(json.dumps(result, allow_nan=False) + "\n")
