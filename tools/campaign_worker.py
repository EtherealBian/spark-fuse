"""One isolated discovery, held-out validation, then a third seed for cross-backend comparison."""

import json, time, traceback
from contextlib import closing
import torch
from provenance import capture
from bench import bench
from tune_once import run
from sparkfuse import Schedule
from sparkfuse.tuning import connect


def save(path, data):
    temp = path.with_suffix(path.suffix + ".writing")
    temp.write_text(json.dumps(data, allow_nan=False) + "\n")
    temp.replace(path)


def revoke_if_same(path, rec):
    if not path.exists():
        return
    with closing(connect(path)) as db, db:
        old = db.execute("SELECT record FROM schedules WHERE key=?", (rec["key"],)).fetchone()
        if old and json.loads(old[0])["selected"] == rec["selected"]:
            db.execute("DELETE FROM schedules WHERE key=?", (rec["key"],))


def execute(j, output, path):
    src = capture(path.parent / "sources")
    started = time.monotonic()
    result = {"job": j, "provenance": src}
    rejected = output.with_suffix(".rejected.json")
    phase = "discovery"
    try:
        dt = getattr(torch, j["dtype"])
        rec = run(j["case"], j["rows"], j["width"], dt, path, j["budget"], j["seed"], store=False)
        result["tuning"] = rec
        phase = "comparison"
        result["comparison"] = bench(
            j["case"],
            j["rows"],
            j["width"],
            dt,
            j["seed"] + 2000,
            do_profile=j.get("profile", False),
            sched=Schedule(**rec["selected"]),
        )
        b = result["comparison"]["backends"]
        if set(b) != {"eager", "inductor", "sparkfuse"}:
            raise ValueError("incomplete backend evidence")
        failures = [name for name, v in b.items() if v["parity"] != "pass"]
        if failures:
            if "sparkfuse" in failures:
                revoke_if_same(path, rec)
            result.update(
                status="parity-failure",
                failed_backends=failures,
                elapsed_s=time.monotonic() - started,
            )
            save(rejected, result)
            return (
                "parity failed for " + ", ".join(failures) + "; retained report: " + rejected.name
            )
        # Durably save the measured evidence before publishing. A crash during discovery
        # or an exception from Inductor can no longer expose an unvalidated candidate.
        result.update(status="ok", elapsed_s=time.monotonic() - started)
        phase = "evidence"
        save(output, result)
        phase = "dispatch"
        with closing(connect(path)) as db, db:
            db.execute(
                "INSERT OR REPLACE INTO schedules VALUES (?,?)",
                (rec["key"], json.dumps(rec, allow_nan=False)),
            )
        return None
    except Exception:
        result.update(
            status=phase + "-error",
            diagnostic=traceback.format_exc()[-5500:],
            elapsed_s=time.monotonic() - started,
        )
        save(rejected, result)
        return result["diagnostic"] + "\nretained report: " + rejected.name
