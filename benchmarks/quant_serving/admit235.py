"""Check the destination file system and Linux MemAvailable before a single-Spark 235B run."""

import argparse
import json
from pathlib import Path
import shutil
from audit235 import GIB, verify_inventory


def assess(budget, available, disk, phase):
    if phase not in ("transfer", "launch") or any(
        type(v) is not int or v < 0 for v in (available, disk)
    ):
        raise ValueError("expected a transfer/launch phase and nonnegative byte counts")
    obj = budget["source_object_bytes"]
    kv = budget["uniform_dense_fp16_kv_bytes"]
    reserve = 22 * GIB
    diskneed = 16 * GIB + (obj if phase == "transfer" else 0)
    need = obj + kv + reserve
    return dict(
        phase=phase,
        available_ram=available,
        free_disk=disk,
        required_ram=need,
        required_disk=diskneed,
        headroom_after_object_and_kv=available - obj - kv,
        reserved_ram=reserve,
        approved=available >= need and disk >= diskneed,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("transfer", "launch"))
    p.add_argument("weights_directory", type=Path)
    p.add_argument("--budget", type=Path, default=Path(__file__).with_name("qwen235b-headers.json"))
    p.add_argument("--lock", type=Path, default=Path(__file__).with_name("qwen235b-q2k.lock.json"))
    a = p.parse_args()
    budget = json.loads(a.budget.read_bytes())
    verify_inventory(a.lock, budget)
    # Only use the mount of the actual existing weights directory; never silently fall back to cwd.
    if not a.weights_directory.is_dir():
        p.error("weights_directory must already exist")
    fields = [
        s.split()
        for s in Path("/proc/meminfo").read_text().splitlines()
        if s.startswith("MemAvailable:")
    ]
    if len(fields) != 1 or len(fields[0]) != 3 or fields[0][2] != "kB":
        raise ValueError("missing Linux MemAvailable in KiB")
    r = assess(
        budget, int(fields[0][1]) * 1024, shutil.disk_usage(a.weights_directory).free, a.phase
    )
    print(json.dumps(r), flush=True)
    return 0 if r["approved"] else 75


if __name__ == "__main__":
    raise SystemExit(main())
