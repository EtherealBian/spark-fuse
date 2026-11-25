"""Deadline-aware, single-GPU empirical tuning. No polling LLM / synthetic activity."""

import argparse, collections, datetime, fcntl, json, os, random, shutil, signal, statistics, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / ".runtime"
REPORT = ROOT / "reports"
NAMES = ("resnorm2", "gate", "decomp", "resnorm", "mean")


def queue():
    seen = set()

    def job(name, row, h, dtype, budget, phase):
        k = (name, row, h, dtype)
        if k in seen:
            return None
        seen.add(k)
        i = len(seen)
        return {
            "id": f"{name}-{row}x{h}-{dtype}",
            "case": name,
            "rows": row,
            "width": h,
            "dtype": dtype,
            "budget": budget,
            "phase": phase,
            "seed": 100 + i * 7,
            "profile": i in (1, 2, 3, 4),
        }

    # Real Qwen dimensions, low-latency decode, prefill, and non-divisible tails first.
    for row in (32, 1, 128, 7, 512):
        for n in NAMES:
            j = job(n, row, 8960 if n == "gate" else 1536, "bfloat16", 16, "qwen-reference")
            yield j
    for row in (1, 47, 97, 257):
        for n in NAMES:
            for dt in ("float32", "float16", "bfloat16"):
                yield job(n, row, 769 if n != "gate" else 1537, dt, 14, "tail-reduction")
    # Width/row stress grid; deterministic ordering, not a promise of full coverage by a deadline.
    grid = [
        (n, r, h, d)
        for n in NAMES
        for r in (1, 2, 4, 8, 16, 31, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024)
        for h in (
            768,
            1024,
            1536,
            2048,
            2560,
            3072,
            3584,
            4096,
            5120,
            6144,
            8192,
            8960,
            11008,
            14336,
        )
        for d in ("bfloat16", "float16", "float32")
        if r * h <= 8 * 1024 * 1024
    ]
    random.Random(3127).shuffle(grid)
    for n, r, h, d in grid:
        j = job(n, r, h, d, 24, "broad-search")
        if j:
            yield j


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def summary(records, until, finish=None):
    ok = [r for r in records if r["status"] == "ok"]
    ratios = [r["us"]["inductor"] / r["us"]["sparkfuse"] for r in ok]
    return {
        "updated_at_utc": utc(),
        "deadline": until.isoformat(),
        "stop_reason": finish,
        "completed": len(ok),
        "failed": len(records) - len(ok),
        "admitted": sum(r["admitted"] for r in ok),
        "sparkfuse_lower_device_median_than_inductor": sum(v > 1 for v in ratios),
        "median_ratio_inductor_over_sparkfuse": statistics.median(ratios) if ratios else None,
        "by_phase": dict(collections.Counter(r["phase"] for r in ok)),
        "last": records[-1] if records else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--until", required=True)
    p.add_argument("--max-cases", type=int)
    p.add_argument("--run", required=True)
    a = p.parse_args()
    until = datetime.datetime.fromisoformat(a.until)
    if until.tzinfo is None or not a.run.replace("-", "").isalnum():
        raise ValueError("need absolute timezone and safe run name")
    folder = LOCAL / a.run
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder / "evidence"
    raw.mkdir(exist_ok=True)
    index = folder / "index.jsonl"
    sumpath = REPORT / f"{a.run}-status.json"
    mutex = (LOCAL / "gpu-campaign.lock").open("a")
    fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
    records = (
        [json.loads(line) for line in index.read_text().splitlines()] if index.exists() else []
    )
    done = {r["id"] for r in records}
    stop = "worklist exhausted"
    streak = 0
    final_checks = REPORT / f"{a.run}-acceptance.json"
    for j in queue():
        if j["id"] in done:
            continue
        left = until.timestamp() - time.time()
        if left < 8:
            stop = "deadline"
            break
        if a.max_cases and len(records) >= a.max_cases:
            stop = "case limit"
            break
        if shutil.disk_usage(ROOT).free < 8 * 1024**3:
            stop = "disk reserve"
            break
        if (
            left < 600
            and not final_checks.is_file()
            and os.environ.get("SPARKFUSE_FINAL_CHECK") == "1"
        ):
            subprocess.run(
                [
                    "flock",
                    str(LOCAL / "gpu-execution.lock"),
                    sys.executable,
                    str(ROOT / "tools/release_check.py"),
                    str(until.timestamp()),
                    str(final_checks),
                ],
                timeout=max(1, left - 1),
                check=True,
            )
            left = until.timestamp() - time.time()
            if left < 8:
                stop = "deadline"
                break
        out = raw / (j["id"] + ".json")
        tmp = out.with_suffix(".pending")
        err = folder / "last-worker.log"
        print(utc(), j["id"], flush=True)
        t = time.monotonic()
        with err.open("w") as f:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(ROOT / "tools/campaign_case.py"),
                    json.dumps(j),
                    str(tmp),
                    str(folder / "dispatch.sqlite"),
                ],
                stdout=f,
                stderr=f,
                start_new_session=True,
            )
            timed = False
            try:
                ret = proc.wait(timeout=max(1, min(480, left - 4)))
            except subprocess.TimeoutExpired:
                timed = True
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    ret = proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    ret = proc.wait()
        rr = {"id": j["id"], "phase": j["phase"], "elapsed_s": round(time.monotonic() - t, 2)}
        if ret == 0 and tmp.is_file():
            data = json.loads(tmp.read_text())
            tmp.replace(out)
            b = data["comparison"]["backends"]
            q = data["tuning"]
            rr.update(
                status="ok",
                compiler=q["compiler"],
                trials=len(q["trials"]),
                invalid=len(q["invalid"]),
                admitted=q["admitted"],
                holdout=q["heldout_speedup_vs_default"],
                us={k: v["device"]["median_us"] for k, v in b.items()},
            )
            streak = 0
        else:
            rr.update(
                status="deadline-interruption"
                if timed and time.time() > until.timestamp() - 8
                else "error",
                stderr=err.read_text(errors="replace")[-4500:],
            )
            streak += 1
        records.append(rr)
        with index.open("a") as f:
            f.write(json.dumps(rr) + "\n")
            f.flush()
            os.fsync(f.fileno())
        write(sumpath, summary(records, until))
        if streak >= 5:
            stop = "five consecutive errors: inspect worker logs"
            break
    write(sumpath, summary(records, until, stop))
    if index.is_file():
        from report import render

        render(folder, REPORT / f"{a.run}-report.md")
    print(utc(), stop, flush=True)


if __name__ == "__main__":
    main()
