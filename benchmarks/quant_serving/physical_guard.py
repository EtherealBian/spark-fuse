"""Guard a single child by pidfd. Report system MemAvailable, not GPU peak usage."""

import json
import os
from pathlib import Path
import signal
import sys
import time

FLOOR = 6 * 2**30


def update(s, available):
    if type(available) is not int or available < 0:
        raise ValueError("invalid MemAvailable sample")
    return dict(
        samples=s["samples"] + 1,
        min_available_bytes=min(available, s["min_available_bytes"]) if s["samples"] else available,
        below=s["below"] + 1 if available < FLOOR else 0,
    )


def snapshot(path, state, reason):
    data = dict(
        schema=2,
        source="Linux /proc/meminfo MemAvailable (system-wide, sampled once per second; not GPU peak)",
        floor_bytes=FLOOR,
        reason=reason,
        **state,
    )
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(data) + "\n")
    temp.replace(path)


PHASES = ("loading", "requests")


def empty_phases():
    return {k: {"samples": 0, "min_available_bytes": None} for k in PHASES}


def observe(state, phases, phase, available):
    state = update(state, available)
    prev = phases[phase]
    phases[phase] = {
        "samples": prev["samples"] + 1,
        "min_available_bytes": min(prev["min_available_bytes"], available)
        if prev["samples"]
        else available,
    }
    return state


def guard(pid, exe, path):
    stopping = False

    def end(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, end)
    signal.signal(signal.SIGINT, end)
    fd = os.pidfd_open(pid)
    proc = Path("/proc") / str(pid)
    state = dict(samples=0, min_available_bytes=None, below=0)
    phases = empty_phases()
    phase = "loading"
    reason = "process-exited"

    def save(why):
        snapshot(path, dict(**state, phases=phases), why)

    try:
        # Allow the child to finish exec, but never sample or signal an unverified PID.
        for _ in range(20):
            if (proc / "exe").samefile(exe):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("backend identity not established")
        while not stopping:
            try:
                if not (proc / "exe").samefile(exe):
                    break
            except FileNotFoundError:
                break
            available = (
                int(
                    next(
                        line.split()[1]
                        for line in Path("/proc/meminfo").read_text().splitlines()
                        if line.startswith("MemAvailable:")
                    )
                )
                * 1024
            )
            if (path.parent / "requests.started").is_file():
                phase = "requests"
            state = observe(state, phases, phase, available)
            if state["below"] >= 3:
                reason = "low-memory-stop"
                save(reason)
                signal.pidfd_send_signal(fd, signal.SIGTERM)
                break
            if (path.parent / "requests.finished").is_file():
                reason = "workload-finished"
                break
            if state["samples"] == 1 or state["samples"] % 5 == 0:
                save("monitoring")
            time.sleep(1)
        if stopping and reason == "process-exited":
            reason = "monitor-stopped"
    finally:
        save(reason)
        os.close(fd)


def accepted(s):
    if not (
        s.get("schema") == 2
        and s.get("floor_bytes") == FLOOR
        and s.get("reason") == "workload-finished"
        and type(s.get("samples")) is int
        and s["samples"] >= 3
        and type(s.get("min_available_bytes")) is int
        and s["min_available_bytes"] >= 0
        and type(s.get("below")) is int
        and 0 <= s["below"] < 3
    ):
        raise ValueError("no accepted sampled guard: refuse capacity result")
    pp = s.get("phases", {})
    if (
        set(pp) != set(PHASES)
        or not all(
            isinstance(x, dict)
            and type(x.get("samples")) is int
            and x["samples"] >= 1
            and type(x.get("min_available_bytes")) is int
            and x["min_available_bytes"] >= 0
            for x in pp.values()
        )
        or sum(x["samples"] for x in pp.values()) != s["samples"]
        or min(x["min_available_bytes"] for x in pp.values()) != s["min_available_bytes"]
    ):
        raise ValueError("missing or inconsistent per-phase memory observations")


if __name__ == "__main__":
    if sys.argv[1] == "check":
        accepted(json.loads(Path(sys.argv[2]).read_bytes()))
    elif sys.argv[1] == "watch":
        guard(int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    else:
        raise ValueError("unknown mode")
