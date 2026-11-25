"""Lock before importing the compiler so worker evidence matches the running code generation."""

import argparse
import fcntl
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("request")
    p.add_argument("output", type=Path)
    p.add_argument("db", type=Path)
    a = p.parse_args()
    with (a.db.parent.parent / "gpu-execution.lock").open("a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        from campaign_worker import execute

        failure = execute(json.loads(a.request), a.output, a.db)
    if failure:
        raise SystemExit(failure)


if __name__ == "__main__":
    main()
