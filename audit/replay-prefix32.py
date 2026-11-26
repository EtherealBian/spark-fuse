"""Replay archived 32B prefix decisions on CPU. Use a Linux LF checkout and Python 3.11+.

The tracked timing reports do not require this archive. Full replay also needs the pinned
release ZIP; the shape control uses NumPy. No network, checkpoint or CUDA is accessed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "benchmarks/pretrained_qwen"
GUARD = ROOT / "benchmarks/quant_serving"
FORK = ROOT / "reports/qwen32b-fork"
CUT = FORK / "cuts"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(p):
    return json.loads(p.read_bytes())


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


# These are the only analysis programs executed from the pinned data release.
REVIEWERS = {
    "shape/full_shape_review.py": "11547a148b8aa2806f4e774842fdc35c7d7093f7c60a81256a6ec36f800992ae",
    "restore/compare_model_trials.py": "2f3479e8b4e2979ec1a7ccf47868e63bca15620be40a519b21700c4ae613cc2a",
    "restore/fused-restore/model-trial/trial_audit.py": "0a57f827f745aef14bcbc6d7cb263a9bad4b7e69af29177a9e936df06e81a0bc",
}


def unpack(archive, dest, only=None):
    meta = read(CUT / "evidence-archive.json")
    need(
        archive.stat().st_size == meta["bytes"] and digest(archive) == meta["sha256"],
        "archive does not match pinned size or SHA-256",
    )
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        names = {x.filename for x in entries}
        need(
            len(entries) == len(names) == meta["members"]
            and sum(x.file_size for x in entries) == meta["expanded_bytes"] < 384 * 2**20,
            "archive inventory or extent changed",
        )
        for entry in entries:
            path = PurePosixPath(entry.filename)
            need(
                path.parts
                and path.parts[0] in ("cuts", "restore", "shape")
                and all(
                    x.isascii()
                    and x
                    and x not in (".", "..")
                    and all(c.isalnum() or c in "-_." for c in x)
                    for x in path.parts
                )
                and path.as_posix() == entry.filename
                and stat.S_ISREG(entry.external_attr >> 16),
                "non-regular or non-relative archive member",
            )
            if only is not None and path.parts[0] != only:
                continue
            target = dest.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(entry) as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
    for name, expected in REVIEWERS.items():
        if only is None or name.startswith(only + "/"):
            need(digest(dest / name) == expected, "unrecognized archived reviewer: " + name)


def check(expected, script, *args, numpy=False):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, (B, GUARD))))
    for key in ("PYTHONHOME", "PYTHONOPTIMIZE", "PYTHONINSPECT", "PYTHONSAFEPATH"):
        env.pop(key, None)
    cmd = [sys.executable, "-B", *([] if numpy else ["-S"]), script, *map(str, args)]
    result = subprocess.run(cmd, cwd=ROOT, env=env, check=True, stdout=subprocess.PIPE)
    need(result.stdout == expected.read_bytes(), "report changed: " + expected.name)
    print("OK", expected.parent.name + "/" + expected.name, flush=True)


def cuts(work):
    run = work / "cuts"
    for n in ("raw.json", "admission.json", "memory-guard.json", "timing.json"):
        shutil.copyfile(CUT / n, run / n)
    check(
        CUT / "summary.json",
        B / "prefix_cuts32/boundary_review.py",
        "bundle",
        run,
        ROOT / "reports",
    )
    check(CUT / "paired-costs.json", ROOT / "audit/prefix_pair_costs.py", run / "raw.json")


def restore(work):
    r = work / "restore"
    p = r / "prefix-fork"
    p.mkdir()
    for n in ("prefix_snapshot.py", "measure_fork32.py", "audit_fork32.py"):
        shutil.copyfile(B / n, p / n)
    for run in (r / "first", r / "replica"):
        check(run / "summary.json", r / "fused-restore/model-trial/trial_audit.py", run, FORK)
    check(
        CUT / "restore-comparison.json", r / "compare_model_trials.py", r / "first", r / "replica"
    )


def link_observations(index, dest, objects, parent):
    for record in read(index)["records"]:
        cell, arm = record["cell"], record["arm"]
        if parent and (cell[0] != 128 or arm not in ("whole", "inplace")):
            continue
        for key in ("logits", "kv"):
            meta = record[key]
            obj = objects / (meta["sha256"] + ".bf16")
            need(
                obj.stat().st_size == meta["bytes"] and digest(obj) == meta["sha256"],
                "incorrect content-addressed shape observation",
            )
            name = "-".join(map(str, (*cell, record["repeat"], arm, key))) + ".bf16"
            os.link(obj, dest / name)


def shape(work):
    r = work / "shape"
    old = r / "parent"
    old.mkdir()
    shutil.copyfile(FORK / "attribution/index.json", old / "index.json")
    for run, parent in ((r / "new", False), (old, True)):
        link_observations(run / "index.json", run, r / "objects", parent)
    check(
        r / "new/summary.json",
        r / "full_shape_review.py",
        r / "new",
        old,
        FORK / "layer0-shape",
        numpy=True,
    )
    observed = read(r / "new/summary.json")
    compact = read(CUT / "shape-verdict.json")
    need(
        {k: v for k, v in observed.items() if k != "measurements"} == compact,
        "shape decision disagrees with raw replay",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive", type=Path)
    ap.add_argument("--only", choices=("cuts", "restore", "shape"), help="select one audit")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="sparkfuse-prefix32-") as temp:
        work = Path(temp)
        unpack(args.archive, work, args.only)
        for audit in (cuts, restore, shape):
            if args.only in (None, audit.__name__):
                audit(work)


if __name__ == "__main__":
    main()
