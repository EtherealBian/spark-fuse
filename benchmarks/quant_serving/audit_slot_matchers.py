"""Replay both CPU-only slot-reservation panels, preserving the unsuccessful design."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

# SHA-256 of the exact pre-registered sources used on the DGX Spark. Never run scripts from the supplied evidence directory.
SOURCES = {
    "ordered64": (
        "186cacca7a46aa387023416f679756a7a1e2a2377051c3a5ecbcb43d2d50d51c",
        "85021010fbd4874f73923d307e644e00c72a3acfa6f10b8d64460d4efccf0826",
        "b864e63bfe8d12ac850087269b0cb4c3dbc8e7196b8cc3627628847dbeee2552",
    ),
    "scoped16": (
        "66e7d02875a02af2c48b1520d701b7a1dc643a24e529865ff5f53195c13b9a5e",
        "d3792f8d375b16bff10b2c6c5c552ef1f02e7bb4f17be4a14cdda698fe4a189e",
        "f892d891bfec4ceb369c1bdff60ec99d4f9226a4a0b5562ce12fc10928561ab1",
    ),
}
NAMES = ("slot_policy.py", "prefix_index.py", "bench_profiled.py")


def read(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4_000_000:
        raise ValueError("expected one bounded regular input: " + path.name)
    return path.read_bytes()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replay(folder):
    code = Path(__file__).resolve().parent
    out = {}
    for key, digests in SOURCES.items():
        base = code / "comparisons" / "ordered64" if key == "ordered64" else code
        if tuple(sha(read(base / n)) for n in NAMES) != digests:
            raise ValueError("benchmark source differs: " + key)
        rows = folder / key / "rows.json"
        expected = read(folder / key / "summary.json")
        raw = read(rows)
        result = subprocess.check_output(
            [
                sys.executable,
                "-B",
                "-E",
                "-S",
                str(base / "bench_profiled.py"),
                "audit",
                str(rows.resolve()),
            ],
            cwd=base,
            timeout=60,
        )
        x = json.loads(result)
        if result != expected or x["rows_sha256"] != sha(raw):
            raise ValueError("summary does not match the rows: " + key)
        out[key] = dict(
            pairs=x["matched_pairs"],
            ten_percent_wins_on_64=x["ten_percent_wins_on_64"],
            passes_opt_in_gate=x["passes_opt_in_gate"],
            outside_regression_bounds=[
                k for k, t in x["cases"].items() if not t["within_regression_bounds"]
            ],
            rows_sha256=x["rows_sha256"],
            summary_sha256=sha(result),
        )
    return dict(
        schema="sparkfuse.slot-matcher.comparison.v1",
        panels=out,
        default="scan",
        scope="Complete CPU reservation API on one pinned Spark core; synthetic 2048-token workloads. No GPU inference or end-to-end service claim. Both panels retained independently.",
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path)
    print(json.dumps(replay(cli.parse_args().directory), indent=2, allow_nan=False))
