"""Inspect the two frozen family winners in a compact-mean panel; never runs a kernel."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess


# One instruction, not its continuation encoding. Keep compiler predication out of the mnemonic.
INSTRUCTION = re.compile(r"/\*[0-9a-f]+\*/\s+(?:@!?[A-Z]*P(?:T|\d+)\s+)?([A-Z][A-Z0-9_.]+)\b")


def op_counts(sass):
    parts = re.split(r"Function\s*:\s*(\S+)", sass)
    functions = {}
    for name, body in zip(parts[1::2], parts[2::2]):
        counts = Counter(INSTRUCTION.findall(body))
        if not counts or name in functions:
            raise ValueError(
                "Expected named functions with one code object per function (single target)."
            )
        functions[name] = dict(sorted(counts.items()))
    if not functions:
        raise ValueError("No disassembled functions found.")
    return functions


def inspect(panel, cache, index):
    raw = panel.read_bytes()
    case = json.loads(raw)["cases"][index]
    families = {}
    for layout in ("block", "compact_mean"):
        schedule = case["selected"][layout]
        chosen = [t for t in case["trials"] if t["schedule"] == schedule]
        if len(chosen) != 1 or not re.fullmatch(r"[0-9a-f]{64}", chosen[0]["kernel_id"]):
            raise ValueError("ambiguous/invalid selected kernel")
        key = chosen[0]["kernel_id"]
        source, binary = (cache / key / f for f in ("kernel.cu", "kernel.so"))
        sass = subprocess.check_output(["cuobjdump", "--dump-sass", str(binary)], text=True)
        families[layout] = dict(
            kernel_cache_id=key,
            schedule=schedule,
            source_sha256=sha256(source.read_bytes()).hexdigest(),
            binary_sha256=sha256(binary.read_bytes()).hexdigest(),
            static_instructions=op_counts(sass),
        )
    return dict(
        panel_sha256=sha256(raw).hexdigest(),
        rows=case["rows"],
        width=case["width"],
        dtype=case["dtype"],
        cuobjdump=subprocess.check_output(["cuobjdump", "--version"], text=True).strip(),
        families=families,
        interpretation="Static instruction sites per function in the cached executable. These counts are not runtime instruction counts, memory transactions, or execution times.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("panel", type=Path)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--case-index", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    summary = inspect(args.panel, args.cache, args.case_index)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
