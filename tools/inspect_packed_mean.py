"""Verify selected binaries against a packed-mean panel, then inspect them without launching CUDA."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess

from inspect_reduction_sass import op_counts


_RESOURCE = re.compile(r"Function\s+(\S+):\s*REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)")


def inspect(spec, cache, emit):
    key = spec["cache_id"]
    if not re.fullmatch(r"[a-f0-9]{64}", key):
        raise ValueError("invalid cache ID")
    src = (cache / key / "kernel.cu").read_bytes()
    binary = cache / key / "kernel.so"
    if (
        sha256(src).hexdigest() != spec["cuda_sha256"]
        or sha256(binary.read_bytes()).hexdigest() != spec["binary_sha256"]
    ):
        raise ValueError("selected source or binary no longer matches the measured artifact")
    usage = subprocess.check_output(["cuobjdump", "--dump-resource-usage", str(binary)], text=True)
    sites = op_counts(subprocess.check_output(["cuobjdump", "--dump-sass", str(binary)], text=True))
    resources = {
        name: dict(
            zip(
                ("registers", "stack_bytes", "static_shared_bytes", "local_bytes"),
                map(int, vals),
                strict=True,
            )
        )
        for name, *vals in _RESOURCE.findall(usage)
    }
    if set(resources) != set(sites):
        raise ValueError("disassembly and resource functions differ")
    if emit is not None:
        emit.mkdir(parents=True, exist_ok=True)
        f = emit / (spec["cuda_sha256"] + ".cu")
        if f.exists():
            if f.read_bytes() != src:
                raise ValueError("refusing to overwrite different existing text")
        else:
            with f.open("xb") as fp:
                fp.write(src)
    return dict(**spec, resources=resources, static_instruction_sites=sites)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("panel", type=Path)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--emit-dir", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    raw = a.panel.read_bytes()
    p = json.loads(raw)
    out = {}
    refs = []
    if p["selection_sha256"] or p["verify_only"]:
        raise ValueError("expected the discovery report")
    for i, case in zip(p["case_indices"], p["cases"], strict=True):
        for align in case["alignments"]:
            paths = {}
            for name, k in align["selected"].items():
                spec = case["kernels"][k]
                key = spec["cache_id"]
                paths[name] = key
                if key not in out:
                    out[key] = inspect(
                        spec, a.cache, a.emit_dir if i in (0, 5) and name == "packed_best" else None
                    )
            refs.append(dict(id=case["id"], offsets=align["offsets"], selected=paths))
    result = dict(
        panel_sha256=sha256(raw).hexdigest(),
        cuobjdump=subprocess.check_output(["cuobjdump", "--version"], text=True).strip(),
        interpretation="Static compiler resources and instruction sites, not observed runtime instruction or transaction counts. Emitted examples correspond to the packed winners for panels 0 and 5.",
        cases=refs,
        binaries=out,
    )
    a.output.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
