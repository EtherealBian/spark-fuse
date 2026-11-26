"""Build the native vocabulary-only observer in a fresh temporary directory and verify the four prompts."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
from natural_oracle import compute
from natural235 import ORACLE_SHA256, need
import hashlib


def verify(src, lib, weights, budget, runtime, probe_dir, cxx="c++"):
    lib = lib.resolve(strict=True)
    src = src.resolve(strict=True)
    observed = probe_dir / "natural_vocab.cpp"
    need(
        hashlib.sha256(observed.read_bytes()).hexdigest()
        == "e6db1e5b0d4bff65902dea5cc993ff6867b0f8d25fe05a3873104ccf49a304a9",
        "native probe source changed",
    )
    with tempfile.TemporaryDirectory() as tmp:
        exe = Path(tmp) / "vocab-only"
        subprocess.run(
            [
                cxx,
                "-O2",
                "-std=c++17",
                "-I" + str(src / "include"),
                "-I" + str(src / "ggml/include"),
                str(observed),
                str(lib),
                "-Wl,-rpath-link," + str(lib.parent),
                "-ldl",
                "-o",
                str(exe),
            ],
            check=True,
            timeout=90,
        )
        v = compute(weights, exe, lib, budget, runtime)
    data = (json.dumps(v, indent=2) + "\n").encode()
    need(
        hashlib.sha256(data).hexdigest() == ORACLE_SHA256
        and (probe_dir / "input-contract.json").read_bytes() == data,
        "independent tokenizer failed to reproduce frozen contract",
    )
    return data


if __name__ == "__main__":
    import sys

    p = argparse.ArgumentParser(description=__doc__)
    for n in ("llama_source", "libllama", "weights", "header_budget", "runtime"):
        p.add_argument(n, type=Path)
    p.add_argument("--cxx", default="c++")
    z = p.parse_args()
    sys.stdout.buffer.write(
        verify(
            z.llama_source,
            z.libllama,
            z.weights,
            z.header_budget,
            z.runtime,
            Path(__file__).resolve().parent,
            z.cxx,
        )
    )
