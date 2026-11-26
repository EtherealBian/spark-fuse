"""Extract a small, path-free runtime receipt. Full server logs stay private."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

PIN = "66d65ec29ba7c440cbc31b6f63b74a17b536ba65"


def read_log(log, layers, slots, ubatch):
    s = re.sub(r"\x1b\[[0-9;]*m", "", log)

    def integer(key):
        all_values = {
            int(n)
            for n in re.findall(
                r"(?m)(?:^|\bllama_context:[ \t]*)" + key + r"[ \t]*=[ \t]*(\d+)[ \t]*$", s
            )
        }
        if len(all_values) != 1:
            raise ValueError("absent or ambiguous " + key)
        return all_values.pop()

    fields = {k: integer(k) for k in ("n_seq_max", "n_ctx", "n_ctx_seq", "n_batch", "n_ubatch")}
    if fields != dict(
        n_seq_max=slots, n_ctx=slots * 4096, n_ctx_seq=4096, n_batch=512, n_ubatch=ubatch
    ):
        raise ValueError("runtime geometry differs from explicit design")
    if f"offloaded {layers}/{layers} layers to GPU" not in s or "NVIDIA GB10" not in s:
        raise ValueError("all-layer placement or device missing")
    memory = []
    for device, kind, mib in re.findall(
        r"\b(CUDA\d+|CPU(?:_Mapped)?|CUDA_Host)\s+(model|KV|compute) buffer size\s*=\s*([\d.]+) MiB",
        s,
    ):
        memory.append(dict(device=device, kind=kind, reported_mib=float(mib)))
    if not any(
        v["device"] == "CUDA0" and v["kind"] == "model" and v["reported_mib"] > 0 for v in memory
    ):
        raise ValueError("missing CUDA weight buffer")
    return dict(
        context=fields,
        backend_buffers=memory,
        device="NVIDIA GB10",
        gpu_layers_including_output=layers,
        scope="Backend-reported buffer sizes, not unified-memory peak or process RSS.",
    )


def hash_file(f):
    with f.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--server", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--layers", type=int, required=True)
    p.add_argument("--slots", type=int, required=True)
    p.add_argument("--ubatch", type=int, required=True)
    a = p.parse_args()
    out = read_log(a.log.read_text(errors="replace"), a.layers, a.slots, a.ubatch)
    v = subprocess.check_output([str(a.server), "--version"], stderr=subprocess.STDOUT, text=True)
    if not re.search(r"version: 8183 \(66d65ec(?:29ba7c440cbc31b6f63b74a17b536ba65)?\)", v):
        raise ValueError("runtime build identifier differs from pinned baseline")
    proc = Path("/proc") / str(a.pid)
    if not (proc / "exe").samefile(a.server):
        raise ValueError("PID is not the launched backend")
    deps = set()
    for line in (proc / "maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6:
            f = Path(fields[5])
            if re.fullmatch(r"lib(?:ggml[\w-]*|llama|mtmd)\.so(?:\.[0-9]+)*", f.name):
                deps.add(f.resolve(strict=True))
    for prefix in ("libggml-cuda.so", "libggml-base.so", "libllama.so"):
        if not any(f.name.startswith(prefix) for f in deps):
            raise ValueError("live server is missing " + prefix)

    eog = sorted(
        {
            int(v)
            for v in re.findall(
                r"(?m)^print_info: EOG token\s*=\s*(\d+)\s", a.log.read_text(errors="replace")
            )
        }
    )
    if not eog:
        raise ValueError("backend did not identify its EOG tokens")
    out["eog_token_ids"] = eog
    out.update(
        llama_cpp_revision=PIN,
        server_sha256=hash_file(a.server),
        runtime_lib_sha256={f.name: hash_file(f) for f in sorted(deps)},
        driver=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip(),
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
