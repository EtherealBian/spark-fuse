"""Reconstruct the four inputs without an HTTP server, model weights, or a CUDA context."""

import hashlib, json, os, struct, subprocess, sys, tempfile
from pathlib import Path
from gguf_budget import Header
from shard_headers import part
from natural235 import KEYS, STEM, TAIL, need, digest


def sha(b):
    return hashlib.sha256(b).hexdigest()


def compute(weights, exe, lib, budget, runtime):
    first = json.loads(budget.read_bytes())["shards"][0]
    lib = lib.resolve()
    need(
        json.loads(runtime.read_bytes())["runtime_lib_sha256"][lib.name] == sha(lib.read_bytes()),
        "unverified tokenizer library",
    )
    with (weights / first["name"]).open("rb") as src:
        raw = src.read(first["tensor_data_start"])
    _, _, start, directory_digest = part(raw, first["source_object_bytes"])
    need(
        start == len(raw) and directory_digest == first["directory_sha256"],
        "unverified first-shard directory",
    )
    h = Header(raw)
    need(
        bytes(h.take(4)) == b"GGUF" and h.read("I") == 3 and h.read("Q") == first["tensors"],
        "wrong directory",
    )
    n = h.read("Q")
    pieces = []
    keys = set()
    align = 32
    need(0 < n < 4096, "unbounded metadata")
    for _ in range(n):
        s = h.pos
        k = h.string(True)
        need(k not in keys, "duplicate keys")
        keys.add(k)
        v = h.value(h.read("I"), False)
        if k == "general.alignment":
            align = v
        if not k.startswith("split."):
            pieces.append(raw[s : h.pos])
    need(
        {"tokenizer.ggml.tokens", "tokenizer.ggml.merges"} <= keys
        and type(align) is int
        and 0 < align <= 4096
        and align & (align - 1) == 0,
        "unexpected tokenizer",
    )
    view = struct.pack("<4sIQQ", b"GGUF", 3, 0, len(pieces)) + b"".join(pieces)
    view += bytes(-len(view) % align)
    prompts = [(True, "<|im_start|>user\n"), (True, "<|im_end|>\n<|im_start|>assistant\n")] + [
        (False, t) for k in ("key", "reason") for t in (STEM[k] * 350, "\n" + TAIL[k])
    ]
    data = "".join(f"{int(sp)}:{text.encode().hex()}\n" for sp, text in prompts)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "vocab.gguf"
        path.write_bytes(view)
        r = subprocess.run(
            [str(exe.resolve()), str(path), str(lib)],
            input=data,
            text=True,
            capture_output=True,
            timeout=30,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "LD_LIBRARY_PATH": str(lib.parent)},
        )
    need(
        r.returncode == 0 and "vocab only - skipping tensors" in r.stderr, "no vocabulary-only run"
    )
    lines = r.stdout.splitlines()
    need(len(lines) == 8 and all(v.startswith("tokens ") for v in lines[2:]), "unexpected protocol")
    eog = list(map(int, lines[1].split()[1:]))
    need(
        lines[1].startswith("eog ") and eog == json.loads(runtime.read_bytes())["eog_token_ids"],
        "EOG drift",
    )
    t = [list(map(int, v.split()[1:])) for v in lines[2:]]
    need(
        t[0].count(151644) == 1 and t[1].count(151644) == t[1].count(151645) == 1,
        "lost role controls",
    )
    inputs = {}
    for k in KEYS:
        n, f = k.split(":")
        b, s = (t[2], t[3]) if f == "key" else (t[4], t[5])
        prefix_len = int(n) - sum(map(len, (t[0], s, t[1])))
        need(0 < prefix_len < len(b), "bad length")
        ids = t[0] + b[:prefix_len] + s + t[1]
        inputs[k] = dict(tokens=len(ids), sha256=digest(ids))
    return dict(
        directory_sha256=directory_digest,
        aligned_prefix_sha256=sha(raw),
        metadata_view_sha256=sha(view),
        library_sha256=sha(lib.read_bytes()),
        eog_token_ids=eog,
        inputs=inputs,
        scope="Independent native vocabulary-only tokenization before serving; no inference result.",
    )


if __name__ == "__main__":
    print(json.dumps(compute(*map(Path, sys.argv[1:])), indent=2))
