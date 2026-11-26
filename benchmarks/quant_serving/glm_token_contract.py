"""Pre-inference input contract using a genuine metadata-only tokenizer, not synthetic weights."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
from gguf_budget import Header
from prefix_replay import digest

HERE = Path(__file__).resolve().parent
HEADER = "cea08a50d72134c49bad68fe60653132880f797336f625e9e23717dab0f06067"
TEXTS = (
    "The ledger stores a previous version alongside each timestamp. " * 190,
    "A service assigns every request a durable identifier. " * 190,
    "\nQuestion: Why might an append-only ledger help recovery? Answer briefly:",
    "\nQuestion: How do identifiers help detect repeated submissions? Answer briefly:",
    "Warm up the inference backend.",
)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def must(x, msg):
    if not x:
        raise ValueError(msg)


def derive(raw):
    must(len(raw) == 16 * 2**20 and sha(raw) == HEADER, "unrecognized bounded first-shard prefix")
    r = Header(raw)
    must(bytes(r.take(4)) == b"GGUF" and r.read("I") == 3 and r.read("Q") == 916, "wrong header")
    n = r.read("Q")
    must(1 <= n < 4096, "metadata bound")
    seen = set()
    pieces = []
    tok_keys = []
    align = 32
    for _ in range(n):
        start = r.pos
        key = r.string(True)
        must(key not in seen, "duplicate metadata")
        seen.add(key)
        kind = r.read("I")
        v = r.value(kind, False)
        if key == "general.alignment":
            align = v
        if not key.startswith("split."):
            pieces.append(raw[start : r.pos])
            if key.startswith("tokenizer."):
                tok_keys.append(key)
    must(
        {"tokenizer.ggml.model", "tokenizer.ggml.tokens", "tokenizer.ggml.merges"} <= set(tok_keys)
        and type(align) is int
        and 1 <= align <= 4096
        and not (align & (align - 1)),
        "tokenizer or alignment missing",
    )
    view = struct.pack("<4sIQQ", b"GGUF", 3, 0, len(pieces)) + b"".join(pieces)
    view += bytes(-len(view) % align)
    return view, sorted(tok_keys)


def produce(header, exe, lib, ref):
    base = json.loads(ref.read_bytes())
    expected = base["runtime_lib_sha256"]
    must(
        sha(lib.read_bytes()) == expected[lib.resolve().name],
        "libllama differs from measured 235B runtime",
    )
    view, keys = derive(header.read_bytes())
    with tempfile.TemporaryDirectory(dir=HERE) as temp:
        file = Path(temp) / "tokenizer-only.gguf"
        file.write_bytes(view)
        proc = subprocess.run(
            [str(exe.resolve()), str(file), str(lib.resolve())],
            input="".join(s.encode().hex() + "\n" for s in TEXTS),
            capture_output=True,
            text=True,
            timeout=25,
            env={
                **os.environ,
                "LD_LIBRARY_PATH": str(lib.resolve().parent),
                "CUDA_VISIBLE_DEVICES": "",
            },
        )
        must(
            proc.returncode == 0 and "vocab only - skipping tensors" in proc.stderr,
            "metadata-only tokenizer failed: " + proc.stderr[-700:],
        )
        lines = proc.stdout.splitlines()
    must(
        len(lines) == 7
        and lines[0] == "vocab 151552"
        and lines[1].startswith("eog ")
        and all(x.startswith("tokens ") for x in lines[2:]),
        "incomplete tokenizer response",
    )
    eog = list(map(int, lines[1].split()[1:]))
    seq = [list(map(int, s.split()[1:])) for s in lines[2:]]
    must(
        eog and eog == sorted(set(eog)) and all(0 <= x < 151552 for s in [eog, *seq] for x in s),
        "invalid vocabulary/stop set",
    )
    inputs = {}
    for n in (128, 1024):
        for j in (0, 1):
            suffix = seq[2 + j]
            must(0 < len(suffix) < n < len(seq[j]), "tokenized template is too short")
            inp = seq[j][: n - len(suffix)] + suffix
            must(not set(inp).intersection(eog), "unexpected prompt EOG")
            inputs[f"{n}:{j}"] = {
                "input_sha256": digest(inp),
                "tokens": len(inp),
                "question_suffix_tokens": len(suffix),
            }
    must(len({x["input_sha256"] for x in inputs.values()}) == 4, "duplicate inputs")
    return dict(
        schema=1,
        source_header_sha256=HEADER,
        metadata_view_sha256=sha(view),
        metadata_view_bytes=len(view),
        tensor_count=0,
        tokenizer_metadata_keys=keys,
        libllama_sha256=sha(lib.read_bytes()),
        vocab_size=151552,
        eog_token_ids=eog,
        inputs=inputs,
        warmup_tokens=len(seq[-1]),
        scope="Pinned tokenizer metadata and CPU tokenization only. No model tensor payload, GPU inference, output IDs, timing or model-quality result.",
    )


def compare(contract, rows, runtime):
    must(
        runtime["eog_token_ids"] == contract["eog_token_ids"]
        and set(rows["inputs"]) == set(contract["inputs"]),
        "live tokenization contract differs",
    )
    for k, ids in rows["inputs"].items():
        c = contract["inputs"][k]
        must(
            isinstance(ids, list)
            and len(ids) == c["tokens"]
            and all(type(v) is int and 0 <= v < contract["vocab_size"] for v in ids)
            and digest(ids) == c["input_sha256"],
            "changed live input: " + k,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["preflight", "compare"])
    p.add_argument("paths", nargs="+", type=Path)
    a = p.parse_args()
    if a.mode == "preflight":
        print(json.dumps(produce(*a.paths), indent=2))
    else:
        compare(*(json.loads(x.read_bytes()) for x in a.paths))
        print(
            "four live inputs and backend stop IDs agree with the independently precomputed tokenizer contract"
        )
