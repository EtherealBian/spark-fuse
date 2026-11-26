"""Fixed upstream identity; validate names, sharded weights and tokenizer before use."""

import hashlib
import json
from pathlib import Path

REPO = "Qwen/Qwen2-7B-Instruct"
REV = "f2826a00ceef68f0f2b946d945ecc0477ce4450c"
META = {
    "config.json": "ae681edbd9486961e27d023f91d97b15562da7a9",
    "model.safetensors.index.json": "4bf699d11c6478a4b70fc2adfb405429de22525f",
    "tokenizer.json": "33ea6c72ebb92a237fa2bdf26c5ff16592efcdae",
    "tokenizer_config.json": "428f5926b8d79604d999b20b3bea98f5d4225a21",
}
SHARDS = {
    f"model-{i:05d}-of-00004.safetensors": sha
    for i, sha in enumerate(
        (
            "26d9919262ccd063fcdfd926763fe9025ef1e3073767aaa8c83a375d7c5140c4",
            "f5bb99fdadcac55c2c176497ec99f088a1764e78ed986fa4a0d45d12426ef0fa",
            "0b749a4446d7cda007d5e7bd9f908849d08d89867192d4c039dc167e9ab5a02e",
            "da724bb7d3c3512eb371aa6caa5bcc08d78bda84f94e00ae9a9b2124e3e9c62f",
        ),
        1,
    )
}


def verify(directory):
    for name, expected in META.items():
        d = (directory / name).read_bytes()
        if hashlib.sha1(f"blob {len(d)}\0".encode() + d).hexdigest() != expected:
            raise ValueError("metadata hash: " + name)
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    if set(index["weight_map"].values()) != set(SHARDS):
        raise ValueError("unexpected shard list")
    for name, expected in SHARDS.items():
        with (directory / name).open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != expected:
                raise ValueError("weight hash: " + name)


if __name__ == "__main__":
    import sys

    verify(Path(sys.argv[1]))
    print("verified checkpoint", REV)
