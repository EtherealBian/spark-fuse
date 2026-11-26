"""Validate all assets used by the offline 32B pilot against a fixed upstream revision."""

import hashlib
import json
from pathlib import Path

LOCK = Path(__file__).with_name("qwen32b.lock.json")
SPEC = json.loads(LOCK.read_text())
REV = "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
MODEL = "Qwen/Qwen2.5-32B-Instruct"
PARAMETERS = 32_763_876_352


def verify(directory):
    if (SPEC["model"], SPEC["revision"], SPEC["bf16_parameters"]) != (MODEL, REV, PARAMETERS):
        raise ValueError("incorrect checkpoint identity")
    expected = SPEC["files"]
    # Extra config, tokenizer or adapter files could change from_pretrained's behavior.
    present = {p.name for p in directory.iterdir()}
    allowed = set(expected) | {"README.md", ".gitattributes", ".cache"}
    if present - allowed or set(expected) - present:
        raise ValueError("unexpected or missing checkpoint files")
    for name, entry in expected.items():
        path = directory / name
        if path.stat().st_size != entry["size"]:
            raise ValueError(f"incorrect size: {name}")
        with path.open("rb") as f:
            if "sha256" in entry:
                ok = hashlib.file_digest(f, "sha256").hexdigest() == entry["sha256"]
            else:
                h = hashlib.sha1(f"blob {entry['size']}\0".encode())
                h.update(f.read())
                ok = h.hexdigest() == entry["git_blob"]
        if not ok:
            raise ValueError(f"digest mismatch: {name}")
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    shards = {k for k in expected if k.endswith(".safetensors")}
    if (
        len(shards) != 17
        or set(index["weight_map"].values()) != shards
        or index["metadata"]["total_size"] != PARAMETERS * 2
    ):
        raise ValueError("incorrect checkpoint index")


if __name__ == "__main__":
    import sys

    verify(Path(sys.argv[1]))
    print("verified all checkpoint assets", MODEL, REV)
