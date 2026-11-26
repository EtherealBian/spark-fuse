"""Read the actual model bytes immediately before loading; emit a path-free receipt."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def verify(lock, source):
    m = json.loads(lock.read_bytes())
    entries = m["files"]
    single = source.is_file()
    if (
        not entries
        or not re.fullmatch("[0-9a-f]{40}", m["revision"])
        or (single and len(entries) != 1)
    ):
        raise ValueError("bad checkpoint lock or source")
    measured = []
    seen = set()
    for x in entries:
        n = x["name"]
        if (
            not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*\.gguf", n)
            or n in seen
            or type(x["bytes"]) is not int
            or x["bytes"] <= 0
            or not re.fullmatch("[0-9a-f]{64}", x["sha256"])
        ):
            raise ValueError("invalid entry")
        seen.add(n)
        f = source if single else source / n
        # A local HF cache may consist of symlinks to read-only content-addressed blobs.
        with f.open("rb") as stream:
            size = stream.seek(0, 2)
            stream.seek(0)
            if size != x["bytes"]:
                raise ValueError(f"{n}: byte length mismatch")
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != x["sha256"]:
                raise ValueError(f"{n}: digest mismatch")
        measured.append(dict(name=n, bytes=size, sha256=actual))
    return dict(
        schema=1,
        model=m["model"],
        revision=m["revision"],
        format=m["format"],
        files=measured,
        total_bytes=sum(x["bytes"] for x in measured),
        lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("lock", type=Path)
    p.add_argument("source", type=Path)
    a = p.parse_args()
    print(json.dumps(verify(a.lock, a.source), indent=2))
