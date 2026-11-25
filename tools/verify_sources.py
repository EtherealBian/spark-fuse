"""Verify the source snapshot referenced by each supplied benchmark JSON; no GPU or extraction."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile

# Same source set as capture(); benchmark data, cache artifacts and weights are not members.
SOURCE_NAME = re.compile(
    r"(?:sparkfuse|examples|tools)/[A-Za-z0-9_]+\.py|sparkfuse/csrc/[A-Za-z0-9_]+\.cpp|pyproject\.toml"
)


def verify_archive(path, expected):
    hashes = {}
    total = 0
    with tarfile.open(path, "r|gz") as archive:
        for item in archive:
            total += item.size
            if (
                not item.isfile()
                or not SOURCE_NAME.fullmatch(item.name)
                or item.name in hashes
                or len(hashes) >= 512
                or item.size > 2 * 1024**2
                or total > 16 * 1024**2
            ):
                raise ValueError("unexpected source archive member: " + item.name)
            with archive.extractfile(item) as f:
                h = hashlib.sha256()
                while chunk := f.read(65536):
                    h.update(chunk)
            hashes[item.name] = h.digest()
    digest = hashlib.sha256()
    for name in sorted(hashes):
        digest.update(name.encode() + b"\0" + hashes[name])
    if "pyproject.toml" not in hashes or digest.hexdigest() != expected:
        raise ValueError("source tree hash mismatch: " + path.name)
    return len(hashes)


def verify_reports(paths):
    seen = {}
    count = 0
    for report in paths:
        p = json.loads(report.read_text()).get("provenance", {})
        key = p.get("source_tree_sha256", "")
        if not re.fullmatch("[0-9a-f]{64}", key) or p.get("archive") != f"sources/{key}.tar.gz":
            raise ValueError("missing or invalid provenance reference: " + report.name)
        archive = report.parent / p["archive"]
        if archive not in seen:
            seen[archive] = verify_archive(archive, key)
        count += 1
    return {
        "verified_reports": count,
        "distinct_archives": len(seen),
        "verified_source_members": sum(seen.values()),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path, nargs="+")
    print(json.dumps(verify_reports(p.parse_args().report)))
