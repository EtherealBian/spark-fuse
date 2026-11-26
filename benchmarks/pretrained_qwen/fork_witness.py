"""Replay stored, content-addressed BF16 observations. No network, GPU or model required."""

import argparse
import hashlib
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path

KINDS = {"logits": (8, 152064), "kv": (128, 3, 8, 128)}
ARMS = ("whole", "inplace", "restored")
METADATA = ("index.json", "admission.json", "memory-guard.json")
MAX_ARCHIVE = 48 * 2**20


def require(v, msg):
    if not v:
        raise ValueError(msg)


def digest(b):
    return hashlib.sha256(b).hexdigest()


def layout(index):
    from math import prod

    rows = index["records"]
    require(index["schema"] == "sparkfuse.fork.attribution.v1" and len(rows) == 48, "bad panel")
    expected = {
        (n, f, b, t, a)
        for n in (128, 2048)
        for f in (1, 2)
        for b in (0, 1)
        for t in (0, 1)
        for a in ARMS
    }
    m = {}
    for r in rows:
        c, t, a = r["cell"], r["repeat"], r["arm"]
        require(
            type(c) is list
            and len(c) == 3
            and all(type(x) is int for x in (*c, t))
            and isinstance(a, str)
            and (*c, t, a) in expected,
            "invalid/duplicate case",
        )
        expected.remove((*c, t, a))
        for kind, shape in KINDS.items():
            s = r[kind]
            h = s["sha256"]
            n = prod(shape) * 2
            require(
                s["shape"] == list(shape)
                and type(s["bytes"]) is int
                and s["bytes"] == n
                and re.fullmatch(r"[0-9a-f]{64}", h),
                "invalid extent/digest",
            )
            m["-".join(map(str, (*c, t, a, kind))) + ".bf16"] = (h, n)
    require(not expected, "incomplete panel")
    return m


def unique(m):
    u = {}
    for name, (h, size) in m.items():
        if h in u:
            require(u[h][0] == size, "inconsistent digest extent")
        else:
            u[h] = (size, name)
    return u


def reference(p):
    b = (p / "index.json").read_bytes()
    require(len(b) < 128 * 1024, "oversize index")
    return digest(b), layout(json.loads(b))


def make(source, archive):
    index_hash, m = reference(source)
    u = unique(m)
    require(set(m) == {p.name for p in source.glob("*.bf16")}, "binary membership differs")
    # Check EVERY original observation, including files sharing a content hash.
    for name, (h, n) in m.items():
        b = (source / name).read_bytes()
        require(len(b) == n and digest(b) == h, "changed source")
    with archive.open("xb") as fp, zipfile.ZipFile(fp, "w") as z:
        for h, (n, name) in sorted(u.items()):
            entry = zipfile.ZipInfo(h, date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100444 << 16
            b = (source / name).read_bytes()
            require(len(b) == n and digest(b) == h, "source changed while packing")
            z.writestr(entry, b, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    data = archive.read_bytes()
    require(0 < len(data) <= MAX_ARCHIVE, "oversize archive")
    return dict(
        schema="sparkfuse.fork.witness.v1",
        index_sha256=index_hash,
        archive_sha256=digest(data),
        archive_bytes=len(data),
        logical_files=len(m),
        logical_bytes=sum(n for h, n in m.values()),
        unique_payloads=len(u),
        unique_bytes=sum(n for n, name in u.values()),
    )


def materialize(report, archive, dest):
    """Write only to a new destination, using derived names, never ZIP paths or ZIP extraction."""
    index_hash, m = reference(report)
    u = unique(m)
    seal = json.loads((report / "witness.json").read_bytes())
    with archive.open("rb") as fp:
        n = os.fstat(fp.fileno()).st_size
        require(0 < n <= MAX_ARCHIVE, "oversize archive")
        actual = hashlib.file_digest(fp, "sha256").hexdigest()
        require(
            seal
            == dict(
                schema="sparkfuse.fork.witness.v1",
                index_sha256=index_hash,
                archive_sha256=actual,
                archive_bytes=n,
                logical_files=len(m),
                logical_bytes=sum(v for h, v in m.values()),
                unique_payloads=len(u),
                unique_bytes=sum(v for v, name in u.values()),
            ),
            "witness seal mismatch",
        )
        fp.seek(0)
        with zipfile.ZipFile(fp) as z:
            infos = z.infolist()
            require(
                len(infos) == len(u) and {i.filename for i in infos} == set(u), "ZIP membership"
            )
            dest.mkdir()  # caller owns a private temporary parent; never merge into old data
            for k in METADATA:
                (dest / k).write_bytes((report / k).read_bytes())
            for i in infos:
                size, name = u[i.filename]
                require(
                    i.file_size == size
                    and i.compress_type == zipfile.ZIP_DEFLATED
                    and not i.flag_bits & 1,
                    "ZIP extent/codec",
                )
                with z.open(i) as src:
                    b = src.read(size + 1)
                require(len(b) == size and digest(b) == i.filename, "ZIP payload digest")
                (dest / name).write_bytes(b)
            for name, (h, _) in m.items():
                canonical = dest / u[h][1]
                if name != canonical.name:
                    try:
                        os.link(canonical, dest / name)
                    except OSError:
                        import shutil

                        shutil.copyfile(canonical, dest / name)


def replay(report, archive, prior):
    from attribution_review import audit

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "observations"
        materialize(report, archive, p)
        return audit(p, prior)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="verb", required=True)
    m = s.add_parser("pack")
    m.add_argument("observations", type=Path)
    m.add_argument("new_zip", type=Path)
    r = s.add_parser("replay")
    r.add_argument("report", type=Path)
    r.add_argument("zip", type=Path)
    r.add_argument("prior_fork", type=Path)
    a = p.parse_args()
    v = (
        make(a.observations, a.new_zip)
        if a.verb == "pack"
        else replay(a.report, a.zip, a.prior_fork)
    )
    print(json.dumps(v, indent=2, allow_nan=False))
