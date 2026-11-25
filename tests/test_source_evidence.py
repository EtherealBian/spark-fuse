import gzip
import hashlib
import io
import json
import tarfile

import pytest
from tools.verify_sources import verify_reports


def packed(tmp_path, names=("pyproject.toml", "sparkfuse/ir.py"), tamper=False):
    h = hashlib.sha256()
    for name in sorted(set(names)):
        h.update(name.encode() + b"\0" + hashlib.sha256(b"reference").digest())
    key = h.hexdigest()
    folder = tmp_path / "sources"
    folder.mkdir()
    filename = f"sources/{key}.tar.gz"
    with tarfile.open(tmp_path / filename, "w:gz") as tar:
        for name in names:
            data = b"changed" if tamper else b"reference"
            item = tarfile.TarInfo(name)
            item.size = len(data)
            tar.addfile(item, io.BytesIO(data))
    p = tmp_path / "case.json"
    p.write_text(json.dumps({"provenance": {"source_tree_sha256": key, "archive": filename}}))
    return p


def test_portable_sources_verified_once(tmp_path):
    p = packed(tmp_path)
    assert verify_reports([p, p]) == {
        "verified_reports": 2,
        "distinct_archives": 1,
        "verified_source_members": 2,
    }


@pytest.mark.parametrize(
    "names,tamper",
    [
        (("pyproject.toml", "../bad.py"), False),
        (("pyproject.toml", "pyproject.toml"), False),
        (("pyproject.toml",), True),
    ],
)
def test_reject_malformed_or_changed_evidence(tmp_path, names, tamper):
    p = packed(tmp_path, names, tamper)
    with pytest.raises(ValueError):
        verify_reports([p])


def test_do_not_follow_arbitrary_report_path(tmp_path):
    p = tmp_path / "case.json"
    p.write_text(
        json.dumps({"provenance": {"source_tree_sha256": "0" * 64, "archive": "../unrelated"}})
    )
    with pytest.raises(ValueError, match="reference"):
        verify_reports([p])


def test_diagnostic_gzip_can_reference_the_same_source(tmp_path):
    p = packed(tmp_path)
    log = p.with_suffix(".json.gz")
    log.write_bytes(gzip.compress(p.read_bytes(), mtime=0))
    assert verify_reports([p, log])["distinct_archives"] == 1
