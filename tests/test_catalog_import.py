from copy import deepcopy
from contextlib import closing
import gzip, json, sqlite3, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from import_campaign import classify, import_bundle

# Synthetic metadata exercises conflict policy only, not the numerical checker.
SHA = "a" * 64
KEY = "b" * 64
CODE = "c" * 64
S = dict(threads=128, vec=2, cache=False, layout="warp")


def row(label):
    return dict(
        index=dict(status="ok", id=label),
        worker=dict(
            status="ok",
            provenance=dict(source_tree_sha256=SHA, archive="sources/" + SHA + ".tar.gz"),
            tuning=dict(compiler=CODE, key=KEY, admitted=True, selected=S.copy()),
            comparison=dict(
                backends={k: {"parity": "pass"} for k in ("eager", "inductor", "sparkfuse")}
            ),
        ),
    )


def test_strict_explicit_consent_and_whole_key_quarantine():
    a = row("one")
    b = row("two")
    r, n = classify([a, b], CODE)
    assert (
        list(r) == [KEY]
        and n == {"imported_records": 2}
        and [z["id"] for z in r[KEY]["evidence"]] == ["one", "two"]
    )
    for kind in ("different", "no_source", "implicit", "failed", "default"):
        bad = deepcopy(b)
        w = bad["worker"]
        if kind == "different":
            w["tuning"]["selected"]["threads"] = 256
        if kind == "no_source":
            w["provenance"] = {}
        if kind == "implicit":
            del w["comparison"]["backends"]["eager"]["parity"]
        if kind == "failed":
            bad["index"]["status"] = "error"
        if kind == "default":
            w["tuning"]["admitted"] = False
        for records in ([a, bad], [bad, a]):
            rows, n = classify(records, CODE)
            assert not rows and sum(n.values()) == 2


def test_exact_compiler_schema_and_typed_descriptors():
    a = row("one")
    assert not classify([a], SHA)[0]
    with pytest.raises(ValueError):
        classify([a], CODE[:10])
    for key, value in [
        ("threads", True),
        ("layout", "unknown"),
        ("vec", 0),
        ("cache", 0),
        ("extra", 1),
    ]:
        bad = deepcopy(a)
        bad["worker"]["tuning"]["selected"][key] = value
        with pytest.raises(ValueError):
            classify([bad], CODE)
    bad = deepcopy(a)
    bad["worker"]["tuning"]["key"] = "wrong"
    with pytest.raises(ValueError):
        classify([bad], CODE)


def test_empty_tampered_archive_and_overwrite(tmp_path):
    fixture = ROOT / "reports/campaign-dispatch-sample.json.gz"
    dest = tmp_path / "catalog.db"
    doc = json.load(gzip.open(fixture))
    compiler = next(
        q["worker"]["tuning"]["compiler"]
        for q in doc["records"]
        if q.get("worker", {}).get("tuning")
    )
    info = import_bundle(fixture, dest, compiler)
    assert info["unique_keys"] == 10 and info["counts"] == {
        "imported_records": 10,
        "failed_or_incomplete": 1,
        "default_retained": 1,
    }
    with closing(sqlite3.connect(dest.as_uri() + "?mode=ro", uri=True)) as db:
        assert json.loads(
            db.execute("select info from provenance").fetchone()[0]
        ) == info and db.execute("select count(*) from schedules").fetchone() == (10,)
    before = dest.read_bytes()
    with pytest.raises(FileExistsError):
        import_bundle(fixture, dest, compiler)
    assert dest.read_bytes() == before
    with pytest.raises(ValueError):
        import_bundle(fixture, tmp_path / "wrong.db", SHA)
    assert not (tmp_path / "wrong.db").exists()
    doc["records"][0]["index"]["elapsed_s"] += 1
    bad = tmp_path / "bad.json.gz"
    bad.write_bytes(gzip.compress(json.dumps(doc).encode()))
    with pytest.raises(ValueError, match="digest mismatch"):
        import_bundle(bad, tmp_path / "bad.db", compiler)
    assert not (tmp_path / "bad.db").exists()
