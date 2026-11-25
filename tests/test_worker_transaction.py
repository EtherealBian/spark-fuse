"""Worker crashes and comparator failures must not silently publish new dispatch entries."""

from contextlib import closing
import json
from pathlib import Path
import pytest
from sparkfuse.tuning import connect
from sparkfuse import Schedule


@pytest.mark.parametrize(
    "outcome,previous,expected",
    [
        ("ok", "other", "same"),
        ("shape-error", "other", "other"),
        ("inductor", None, None),
        ("sparkfuse", "same", None),
        ("sparkfuse", "other", "other"),
    ],
)
def test_worker_publication(monkeypatch, tmp_path, outcome, previous, expected):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    import campaign_case as worker

    p = tmp_path / "db"
    dest = tmp_path / "example.pending"
    rec = {"selected": vars(Schedule()), "key": "key"}
    old = {"key": "key", "selected": vars(Schedule(64, "warp"))}
    if previous:
        with closing(connect(p)) as db, db:
            db.execute(
                "INSERT INTO schedules VALUES (?,?)",
                ("key", json.dumps(rec if previous == "same" else old)),
            )

    def discover(*args, store=True):
        assert not store
        return rec

    def compare(*args, **kwargs):
        if outcome == "shape-error":
            raise ValueError("fixture before result")
        return {
            "backends": {
                n: {"parity": "fail" if n == outcome else "pass"}
                for n in ("eager", "inductor", "sparkfuse")
            }
        }

    monkeypatch.setattr(worker, "run", discover)
    monkeypatch.setattr(worker, "bench", compare)
    monkeypatch.setattr(worker, "capture", lambda p: {"source_tree_sha256": "fixture"})
    msg = worker.execute(
        dict(case="fixture", rows=2, width=4, budget=2, seed=9, dtype="float32"), dest, p
    )
    if outcome != "ok":
        assert msg and not dest.exists()
        report = json.loads(dest.with_suffix(".rejected.json").read_text())
        assert report["status"] == (
            "comparison-error" if outcome == "shape-error" else "parity-failure"
        )
    else:
        assert msg is None and json.loads(dest.read_text())["status"] == "ok"
    if not p.exists():
        assert expected is None
        return
    with closing(connect(p)) as db:
        row = db.execute("SELECT record FROM schedules WHERE key=?", ("key",)).fetchone()
    assert (json.loads(row[0]) if row else None) == (
        {"same": rec, "other": old, None: None}[expected]
    )
