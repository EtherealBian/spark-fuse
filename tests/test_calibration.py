import sqlite3
import torch, pytest
from sparkfuse import calibration as cal
from sparkfuse.partition import partition_export
from sparkfuse.measure import compare
from examples.graphs import ScoreSoftmax


def fixture():
    gen = torch.Generator(device="cuda").manual_seed(888)
    args = [
        (
            torch.randn(7, 33, device="cuda", generator=gen),
            torch.randn(33, device="cuda", generator=gen),
        )
        for _ in range(4)
    ]
    m = ScoreSoftmax(True)
    return m, torch.export.export(m, args[0]), args


def test_transactional_calibration_roundtrip(tmp_path):
    m, ep, args = fixture()
    db = tmp_path / "dispatch.sqlite"
    _, r = cal.calibrate_export(
        ep, *args[:3], db, budget=2, host_backend="python", exact_outputs=(1,)
    )
    gm, records = partition_export(ep, database=db)
    rr = next(iter(r["records"].values()))
    assert (
        records[0]["schedule"] == rr["selected"]
        and r["captured_distinct_storage_bytes"] == 3 * (7 * 33 + 33) * 4
    )
    compare(gm(*args[3]), m(*args[3]), exact=(1,))
    with sqlite3.connect(db) as conn:
        assert conn.execute("select count(*) from schedules").fetchone()[0] == 1


def test_capture_over_budget_cannot_publish(tmp_path):
    m, ep, args = fixture()
    db = tmp_path / "db"
    with pytest.raises(ValueError, match="storage exceeds"):
        cal.calibrate_export(ep, *args[:3], db, budget=1, capture_bytes_limit=1)
    assert not db.exists()


def test_composed_failure_does_not_publish(monkeypatch, tmp_path):
    m, ep, args = fixture()
    db = tmp_path / "db"
    real = cal.compare

    def reject(out, ref, exact=()):
        if exact == (1,):
            raise AssertionError("injected final failure")
        return real(out, ref, exact=exact)

    monkeypatch.setattr(cal, "compare", reject)
    with pytest.raises(AssertionError, match="injected final failure"):
        cal.calibrate_export(ep, *args[:3], db, budget=1, exact_outputs=(1,))
    assert not db.exists() and not list(tmp_path.glob(".sparkfuse-calibration-*"))


@pytest.mark.parametrize("failing_stage", ["train", "holdout", "audit"])
def test_all_composed_requests_guard_existing_database(monkeypatch, tmp_path, failing_stage):
    _, ep, args = fixture()
    db = tmp_path / "dispatch.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE sentinel(v)")
        conn.execute("INSERT INTO sentinel VALUES(42)")
    before = db.read_bytes()
    real = cal.compare
    calls = []

    def check(out, ref, exact=()):
        if exact == (1,):
            phase = ("train", "holdout", "audit")[len(calls)]
            calls.append(phase)
            if phase == failing_stage:
                raise AssertionError("injected at this input only")
        return real(out, ref, exact=exact)

    monkeypatch.setattr(cal, "compare", check)
    with pytest.raises(AssertionError, match="whole-graph " + failing_stage):
        cal.calibrate_export(ep, *args[:3], db, budget=1, exact_outputs=(1,))
    assert calls[-1] == failing_stage and db.read_bytes() == before
