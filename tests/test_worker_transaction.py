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
    import campaign_worker as worker

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


def test_fault_diagnostic_keeps_original_error_and_kernel(tmp_path):
    from types import SimpleNamespace
    from sparkfuse.tuning import trial_context

    k = SimpleNamespace(
        schedule=Schedule(64, "warp"), path=tmp_path / "compiled-hash" / "kernel.cu"
    )
    fault = RuntimeError("device failure")
    with pytest.raises(RuntimeError) as err:
        with trial_context("holdout parity", k):
            raise fault
    assert err.value is fault
    assert any(
        "holdout parity" in n and "compiled-hash" in n and "warp" in n for n in fault.__notes__
    )


def test_cli_waits_before_importing_worker(tmp_path):
    """An arriving case must not import a stale backend while code/another worker owns the GPU lock."""
    import fcntl
    import os
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "tools/campaign_case.py"
    job = tmp_path / "job"
    job.mkdir()
    present = tmp_path / "entered"
    # Execute in a fresh interpreter. Replace only the worker module, keep the actual file lock.
    program = """
import fcntl, importlib.abc, importlib.util, os, runpy, sys
from pathlib import Path
class Fake(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self,name,path,target=None):
        if name=='campaign_worker': return importlib.util.spec_from_loader(name,self)
    def create_module(self,spec):return None
    def exec_module(self,mod):
        Path(os.environ['ENTERED']).touch();mod.execute=lambda *args: None
sys.meta_path.insert(0,Fake()); orig=fcntl.flock
def record(*args):
    print('waiting',flush=True);return orig(*args)
fcntl.flock=record
sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')
"""
    with (tmp_path / "gpu-execution.lock").open("a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        with subprocess.Popen(
            [
                sys.executable,
                "-c",
                program,
                str(script),
                "{}",
                str(job / "output"),
                str(job / "db"),
            ],
            stdout=subprocess.PIPE,
            text=True,
            env={**os.environ, "ENTERED": str(present)},
        ) as child:
            try:
                import select

                assert (
                    select.select([child.stdout], [], [], 10)[0]
                    and child.stdout.readline().strip() == "waiting"
                )
                assert not present.exists()
                fcntl.flock(lk, fcntl.LOCK_UN)
                assert child.wait(10) == 0 and present.is_file()
            finally:
                if child.poll() is None:
                    child.kill()
