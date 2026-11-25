import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


def test_expired_deadline_retains_report_and_fails_cli(tmp_path):
    path = tmp_path / "acceptance.json"
    tool = Path(__file__).resolve().parents[1] / "tools/release_check.py"
    p = subprocess.run([sys.executable, str(tool), "0", str(path)], capture_output=True)
    assert p.returncode == 1 and not p.stderr
    doc = json.loads(path.read_text())
    assert doc["passed"] is False
    assert doc["checks"] == [{"check": "tests", "status": "no time"}]
    assert not path.with_suffix(".json.tmp").exists()


def test_worker_failure_is_not_hidden_by_subsequent_pass(tmp_path, monkeypatch):
    from tools import release_check

    monkeypatch.delenv("SPARKFUSE_QWEN_PY", raising=False)
    monkeypatch.setattr(
        release_check.subprocess,
        "run",
        lambda cmd, **kw: SimpleNamespace(
            returncode=12 if cmd[1:3] == ["-m", "pytest"] else 0, stdout="diagnostic"
        ),
    )
    report = tmp_path / "summary.json"
    assert not release_check.check(time.time() + 60, report)
    data = json.loads(report.read_text())
    checks = {x["check"]: x["status"] for x in data["checks"]}
    assert not data["passed"] and checks.pop("tests") == "fail"
    assert {"installed_wheel", "dynamic_native_shape_matrix"} <= checks.keys()
    assert set(checks.values()) == {"pass"}
