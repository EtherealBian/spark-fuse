"""Isolate the stream lab from other inference backends during repository-wide discovery."""

import os
from pathlib import Path
import re
import subprocess
import sys
import pytest

HERE = Path(__file__).resolve().parent


@pytest.mark.parametrize(
    ("module", "count"), [("suite_natural235", 7), ("suite_delivery_ledger", 6)]
)
def test_isolated(module, count):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, (HERE, HERE.parent))))
    v = subprocess.run(
        [sys.executable, "-B", "-S", "-m", "unittest", module],
        cwd=HERE.parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=65,
    )
    assert v.returncode == 0, v.stderr
    assert re.search(rf"(?m)^Ran {count} tests in ", v.stderr) and v.stderr.rstrip().endswith("OK")
