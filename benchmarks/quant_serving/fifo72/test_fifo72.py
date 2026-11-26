"""Run the broker, trace and loopback suites in isolation from other benchmark backends."""
import os
from pathlib import Path
import re
import subprocess
import sys
import pytest

HERE = Path(__file__).resolve().parent


@pytest.mark.parametrize(('module', 'count'), [
    ('suite_slot_admission', 7), ('suite_admission_trace', 3),
    ('suite_live72', 3), ('suite_wave_ledger', 3),
])
def test_fifo_suite(module, count):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, [HERE/'live', HERE, HERE.parent])))
    result = subprocess.run([sys.executable, '-B', '-S', '-m', 'unittest', module],
                            cwd=HERE.parents[2], env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert re.search(rf'(?m)^Ran {count} tests? in ', result.stderr) and result.stderr.rstrip().endswith('OK')
