"""Keep the isolated Qwen implementation outside the import path of unrelated backends."""

import os
from pathlib import Path
import re
import subprocess
import sys
import pytest

HERE = Path(__file__).resolve().parent


@pytest.mark.parametrize("module", ["suite_shape", "suite_projection_witness"])
def test_projection_panel(module):
    # The shape suite uses tensor libraries and Transformers; the compact one is stdlib only.
    paths = [str(HERE), str(HERE.parent), os.environ.get("PYTHONPATH", "")]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(paths), CUDA_VISIBLE_DEVICES="")
    flags = ["-B", "-S"] if module == "suite_projection_witness" else ["-B"]
    v = subprocess.run(
        [sys.executable, *flags, "-m", "unittest", module],
        cwd=HERE,
        env=env,
        text=True,
        capture_output=True,
        timeout=100,
    )
    assert v.returncode == 0, v.stderr
    assert re.search(r"(?m)^Ran 5 tests in ", v.stderr) and v.stderr.rstrip().endswith("OK")
