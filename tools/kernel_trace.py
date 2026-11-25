"""Count CUDA kernel intervals from one ordinary forward, outside all timing windows."""

from collections import Counter
from tempfile import TemporaryDirectory
from pathlib import Path
import json
import torch


def kernel_census(fn):
    """Count GPU kernel intervals in ONE regular call, outside either timing loop."""
    fn()
    torch.cuda.synchronize()
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "kineto.json"
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            fn()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(path))
        events = json.loads(path.read_text())["traceEvents"]
    counts = Counter(e["name"] for e in events if e.get("ph") == "X" and e.get("cat") == "kernel")
    if not counts:
        raise RuntimeError("No Kineto GPU kernel intervals; check CUPTI availability")
    return dict(count=counts.total(), kernel_names=dict(sorted(counts.items())), calls=1)
