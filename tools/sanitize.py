"""Small set that actually launches the generated kernels under Compute Sanitizer."""

import torch
from examples.graphs import case
from sparkfuse import Schedule, compile_export
from sparkfuse.measure import compare

with torch.inference_mode():
    for name, rows, h, s in [
        ("resnorm2", 7, 113, Schedule(128, "warp", True)),
        ("resnorm2", 7, 113, Schedule(64, "warp", True, 8)),
        ("gate", 5, 3, Schedule(128, "flat", False, 8)),
        ("mean", 3, 127, Schedule(128, "block", False, 4)),
        ("decomp", 11, 128, Schedule(128, "block", True, 4)),
        ("gate", 47, 128, Schedule(128, "flat", False, 4)),
        ("mean", 47, 128, Schedule(64, "warp", False, 4)),
        ("scores2", 13, 97, Schedule(128, "warp", True)),
        ("scores", 7, 769, Schedule(256, "block", False)),
        ("scores2", 13, 97, Schedule(128, "warp", True, 8)),
        ("scores2", 3, 769, Schedule(128, "block", False, 4)),
    ]:
        m, a = case(name, rows, h, torch.bfloat16)
        a = tuple(torch.cat([x.new_zeros(1), x.flatten()])[1:].view_as(x) for x in a)
        k = compile_export(torch.export.export(m, a), schedule=s)
        compare(k(*a), m(*a), exact=(1,) if name in ("resnorm2", "scores2") else ())
        print(name, rows, h, "passed", flush=True)
