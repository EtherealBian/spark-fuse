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
        ("scores2", 13, 97, Schedule(128, "warp_xor", True)),
        ("scores2", 5, 257, Schedule(64, "warp_xor", False)),
        ("layernorm2", 11, 768, Schedule(128, "block_welford", True, 4)),
        ("layernorm2", 11, 1536, Schedule(128, "block_welford", False, 4)),
        ("layernorm2", 14, 768, Schedule(128, "block", True, 4)),
        ("layernorm2", 17, 129, Schedule(64, "warp", False, 8)),
    ]:
        m, a = case(name, rows, h, torch.bfloat16)
        a = tuple(torch.cat([x.new_zeros(1), x.flatten()])[1:].view_as(x) for x in a)
        k = compile_export(torch.export.export(m, a), schedule=s)
        compare(k(*a), m(*a), exact=(1,) if name in ("resnorm2", "scores2", "layernorm2") else ())
        print(name, rows, h, s, "passed", flush=True)

    from activation_audit import Activations

    for dtype, vec in ((torch.float16, 4), (torch.bfloat16, 8)):
        x = torch.randn(33 * 129 + 1, device="cuda", dtype=dtype)[1:].view(33, 129)
        m = Activations()
        k = compile_export(torch.export.export(m, (x,)), schedule=Schedule(128, "flat", False, vec))
        compare(k(x), m(x), exact=(0, 1, 2))
        print("gelu erf/tanh and cube", dtype, vec, "passed", flush=True)
