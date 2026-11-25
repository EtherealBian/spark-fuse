"""Cached Softmax rows with large private arrays: original geometries and adjacent odd tails."""

import pytest
import torch
from examples.graphs import case
from sparkfuse import Schedule, compile_export
from sparkfuse.measure import compare


@pytest.mark.parametrize(
    "name,rows,h,dtype,threads,layout,vec",
    [
        ("scores", 24, 8192, torch.float16, 64, "block", 2),
        ("scores2", 24, 2048, torch.float16, 512, "warp", 8),
        ("scores", 384, 4096, torch.bfloat16, 32, "block", 2),
        ("scores2", 9, 2047, torch.float16, 256, "warp", 8),
        ("scores", 17, 4097, torch.bfloat16, 64, "block", 4),
        ("scores", 13, 2049, torch.float32, 32, "block", 1),
    ],
)
@pytest.mark.parametrize("offset", [0, 1])
@torch.inference_mode()
def test_long_cached_row_matches_and_preserves_visible_logits(
    name, rows, h, dtype, threads, layout, vec, offset
):
    m, a = case(name, rows, h, dtype, seed=409 + 17 * offset)
    if offset:
        a = tuple(torch.cat([v.new_zeros(1), v.flatten()])[1:].view_as(v) for v in a)
    k = compile_export(torch.export.export(m, a), schedule=Schedule(threads, layout, True, vec))
    result = k(*a)
    torch.cuda.synchronize()
    compare(result, m(*a), exact=(1,) if name == "scores2" else ())
