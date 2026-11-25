"""Build a static index program using the same guarded, current-stream ABI as scalar kernels."""

from types import SimpleNamespace
import torch
from torch import nn
from torch.utils import _pytree as tree
from .ir import Val, Unsupported
from .runtime import DenseKernelCall, build


class IndexProgram(DenseKernelCall):
    def __init__(self, inputs, outputs, dtype, device, source):
        n = len(inputs)
        m = len(outputs)
        self.plan = SimpleNamespace(
            inputs=list(range(n)),
            outputs=list(range(n, n + m)),
            vals=[
                Val("input" if j < n else "index", tuple(shape), dtype)
                for j, shape in enumerate([*inputs, *outputs])
            ],
            rows=1,
        )
        self.ep = SimpleNamespace(
            call_spec=SimpleNamespace(
                in_spec=tree.tree_flatten((tuple(range(n)), {}))[1],
                out_spec=tree.tree_flatten(tuple(range(m)))[1],
            )
        )
        self.bind = [None] * n
        dev = torch.device(device)
        if dev.type != "cuda":
            raise Unsupported("CUDA device required")
        self.device = torch.device(
            "cuda", torch.cuda.current_device() if dev.index is None else dev.index
        )
        self.source = source
        self.lib, self.path = build(source, self.device)


class IndexRegion(nn.Module):
    def __init__(self, compiled, host_backend):
        super().__init__()
        from .native import native_dispatch

        self.program = compiled
        self.dispatch = native_dispatch(compiled) if host_backend == "native" else compiled

    def forward(self, *args):
        return self.dispatch(*args)
