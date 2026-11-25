"""Build a static index program using the same guarded, current-stream ABI as scalar kernels."""

from types import SimpleNamespace
import torch
from torch import nn
from torch.utils import _pytree as tree
from .ir import Val, Unsupported
from .runtime import DenseKernelCall, build


class IndexProgram(DenseKernelCall):
    def __init__(self, inputs, outputs, dtype, device, source, *, input_strides=None):
        n = len(inputs)
        m = len(outputs)
        # A descriptor is relative to data_ptr(), never the backing storage base.
        # None keeps the normal dense guard. Only an explicitly specialized index
        # program can accept a noncompact input. Output allocation is unchanged.
        if input_strides is not None:
            if len(input_strides) != n:
                raise ValueError("one stride descriptor required per input")
            checked = []
            for shape, st in zip(inputs, input_strides, strict=True):
                if st is None:
                    checked.append(())
                    continue
                if (
                    not shape
                    or len(st) != len(shape)
                    or any(type(x) is not int or not 0 < x < 2**31 for x in (*shape, *st))
                    or sum((size - 1) * step for size, step in zip(shape, st)) >= 2**29
                ):
                    raise ValueError("invalid fixed input stride or address span")
                checked.append(tuple(st))
            self.input_strides = tuple(checked)
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
