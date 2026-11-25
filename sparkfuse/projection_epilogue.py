"""Ordered INT32 accumulator -> FP32, row scale, then channel scale; no new GEMM."""

from hashlib import sha256
from math import prod
import operator

import torch

from .index_program import IndexProgram, IndexRegion
from .rotary import _is
from .token_quant import _dense

A = torch.ops.aten


def candidate(end, dev, pos):
    if not (_is(end, A.mul.Tensor, 2) and end.users):
        return None
    first, channel = end.args
    if not _is(first, A.mul.Tensor, 2):
        return None
    cast, row = first.args
    if not (_is(cast, A.to.dtype, 2) and cast.args[1] is torch.float32):
        return None
    acc = cast.args[0]
    v = getattr(acc, "meta", {}).get("val")
    c = getattr(channel, "meta", {}).get("val")
    if not isinstance(v, torch.Tensor) or v.ndim != 2 or not isinstance(c, torch.Tensor):
        return None
    shape = tuple(v.shape)
    if not all(type(k) is int and k > 0 for k in shape) or prod(shape) > 2**28:
        return None
    m, n = shape
    ch = tuple(c.shape)
    if (
        not all(type(k) is int for k in ch)
        or ch not in ((1, n), (n,))
        or not _dense(acc, shape, torch.int32, dev)
        or not _dense(row, (m, 1), torch.float32, dev)
        or not _dense(channel, ch, torch.float32, dev)
        or not all(_dense(t, shape, torch.float32, dev) for t in (cast, first, end))
    ):
        return None
    nodes = (cast, first, end)
    if row in nodes or channel in nodes:
        return None
    live = []
    for t, name in zip(nodes, ("cast", "row_scaled", "output"), strict=True):
        outside = [u for u in t.users if u not in nodes]
        if outside:
            # The scale could itself depend on an intermediate; do not move it before its producer.
            if any(pos[u] <= pos[end] for u in outside):
                return None
            live.append((t, name))
    if not live or live[-1][0] is not end:
        return None
    return (acc, row, channel), nodes, live, shape, ch


def emit(shape, roles):
    m, n = shape
    ptrs = ",".join("float* " + s for s in roles)
    args = ",".join(f"(float*)p[{i + 3}]" for i in range(len(roles)))
    return f"""#include <cuda_runtime.h>
__global__ void ordered_int32_epilogue(const int* acc,const float* token,const float* channel,{ptrs}) {{
    const int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>={m * n}) return;
    const float a=__int2float_rn(acc[i]);
    const float r=__fmul_rn(a,token[i/{n}]);
    {"cast[i]=a;" if "cast" in roles else "// cast is register-only"}
    {"row_scaled[i]=r;" if "row_scaled" in roles else "// row-scaled value is register-only"}
    output[i]=__fmul_rn(r,channel[i%{n}]);
}}
extern "C" int launch(void** p,int,void* stream) {{
    ordered_int32_epilogue<<<{(m * n + 127) // 128},128,0,(cudaStream_t)stream>>>((int*)p[0],(float*)p[1],(float*)p[2],{args});
    return int(cudaGetLastError());
}}
"""


def fuse_epilogues(gm, device, host_backend):
    dev = torch.device(device)
    records = []
    if dev.type != "cuda":
        return records
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    for end in list(gm.graph.nodes):
        if not _is(end, A.mul.Tensor, 2) or not end.users:
            continue
        pos = {n: i for i, n in enumerate(gm.graph.nodes)}
        site = candidate(end, dev, pos)
        if site is None:
            continue
        args, nodes, live, shape, ch = site
        roles = [s for _, s in live]
        source = emit(shape, roles)
        compiled = IndexProgram(
            [shape, (shape[0], 1), ch],
            [shape] * len(live),
            "f32",
            dev,
            source,
            input_dtypes=("i32", "f32", "f32"),
        )
        label = f"_sparkfuse_projection_epilogue_{len(records)}"
        gm.add_module(label, IndexRegion(compiled, host_backend))
        with gm.graph.inserting_before(end):
            call = gm.graph.call_module(label, args)
            for i, (old, _) in enumerate(live):
                y = gm.graph.call_function(operator.getitem, (call, i))
                y.meta = dict(old.meta)
                old.replace_all_uses_with(y)
        for t in reversed(nodes):
            gm.graph.erase_node(t)
        records.append(
            dict(
                kind="ordered_int32_epilogue",
                module=label,
                shape=list(shape),
                channel_shape=list(ch),
                output_roles=roles,
                ops=[t.name for t in nodes],
                live_ins=[t.name for t in args],
                live_outs=[t.name for t, _ in live],
                host_backend=host_backend,
                cuda_sha256=sha256(source.encode()).hexdigest(),
                schedule=dict(threads=128),
            )
        )
    if records:
        gm.graph.lint()
        gm.recompile()
    return records
