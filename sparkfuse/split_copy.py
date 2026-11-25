"""Horizontally fuse three explicit, contiguous split copies. Never change a GEMM."""

from hashlib import sha256
from math import prod
import operator

import torch
from .index_program import IndexProgram, IndexRegion
from .ir import DT
from .rotary import _is, _tensor

A = torch.ops.aten


def candidate(split):
    if not _is(split, A.split_with_sizes.default, 3):
        return None
    x, widths, dim = split.args
    v = _tensor(x)
    if (
        v is None
        or not v.is_contiguous()
        or v.ndim < 1
        or dim not in (-1, v.ndim - 1)
        or not isinstance(widths, (tuple, list))
        or len(widths) != 3
        or not all(type(k) is int and k > 0 for k in widths)
        or sum(widths) != v.shape[-1]
        or v.numel() > 2**29
    ):
        return None
    picks = {}
    for n in split.users:
        if not _is(n, operator.getitem, 2):
            continue  # e.g. the raw split tuple can also escape
        idx = n.args[1]
        if type(idx) is not int or not -3 <= idx < 3:
            return None
        i = idx % 3
        clones = [
            u
            for u in n.users
            if u.op == "call_function"
            and u.target is A.clone.default
            and u.args == (n,)
            and u.kwargs == {"memory_format": torch.contiguous_format}
            and u.users
        ]
        if not clones:
            continue
        if len(clones) != 1 or i in picks:
            return None
        c = clones[0]
        y = _tensor(c)
        shape = (*v.shape[:-1], widths[i])
        strides = tuple(prod(shape[k + 1 :]) for k in range(len(shape)))
        if (
            y is None
            or tuple(y.shape) != shape
            or y.device != v.device
            or y.dtype != v.dtype
            or tuple(y.stride()) != strides
        ):
            return None
        picks[i] = (n, c)
    if set(picks) != {0, 1, 2}:
        return None
    return x, tuple(widths), [picks[k] for k in range(3)]


def emit(rows, widths, dt):
    w0, w1, w2 = widths
    total = sum(widths)
    size = 4 if dt == "f32" else 2
    packs = [k for k in (8, 4, 2, 1) if size * k <= 16 and all(w % k == 0 for w in widths)]
    typ = {2: "unsigned short", 4: "unsigned int", 8: "uint2", 16: "uint4"}
    body = []
    for k in packs:
        cond = f"if(!(addr & {k * size - 1})) " if k > 1 else ""
        body.append(
            f"{cond}{{ split_copy3<{k},{typ[k * size]}><<<{(rows * total // k + 127) // 128},128,0,(cudaStream_t)stream>>>(src,a,b,c); return int(cudaGetLastError()); }}"
        )
    return (
        f"""#include <cuda_runtime.h>
#include <stdint.h>
using E={typ[size]};
template <int K, class P> __global__ void split_copy3(const E* __restrict__ src, E* __restrict__ a, E* __restrict__ b, E* __restrict__ c) {{
  int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>={rows * total}/K)return;
  int row=i/({total}/K),col=(i%({total}/K))*K;
  E* dst;
  if(col<{w0}) dst=a+row*{w0}+col;
  else if(col<{w0 + w1}) dst=b+(row*{w1}+(col-{w0}));
  else dst=c+(row*{w2}+(col-{w0 + w1}));
  *reinterpret_cast<P*>(dst)=reinterpret_cast<const P*>(src)[i];
}}
extern "C" int launch(void** p,int,void* stream) {{
  auto *src=(const E*)p[0]; auto *a=(E*)p[1], *b=(E*)p[2], *c=(E*)p[3];
  uintptr_t addr=uintptr_t(src)|uintptr_t(a)|uintptr_t(b)|uintptr_t(c);
  {chr(10).join(body)}
}}
""",
        packs[0],
    )


def fuse_split_copies(gm, device, host_backend):
    records = []
    for split in list(gm.graph.nodes):
        match = candidate(split)
        if match is None:
            continue
        x, widths, picks = match
        v = _tensor(x)
        dt = DT[v.dtype]
        out_shapes = [(*v.shape[:-1], w) for w in widths]
        source, packet = emit(prod(v.shape[:-1]), widths, dt)
        prog = IndexProgram([tuple(v.shape)], out_shapes, dt, device, source)
        label = f"_sparkfuse_split_copy3_{len(records)}"
        gm.add_module(label, IndexRegion(prog, host_backend))
        # The input is available at the split; insert here even with interleaved users of the first clone.
        with gm.graph.inserting_before(split):
            call = gm.graph.call_module(label, (x,))
            for i, (_, c) in enumerate(picks):
                y = gm.graph.call_function(operator.getitem, (call, i))
                y.meta = dict(c.meta)
                c.replace_all_uses_with(y)
        records.append(
            dict(
                kind="split_copy3",
                module=label,
                live_ins=[x.name],
                live_outs=[c.name for _, c in picks],
                widths=list(widths),
                shape=list(v.shape),
                host_backend=host_backend,
                schedule=dict(max_packet_elements=packet, threads=128),
                cuda_sha256=sha256(source.encode()).hexdigest(),
            )
        )
        for n, c in picks:
            gm.graph.erase_node(c)
            if not n.users:
                gm.graph.erase_node(n)
        if not split.users:
            gm.graph.erase_node(split)
    if records:
        gm.graph.lint()
        gm.recompile()
    return records
