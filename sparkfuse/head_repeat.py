"""Fuse two materializing head repeats. Do not change either consumer's GEMM shape."""

from dataclasses import dataclass
from hashlib import sha256
from math import prod
import operator

import torch
from torch import fx

from .index_program import IndexProgram, IndexRegion
from .ir import DT
from .rotary import _is, _tensor

A = torch.ops.aten


@dataclass(frozen=True)
class Repeat:
    src: fx.Node
    unsqueeze: fx.Node
    expand: fx.Node
    out: fx.Node
    shape: tuple
    dtype: torch.dtype
    device: torch.device


def _branch(y):
    if not _is(y, A.reshape.default, 2):
        return None
    e = y.args[0]
    if not _is(e, A.expand.default, 2):
        return None
    u = e.args[0]
    if not _is(u, A.unsqueeze.default, 2) or u.args[1] not in (2, -3):
        return None
    x = u.args[0]
    vx, vy = _tensor(x), _tensor(y)
    ve = e.meta.get("val")  # axis 2 must have zero stride, so not a dense program input
    if (
        vx is None
        or vy is None
        or not isinstance(ve, torch.Tensor)
        or vx.ndim != 4
        or ve.ndim != 5
        or not vx.is_contiguous()
    ):
        return None
    if not all(type(n) is int and n > 0 for n in ve.shape):
        return None
    b, h, s, d = vx.shape
    g = ve.shape[2]
    if not (h > 1 and 1 < g <= 32 and tuple(ve.shape) == (b, h, g, s, d) and ve.stride(2) == 0):
        return None
    if (
        vy.dtype != vx.dtype
        or vy.device != vx.device
        or tuple(vy.shape) != (b, h * g, s, d)
        or tuple(vy.stride()) != (h * g * s * d, s * d, d, 1)
        or prod(vy.shape) > 2**29
    ):
        return None
    return Repeat(x, u, e, y, (b, h, g, s, d), vx.dtype, vx.device)


def emit(shape, dtype):
    b, h, g, s, d = shape
    size = 4 if dtype == "f32" else 2
    types = {2: "unsigned short", 4: "unsigned int", 8: "uint2", 16: "uint4"}
    widths = [w for w in (8, 4, 2, 1) if w * size <= 16 and (s * d) % w == 0]
    launch = []
    if len(widths) > 1:
        launch.append("  uintptr_t addr=0;for(int i=0;i<4;++i)addr|=uintptr_t(p[i]);")
    for w in widths:
        pack = types[w * size]
        cond = f"if((addr & {w * size - 1})==0)" if w > 1 else ""
        blocks = (b * h * s * d // w + 127) // 128
        launch.append(
            f"  {cond}{{ paired_head_repeat<{w},{pack}><<<dim3({blocks},2),128,0,(cudaStream_t)stream>>>((const {pack}*)p[0],(const {pack}*)p[1],({pack}*)p[2],({pack}*)p[3]);return int(cudaGetLastError()); }}"
        )
    body = "\n".join(launch)
    return (
        f"""#include <cuda_runtime.h>
#include <stdint.h>
template<int W,typename P> __global__ void paired_head_repeat(const P* a,const P* b,P* ya,P* yb) {{
  constexpr int plane={s * d}/W;
  int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>={b * h}*plane)return;
  P value=(blockIdx.y?b:a)[i];
  P* dst=(blockIdx.y?yb:ya)+(i/plane)*({g}*plane)+i%plane;
  #pragma unroll {min(g, 8)}
  for(int j=0;j<{g};++j)dst[j*plane]=value;
}}
extern "C" int launch(void** p,int,void* stream) {{
{body}
}}
""",
        widths[0],
    )


def fuse_head_repeats(gm, device, host_backend):
    records = []
    used = set()
    candidates = [n for n in gm.graph.nodes if _branch(n) is not None]
    for idx, u in enumerate(candidates):
        if u in used:
            continue
        for v in candidates[idx + 1 :]:
            if v in used:
                continue
            a, b = _branch(u), _branch(v)
            if (
                a is None
                or b is None
                or (a.shape, a.dtype, a.device) != (b.shape, b.dtype, b.device)
            ):
                continue
            if not u.users or not v.users:
                continue
            pos = {n: i for i, n in enumerate(gm.graph.nodes)}
            anchor = min(set(u.users) | set(v.users), key=pos.get)
            if {u, v}.intersection((a.src, b.src)) or max(pos[a.src], pos[b.src]) >= pos[anchor]:
                continue
            bs, h, g, s, d = a.shape
            dt = DT[a.dtype]
            src, width = emit(a.shape, dt)
            pg = IndexProgram([(bs, h, s, d)] * 2, [(bs, h * g, s, d)] * 2, dt, device, src)
            label = f"_sparkfuse_paired_head_repeat_{len(records)}"
            gm.add_module(label, IndexRegion(pg, host_backend))
            with gm.graph.inserting_before(anchor):
                both = gm.graph.call_module(label, (a.src, b.src))
                for i, n in enumerate((u, v)):
                    y = gm.graph.call_function(operator.getitem, (both, i))
                    y.meta = dict(n.meta)
                    n.replace_all_uses_with(y)
            records.append(
                dict(
                    kind="paired_head_repeat",
                    module=label,
                    logical_shape=list(a.shape),
                    live_ins=[a.src.name, b.src.name],
                    live_outs=[u.name, v.name],
                    ops=[[q.name for q in (br.unsqueeze, br.expand, br.out)] for br in (a, b)],
                    schedule=dict(
                        threads=128,
                        max_packet_elements=width,
                        alignment_policy="runtime_width",
                        copies_per_load=g,
                    ),
                    cuda_sha256=sha256(src.encode()).hexdigest(),
                    host_backend=host_backend,
                )
            )
            used.update((u, v))
            # Views can be externally visible or shared by both branches. Remove only dead nodes.
            for n in dict.fromkeys((u, v, a.expand, b.expand, a.unsqueeze, b.unsqueeze)):
                if not n.users:
                    gm.graph.erase_node(n)
            break
    if records:
        gm.graph.lint()
        gm.recompile()
    return records
