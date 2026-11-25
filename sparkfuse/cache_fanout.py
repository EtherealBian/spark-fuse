"""Append KV and materialize repeated GEMM operands in one bit-copy kernel."""

from hashlib import sha256
import operator

import torch
from .cache_append import _candidate
from .head_repeat import _branch
from .index_program import IndexProgram, IndexRegion
from .ir import DT
from .rotary import _tensor


def emit(shape, groups, dtype, swapped):
    b, h, past, t, d = shape
    s = past + t
    size = 4 if dtype == "f32" else 2
    types = {2: "unsigned short", 4: "unsigned int", 8: "uint2", 16: "uint4"}
    widths = [w for w in (8, 4, 2, 1) if (w * size <= 16 and d % w == 0)]
    launch = []
    if len(widths) > 1:
        launch.append("uintptr_t addr=0; for(int j=0;j<8;++j)addr|=uintptr_t(p[j]);")
    for w in widths:
        ty = types[w * size]
        cond = f"if((addr & {w * size - 1})==0)" if w > 1 else ""
        args = ",".join(f"({'const ' if j < 4 else ''}{ty}*)p[{j}]" for j in range(8))
        blocks = (b * h * s * d // w + 127) // 128
        launch.append(
            f"{cond}{{cache_fanout<{w},{ty}><<<dim3({blocks},2),128,0,(cudaStream_t)stream>>>({args});return int(cudaGetLastError());}}"
        )
    flags = [str(x).lower() for x in swapped]
    return (
        f"""#include <cuda_runtime.h>
#include <stdint.h>
// Copy raw packets, not floating-point values. Old and incoming cache layouts may differ.
template<int W,typename P,bool old_sw,bool new_sw>
__device__ __forceinline__ void copy_one(const P* old,const P* next,P* cache,P* repeated,int i) {{
  constexpr int packs={d}/W, plane={s}*packs;
  int bh=i/plane, batch=bh/{h}, head=bh%{h}, tok=(i%plane)/packs, j=i%packs;
  P value;
  if(tok<{past}) {{ int row=old_sw ? (batch*{past}+tok)*{h}+head : bh*{past}+tok; value=old[row*packs+j]; }}
  else {{ int q=tok-{past}, row=new_sw ? (batch*{t}+q)*{h}+head : bh*{t}+q; value=next[row*packs+j]; }}
  cache[i]=value;
  P* dst=repeated+bh*({groups}*plane)+i%plane;
  #pragma unroll {min(groups, 8)}
  for(int group=0;group<{groups};++group) dst[group*plane]=value;
}}
template<int W,typename P>
__global__ void cache_fanout(const P* k,const P* kt,const P* v,const P* vt,P* ko,P* vo,P* kr,P* vr) {{
  int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>={b * h * s * d}/W) return;
  if(blockIdx.y==0)copy_one<W,P,{flags[0]},{flags[1]}>(k,kt,ko,kr,i);
  else copy_one<W,P,{flags[2]},{flags[3]}>(v,vt,vo,vr,i);
}}
extern "C" int launch(void** p,int,void* stream) {{
"""
        + "\n".join(launch)
        + "\n}\n",
        widths[0],
    )


def fuse_cache_fanout(gm, device, host_backend):
    """Require two materializing repeats of independent fresh length-axis cat results."""
    records = []
    used = set()
    branches = [
        br
        for n in gm.graph.nodes
        if (br := _branch(n)) is not None and _candidate(br.src) is not None
    ]
    for idx, a in enumerate(branches):
        if a.src in used:
            continue
        for b in branches[idx + 1 :]:
            if (
                b.src in used
                or a.src is b.src
                or (a.shape, a.dtype, a.device) != (b.shape, b.dtype, b.device)
            ):
                continue
            ca, cb = _candidate(a.src), _candidate(b.src)
            if ca[:3] != cb[:3] or not a.out.users or not b.out.users:
                continue
            # This schedule writes four tensors. Only use it when both caches have a use
            # beyond their private head-repeat view chain; never force an unused write.
            if any(
                not (
                    set(x.src.users) - {x.unsqueeze}
                    or set(x.unsqueeze.users) - {x.expand}
                    or set(x.expand.users) - {x.out}
                )
                for x in (a, b)
            ):
                continue
            args = (*a.src.args[0], *b.src.args[0])
            old = (a.src, b.src, a.out, b.out)
            if set(old).intersection(args):
                continue
            pos = {n: j for j, n in enumerate(gm.graph.nodes)}
            anchor = min(set().union(*(set(o.users) for o in old)), key=pos.get)
            if max(pos[n] for n in args) >= pos[anchor]:
                continue
            shape, dt, _, sw1 = ca
            sw = sw1 + cb[-1]
            g = a.shape[2]
            physical = [
                _tensor(n).transpose(1, 2) if swap else _tensor(n)
                for n, swap in zip(args, sw, strict=True)
            ]
            src, width = emit(shape, g, DT[dt], sw)
            pg = IndexProgram(
                [tuple(x.shape) for x in physical],
                [tuple(_tensor(n).shape) for n in old],
                DT[dt],
                device,
                src,
            )
            name = f"_sparkfuse_cache_fanout_{len(records)}"
            gm.add_module(name, IndexRegion(pg, host_backend))
            with gm.graph.inserting_before(anchor):
                live = []
                for n, swap, x in zip(args, sw, physical, strict=True):
                    if swap:
                        n = gm.graph.call_function(torch.ops.aten.transpose.int, (n, 1, 2))
                        n.meta["val"] = x
                    live.append(n)
                pack = gm.graph.call_module(name, tuple(live))
                for j, n in enumerate(old):
                    y = gm.graph.call_function(operator.getitem, (pack, j))
                    y.meta = dict(n.meta)
                    n.replace_all_uses_with(y)
            records.append(
                dict(
                    kind="cache_fanout",
                    module=name,
                    logical_shape=list(shape),
                    query_groups=g,
                    live_ins=[n.name for n in args],
                    live_outs=[n.name for n in old],
                    ops=[[n.name for n in (x.src, x.unsqueeze, x.expand, x.out)] for x in (a, b)],
                    physical_order=["BSHD" if x else "BHSD" for x in sw],
                    schedule=dict(
                        threads=128,
                        max_packet_elements=width,
                        alignment_policy="runtime_width",
                        stores_per_load=g + 1,
                    ),
                    cuda_sha256=sha256(src.encode()).hexdigest(),
                    host_backend=host_backend,
                )
            )
            used.update((a.src, b.src))
            for n in (a.out, b.out, a.expand, b.expand, a.unsqueeze, b.unsqueeze, a.src, b.src):
                if not n.users:
                    gm.graph.erase_node(n)
            break
    if records:
        gm.graph.lint()
        gm.recompile()
    return records
