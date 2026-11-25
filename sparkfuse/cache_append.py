"""Join two independent length-axis concatenations (e.g. a functional KV-cache append).

Inputs may be physically head- or token-major. Both outputs remain distinct, fresh,
head-major tensors. No input mutation, spare capacity, cache arena, or padding assumed.
"""

from hashlib import sha256
from math import prod
import operator
import torch
from .index_program import IndexProgram, IndexRegion
from .ir import DT
from .rotary import _is, _tensor

A = torch.ops.aten


def _candidate(n):
    if (
        not _is(n, A.cat.default, 2)
        or n.args[1] not in (-2, 2)
        or not isinstance(n.args[0], (list, tuple))
        or len(n.args[0]) != 2
    ):
        return None
    vals = [_tensor(x) for x in (*n.args[0], n)]
    if any(v is None or v.ndim != 4 for v in vals):
        return None
    old, new, out = vals
    b, h, past, d = old.shape
    t = new.shape[2]
    if (
        any(x.dtype != old.dtype or x.device != old.device for x in vals)
        or tuple(new.shape) != (b, h, t, d)
        or tuple(out.shape) != (b, h, past + t, d)
        or tuple(out.stride()) != (h * (past + t) * d, (past + t) * d, d, 1)
        or prod(out.shape) > 2**29
    ):
        return None
    orders = []
    for x in (old, new):
        if x.is_contiguous():
            orders.append(False)
        elif x.transpose(1, 2).is_contiguous():
            orders.append(True)
        else:
            return None
    return (b, h, past, t, d), old.dtype, old.device, orders


def emit(shape, dt, sw, adaptive=True):
    b, h, past, t, d = shape
    size = 4 if dt == "f32" else 2
    widths = [v for v in (8, 4, 2, 1) if v * size <= 16 and d % v == 0]
    largest = widths[0]
    uint = {2: "unsigned short", 4: "unsigned int", 8: "uint2", 16: "uint4"}
    typ = uint[size]
    body = []
    if adaptive and len(widths) > 1:
        body.append("  uintptr_t addr=0; for(int i=0;i<6;++i) addr|=uintptr_t(p[i]);")
    for w in widths if adaptive else [largest]:
        g = (b * h * (past + t) * (d // w) + 127) // 128
        pack = uint[size * w]
        guarded = str(not adaptive).lower()
        cond = f"if((addr & {w * size - 1})==0) " if adaptive and w > 1 else ""
        body.append(
            f"  {cond}{{ paired_append<{w},{pack},{guarded}><<<dim3({g},2),128,0,(cudaStream_t)stream>>>((const E*)p[0],(const E*)p[1],(const E*)p[2],(const E*)p[3],(E*)p[4],(E*)p[5]); return int(cudaGetLastError()); }}"
        )
    launch = "\n".join(body)
    return (
        f"""#include <cuda_runtime.h>
#include <stdint.h>
using E={typ};
template<int W,typename P,bool guarded,int S,bool swapped>
__device__ __forceinline__ void move_bits(const E* x,E* out,int batch,int head,int token,int j) {{
  const int row=swapped ? (batch*S+token)*{h}+head : (batch*{h}+head)*S+token;
  const E* p=x+row*{d}+j;
  if constexpr (guarded) {{
    if((uintptr_t(p)|uintptr_t(out)) % sizeof(P)) {{
      #pragma unroll
      for(int k=0;k<W;++k)out[k]=p[k];
      return;
    }}
  }}
  *reinterpret_cast<P*>(out)=*reinterpret_cast<const P*>(p);
}}
template<int W,typename P,bool guarded,bool old_sw,bool new_sw>
__device__ __forceinline__ void append(const E* a,const E* b,E* y,int i) {{
  constexpr int packets={d}/W;
  int row=i/packets,j=(i%packets)*W,token=row%{past + t},head=(row/{past + t})%{h},batch=row/({h}*{past + t});
  if(token<{past}) move_bits<W,P,guarded,{past},old_sw>(a,y+i*W,batch,head,token,j);
  else move_bits<W,P,guarded,{t},new_sw>(b,y+i*W,batch,head,token-{past},j);
}}
template<int W,typename P,bool guarded>
__global__ void paired_append(const E* k,const E* kt,const E* v,const E* vt,E* ko,E* vo) {{
  int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>={b * h * (past + t) * d}/W)return;
  if(blockIdx.y==0)append<W,P,guarded,{str(sw[0]).lower()},{str(sw[1]).lower()}>(k,kt,ko,i);
  else append<W,P,guarded,{str(sw[2]).lower()},{str(sw[3]).lower()}>(v,vt,vo,i);
}}
extern "C" int launch(void** p,int,void* stream) {{
{launch}
}}
""",
        largest,
    )


def fuse_appends(gm, device, host_backend, adaptive=True):
    records = []
    cand = [n for n in gm.graph.nodes if _candidate(n)]
    used = set()
    for i, a in enumerate(cand):
        if a in used:
            continue
        for b in cand[i + 1 :]:
            if b in used:
                continue
            ca, cb = _candidate(a), _candidate(b)
            if ca is None or cb is None or ca[:3] != cb[:3]:
                continue
            args = (*a.args[0], *b.args[0])
            outside = (set(a.users) | set(b.users)) - {a, b}
            if not a.users or not b.users or not outside or {a, b}.intersection(args):
                continue
            pos = {n: k for k, n in enumerate(gm.graph.nodes)}
            anchor = min(outside, key=pos.get)
            if max(pos[x] for x in args) >= pos[anchor]:
                continue
            shape, dt, _, sw1 = ca
            sw = sw1 + cb[-1]
            source, vec = emit(shape, DT[dt], sw, adaptive)
            shapes = [
                tuple((_tensor(x).transpose(1, 2) if swap else _tensor(x)).shape)
                for x, swap in zip(args, sw, strict=True)
            ]
            program = IndexProgram(
                shapes, [tuple(_tensor(x).shape) for x in (a, b)], DT[dt], device, source
            )
            label = f"_sparkfuse_paired_append_{len(records)}"
            gm.add_module(label, IndexRegion(program, host_backend))
            inputs = []
            with gm.graph.inserting_before(anchor):
                for x, swap in zip(args, sw, strict=True):
                    if swap:
                        physical = gm.graph.call_function(A.transpose.int, (x, 1, 2))
                        physical.meta["val"] = _tensor(x).transpose(1, 2)
                        x = physical
                    inputs.append(x)
                pair = gm.graph.call_module(label, tuple(inputs))
                for j, n in enumerate((a, b)):
                    y = gm.graph.call_function(operator.getitem, (pair, j))
                    y.meta = dict(n.meta)
                    n.replace_all_uses_with(y)
            records.append(
                dict(
                    module=label,
                    kind="paired_append",
                    ops=[a.name, b.name],
                    live_ins=[x.name for x in args],
                    live_outs=[a.name, b.name],
                    physical_order=["BSHD" if x else "BHSD" for x in sw],
                    logical_shape=list(shape),
                    schedule=dict(
                        threads=128,
                        max_packet_elements=vec,
                        alignment_policy="runtime_width" if adaptive else "element_fallback",
                    ),
                    cuda_sha256=sha256(source.encode()).hexdigest(),
                    host_backend=host_backend,
                )
            )
            gm.graph.erase_node(a)
            gm.graph.erase_node(b)
            used.update((a, b))
            break
    if records:
        gm.graph.lint()
        gm.recompile()
    return records
