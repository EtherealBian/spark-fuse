"""Lower ordered half-RoPE on two projected slices, including an unrotated head suffix.

The projected tensor is already present in the graph. Only replace the two fresh
cat outputs; keep all observable views of the source, including V, as views.
"""

from dataclasses import dataclass
from hashlib import sha256
from math import prod
import operator

import torch
from .index_program import IndexProgram, IndexRegion
from .ir import DT
from .rotary import _arithmetic, _is, _tensor

A = torch.ops.aten


def _last_slice(n, rank):
    if not (_is(n, A.slice.Tensor, 4) or _is(n, A.slice.Tensor, 5)):
        return None
    x, axis, start, end, *step = n.args
    if (
        axis not in (-1, rank - 1)
        or type(start) is not int
        or type(end) is not int
        or start < 0
        or end < start
        or (step and step != [1])
    ):
        return None
    return x, start, end


@dataclass
class Part:
    out: object
    wide: object
    cos: object
    sin: object
    start: int
    heads: int
    dim: int
    rotary: int
    members: frozenset


def _part(out):
    if (
        not _is(out, A.cat.default, 2)
        or out.args[1] not in (-1, 3)
        or not isinstance(out.args[0], (tuple, list))
        or len(out.args[0]) != 2
    ):
        return None
    add, tail = out.args[0]
    matched = _arithmetic(add)
    if matched is None or set(add.users) != {out}:
        return None
    front, cos, sin, math_nodes = matched
    a, b = _last_slice(front, 4), _last_slice(tail, 4)
    # Export may canonicalize a full-width prefix slice to an alias. The suffix
    # must still be the matching empty slice of exactly this head tensor.
    if _is(front, A.alias.default, 1):
        a = (front.args[0], 0, _tensor(front).shape[-1])
    if a is None or b is None or a[0] is not b[0] or a[1] != 0:
        return None
    trans = a[0]
    if not _is(trans, A.transpose.int, 3) or trans.args[1:] not in ((1, 2), (2, 1)):
        return None
    view = trans.args[0]
    v = _tensor(trans)
    result = _tensor(out)
    if (
        not _is(view, A.view.default, 2)
        or v is None
        or v.ndim != 4
        or result is None
        or result.shape != v.shape
    ):
        return None
    batch, h, t, d = v.shape
    rd = _tensor(front).shape[-1]
    if not (
        0 < rd <= d
        and min(a[2], d) == rd
        and b[1] == rd
        and b[2] >= d
        and tuple(result.stride()) == (h * t * d, t * d, d, 1)
    ):
        return None
    raw = view.args[0]
    source = _last_slice(raw, 3)
    if source is None:
        return None
    wide, off, end = source
    x = _tensor(wide)
    vv = _tensor(view)
    if x is None or x.ndim != 3 or not x.is_contiguous() or x.numel() > 2**29 or vv is None:
        return None
    if (
        tuple(x.shape[:2]) != (batch, t)
        or tuple(vv.shape) != (batch, t, h, d)
        or min(end, x.shape[-1]) - off != h * d
    ):
        return None
    if any(
        z.dtype != x.dtype or z.device != x.device for z in (v, result, _tensor(cos), _tensor(sin))
    ):
        return None
    return Part(
        out, wide, cos, sin, off, h, d, rd, math_nodes | {front, tail, out, trans, view, raw}
    )


def emit(b, t, w, d, r, heads, offsets, shared, dt):
    hq, hk = heads
    oq, ok = offsets
    half = r // 2
    lanes = half + d - r
    nq = b * hq * t * lanes
    total = b * (hq + hk) * t * lanes
    typ = {"bf16": "__nv_bfloat16", "f16": "__half", "f32": "float"}[dt]
    bits = "unsigned int" if dt == "f32" else "unsigned short"
    return f"""#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
using T={typ}; using Bits={bits};
template<int H,int Off> __device__ __forceinline__ void sliced_head(const T*x,const T*c,const T*s,T*y,int i){{
  const int row=i/{lanes}, j=i%{lanes}, batch=row/(H*{t}), token=row%{t}, head=(row/{t})%H;
  const int in=(batch*{t}+token)*{w}+Off+head*{d}, out=row*{d};
  if(j<{half}){{
    const int p=(({0 if shared else "batch"}*{t}+token)*{r})+j;
    const float l=float(x[in+j]), h=float(x[in+j+{half}]);
    const T lc=T(l*float(c[p])),ls=T((-h)*float(s[p]));
    const T hc=T(h*float(c[p+{half}])),hs=T(l*float(s[p+{half}]));
    y[out+j]=T(float(lc)+float(ls)); y[out+j+{half}]=T(float(hc)+float(hs));
  }} else {{const int k=j+{half};reinterpret_cast<Bits*>(y)[out+k]=reinterpret_cast<const Bits*>(x)[in+k];}}
}}
__global__ void sliced_rotary_pair(const T*x,const T*c,const T*s,T*q,T*k){{
  const int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<{nq}) sliced_head<{hq},{oq}>(x,c,s,q,i);
  else if(i<{total}) sliced_head<{hk},{ok}>(x,c,s,k,i-{nq});
}}
extern "C" int launch(void**p,int,void*stream){{
  sliced_rotary_pair<<<{(total + 127) // 128},128,0,(cudaStream_t)stream>>>((const T*)p[0],(const T*)p[1],(const T*)p[2],(T*)p[3],(T*)p[4]);
  return int(cudaGetLastError());
}}
"""


def fuse_sliced_rotary(gm, device, host_backend):
    records = []
    used = set()
    outs = [n for n in gm.graph.nodes if _part(n) is not None]
    for i, n in enumerate(outs):
        if n in used or (a := _part(n)) is None:
            continue
        for n2 in outs[i + 1 :]:
            if n2 in used or (b := _part(n2)) is None:
                continue
            if not (
                a.wide is b.wide
                and a.cos is b.cos
                and a.sin is b.sin
                and a.dim == b.dim
                and a.rotary == b.rotary
            ):
                continue
            first, second = sorted((a, b), key=lambda z: z.start)
            if first.start + first.heads * first.dim > second.start:
                continue
            pos = {node: j for j, node in enumerate(gm.graph.nodes)}
            group = a.members | b.members
            users = set(a.out.users) | set(b.out.users)
            args = (a.wide, a.cos, a.sin)
            if not users or users & group or set(args) & group:
                continue
            anchor = min(users, key=pos.get)
            if any(pos[z] >= pos[anchor] for z in args):
                continue
            x = _tensor(a.wide)
            batch, t, w = x.shape
            shapes = [tuple(_tensor(z.out).shape) for z in (a, b)]
            dt = DT[x.dtype]
            assert sum(prod(v) for v in shapes) <= x.numel()
            heads = [a.heads, b.heads]
            offsets = [a.start, b.start]
            shared = _tensor(a.cos).shape[0] == 1
            src = emit(batch, t, w, a.dim, a.rotary, heads, offsets, shared, dt)
            prog = IndexProgram([tuple(_tensor(z).shape) for z in args], shapes, dt, device, src)
            label = f"_sparkfuse_sliced_rotary_{len(records)}"
            gm.add_module(label, IndexRegion(prog, host_backend))
            with gm.graph.inserting_before(anchor):
                call = gm.graph.call_module(label, args)
                for j, br in enumerate((a, b)):
                    z = gm.graph.call_function(operator.getitem, (call, j))
                    z.meta = dict(br.out.meta)
                    br.out.replace_all_uses_with(z)
            records.append(
                dict(
                    kind="sliced_rotary_pair",
                    module=label,
                    shape=list(x.shape),
                    head_dim=a.dim,
                    rotary_dim=a.rotary,
                    heads=heads,
                    offsets=offsets,
                    shared_phase=shared,
                    live_ins=[z.name for z in args],
                    live_outs=[a.out.name, b.out.name],
                    host_backend=host_backend,
                    schedule={"threads": 128},
                    cuda_sha256=sha256(src.encode()).hexdigest(),
                )
            )
            for z in sorted(group, key=pos.get, reverse=True):
                if not z.users:
                    gm.graph.erase_node(z)
            used.update((n, n2))
            break
    if records:
        gm.graph.lint()
        gm.recompile()
    return records
