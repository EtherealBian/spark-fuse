"""Horizontal fusion of two proven half-rotate products sharing dense phase tables.

This is an optional static index program; it does not pretend irregular reads are
scalar DAG nodes. Physical contiguous arguments reuse the normal guarded launch ABI.
"""

from dataclasses import dataclass
from hashlib import sha256
from math import prod
import operator

import torch
from torch import fx

from .ir import DT
from .index_program import IndexProgram, IndexRegion

A = torch.ops.aten


def _is(n, target, arity):
    return (
        isinstance(n, fx.Node)
        and n.op == "call_function"
        and n.target is target
        and not n.kwargs
        and len(n.args) == arity
    )


def _tensor(n):
    if not isinstance(n, fx.Node):
        return None
    v = n.meta.get("val")
    return (
        v
        if isinstance(v, torch.Tensor)
        and v.dtype in DT
        and v.layout == torch.strided
        and v.device.type == "cuda"
        and not v.is_neg()
        and not v.is_conj()
        and all(type(k) is int and k > 0 for k in (*v.shape, *v.stride()))
        else None
    )


@dataclass
class Branch:
    out: fx.Node
    x: fx.Node
    c: fx.Node
    s: fx.Node
    members: frozenset
    bshd: bool


def _arithmetic(out):
    # Match the ordered arithmetic tree, not node names or module classes.
    if not _is(out, A.add.Tensor, 2):
        return None
    direct, rot = out.args
    if not (_is(direct, A.mul.Tensor, 2) and _is(rot, A.mul.Tensor, 2)):
        return None
    x, c = direct.args
    cat, s = rot.args
    if (
        not (_is(cat, A.cat.default, 2) and cat.args[1] in (-1, 3))
        or not isinstance(cat.args[0], (list, tuple))
        or len(cat.args[0]) != 2
    ):
        return None
    neg, lo = cat.args[0]
    if not _is(neg, A.neg.default, 1):
        return None
    hi = neg.args[0]
    vals = [_tensor(n) for n in (x, c, s, out)]
    if any(v is None or v.ndim != 4 for v in vals):
        return None
    vx, vc, vs, vo = vals
    b, h, t, d = vx.shape
    if (
        d % 2
        or tuple(vo.shape) != tuple(vx.shape)
        or any(v.dtype != vx.dtype for v in vals)
        or tuple(vc.shape) != tuple(vs.shape)
        or tuple(vc.shape) not in ((b, 1, t, d), (1, 1, t, d))
        or not vc.is_contiguous()
        or not vs.is_contiguous()
    ):
        return None
    for v, start, end in [(lo, 0, d // 2), (hi, d // 2, None)]:
        if not (_is(v, A.slice.Tensor, 4) or _is(v, A.slice.Tensor, 5)):
            return None
        if (
            v.args[0] is not x
            or v.args[1] not in (-1, 3)
            or v.args[2] != start
            or type(v.args[3]) is not int
            or (v.args[3] != end if end is not None else v.args[3] < d)
            or (len(v.args) == 5 and v.args[4] != 1)
        ):
            return None
    inside = frozenset((direct, rot, lo, hi, neg, cat, out))
    if len(inside) != 7 or any(set(n.users) - inside for n in inside if n is not out):
        return None
    return x, c, s, inside


def _branch(out):
    m = _arithmetic(out)
    if m is None:
        return None
    x, c, s, inside = m
    vx, vo = _tensor(x), _tensor(out)
    _, h, t, d = vx.shape
    # Output stride, including singleton axes, is observable. Only build a view
    # when its exact physical layout agrees with what ATen exported.
    heads = (h * t * d, t * d, d, 1)
    tokens = (t * h * d, d, h * d, 1)
    if vx.is_contiguous() and tuple(vo.stride()) == heads:
        bshd = False
    elif vx.transpose(1, 2).is_contiguous() and tuple(vo.stride()) == tokens:
        bshd = True
    else:
        return None
    return Branch(out, x, c, s, inside, bshd)


def emit(shapeq, shapek, shared_batch, layouts, dtype):
    b, hq, t, d = shapeq
    hk = shapek[1]
    half = d // 2
    nq = b * hq * t * half
    total = nq + b * hk * t * half
    typ = {"bf16": "__nv_bfloat16", "f16": "__half", "f32": "float"}[dtype]
    # Each product explicitly narrows before addition. build() disables FMA.
    return f"""#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
using T={typ};
template<int H, bool token_major>
__device__ __forceinline__ void rotate(const T* x,const T* c,const T* s,T* y,int i) {{
  const int row=i/{half}, j=i%{half}, batch=row/(H*{t});
  const int token=token_major ? (row/H)%{t} : row%{t};
  const int p=(({0 if shared_batch else "batch"}*{t}+token)*{d})+j, a=row*{d}+j;
  const float l=float(x[a]), r=float(x[a+{half}]);
  const T cl=T(l*float(c[p])), sl=T((-r)*float(s[p]));
  const T cr=T(r*float(c[p+{half}])), sr=T(l*float(s[p+{half}]));
  y[a]=T(float(cl)+float(sl)); y[a+{half}]=T(float(cr)+float(sr));
}}
__global__ void rotary_pair(const T* q,const T* k,const T* c,const T* s,T* oq,T* ok) {{
  int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<{nq}) rotate<{hq},{str(layouts[0]).lower()}>(q,c,s,oq,i);
  else if(i<{total}) rotate<{hk},{str(layouts[1]).lower()}>(k,c,s,ok,i-{nq});
}}
extern "C" int launch(void** p,int,void* stream) {{
  rotary_pair<<<{(total + 255) // 256},256,0,(cudaStream_t)stream>>>((const T*)p[0],(const T*)p[1],(const T*)p[2],(const T*)p[3],(T*)p[4],(T*)p[5]);
  return int(cudaGetLastError());
}}
"""


class _Call(IndexProgram):
    def __init__(self, q, k, device):
        v = [_tensor(n) for n in (q.x, k.x, q.c, q.s)]
        shapes = [
            tuple(x.transpose(1, 2).shape if swap else x.shape)
            for x, swap in zip(v, (q.bshd, k.bshd, False, False), strict=True)
        ]
        dt = DT[v[0].dtype]
        source = emit(
            tuple(v[0].shape), tuple(v[1].shape), v[2].shape[0] == 1, (q.bshd, k.bshd), dt
        )
        super().__init__(shapes, shapes[:2], dt, device, source)


def fuse_pairs(gm, device, host_backend):
    records = []
    branches = [m for n in gm.graph.nodes if (m := _branch(n)) is not None]
    used = set()
    for i, candidate in enumerate(branches):
        if candidate.out in used or (q := _branch(candidate.out)) is None:
            continue
        for candidate in branches[i + 1 :]:
            if candidate.out in used or (k := _branch(candidate.out)) is None:
                continue
            if k.x is q.x or k.c is not q.c or k.s is not q.s or k.members & q.members:
                continue
            vq, vk = _tensor(q.x), _tensor(k.x)
            if (vq.shape[0], *vq.shape[2:]) != (vk.shape[0], *vk.shape[2:]) or prod(
                vq.shape
            ) + prod(vk.shape) > 2**30:
                continue
            pos = {n: j for j, n in enumerate(gm.graph.nodes)}
            group = q.members | k.members
            outside = (set(q.out.users) | set(k.out.users)) - group
            if not outside or set((q.x, k.x, q.c, q.s)) & group:
                continue
            anchor = min(outside, key=pos.get)
            if any(pos[n] >= pos[anchor] for n in (q.x, k.x, q.c, q.s)):
                continue
            label = f"_sparkfuse_rotary_pair_{len(records)}"
            program = _Call(q, k, device)
            gm.add_module(label, IndexRegion(program, host_backend))
            with gm.graph.inserting_before(anchor):
                args = []
                for br in (q, k):
                    x = br.x
                    if br.bshd:
                        if _is(x, A.transpose.int, 3) and tuple(x.args[1:]) in ((1, 2), (2, 1)):
                            x = x.args[0]
                        else:
                            x = gm.graph.call_function(A.transpose.int, (x, 1, 2))
                            x.meta["val"] = _tensor(br.x).transpose(1, 2)
                    args.append(x)
                pair = gm.graph.call_module(label, (*args, q.c, q.s))
                for j, br in enumerate((q, k)):
                    y = gm.graph.call_function(operator.getitem, (pair, j))
                    v = _tensor(br.out)
                    if br.bshd:
                        y.meta["val"] = v.transpose(1, 2)
                        y = gm.graph.call_function(A.transpose.int, (y, 1, 2))
                    y.meta = dict(br.out.meta)
                    br.out.replace_all_uses_with(y)
            records.append(
                dict(
                    module=label,
                    kind="rotary_pair",
                    cuda_sha256=sha256(program.source.encode()).hexdigest(),
                    ops=[n.name for n in sorted(group, key=pos.get)],
                    live_ins=[n.name for n in (*args, q.c, q.s)],
                    live_outs=[q.out.name, k.out.name],
                    physical_order=["BSHD" if br.bshd else "BHSD" for br in (q, k)],
                    schedule={"layout": "paired_half_rotate", "threads": 256},
                    host_backend=host_backend,
                )
            )
            for n in sorted(group, key=pos.get, reverse=True):
                gm.graph.erase_node(n)
            for br in (q, k):
                if _is(br.x, A.transpose.int, 3) and not br.x.users:
                    gm.graph.erase_node(br.x)
            used.update((q.out, k.out))
            break
    if records:
        gm.graph.lint()
        gm.recompile()
    return records
