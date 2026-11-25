"""Read one dense packed Q/K/V input; rotate private Q/K and copy the observable V.

Does not compute, merge or otherwise substitute any projection GEMM. All three explicit
contiguous clones, two ordered half-rotations and exact physical output strides are proven
from the exported graph. Any still-live raw split views continue to alias the original.
"""

from hashlib import sha256
import operator
import torch
from .ir import DT
from .index_program import IndexProgram, IndexRegion
from .rotary import _branch, _is, _tensor
from .split_copy import candidate as split_copies

A = torch.ops.aten


def _from_clone(branch, clone, b, t, width):
    trans = branch.x
    if not _is(trans, A.transpose.int, 3) or tuple(trans.args[1:]) not in ((1, 2), (2, 1)):
        return None
    view = trans.args[0]
    if not _is(view, A.view.default, 2) or view.args[0] is not clone:
        return None
    vt, vv = _tensor(trans), _tensor(view)
    if vt is None or vv is None:
        return None
    batch, h, tokens, d = vt.shape
    if (batch, tokens) != (b, t) or h * d != width or tuple(vv.shape) != (b, t, h, d):
        return None
    if (
        set(clone.users) != {view}
        or set(view.users) != {trans}
        or set(trans.users) - branch.members
    ):
        return None
    return view, trans


def emit(b, t, d, hq, hk, wv, orders, shared_batch, dt):
    half = d // 2
    wq = hq * d
    wk = hk * d
    width = wq + wk + wv
    nq = b * t * hq * half
    nk = b * t * hk * half
    nv = b * t * wv
    total = nq + nk + nv
    typ = {"bf16": "__nv_bfloat16", "f16": "__half", "f32": "float"}[dt]
    raw = "unsigned int" if dt == "f32" else "unsigned short"
    return f"""#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
using T={typ}; using Bits={raw};
template<int H,int Offset,bool TokenMajor>
__device__ __forceinline__ void rotate_packed(const T* x,const T* c,const T* s,T* y,int i) {{
  const int row=i/{half}, j=i%{half}, batch=row/(H*{t});
  const int token=TokenMajor?(row/H)%{t}:row%{t}, head=TokenMajor?row%H:(row/{t})%H;
  const int p=(({"0" if shared_batch else "batch"}*{t}+token)*{d})+j, a=row*{d}+j;
  const int in=(batch*{t}+token)*{width}+Offset+head*{d}+j;
  const float l=float(x[in]), r=float(x[in+{half}]);
  const T cl=T(l*float(c[p])), sl=T((-r)*float(s[p]));
  const T cr=T(r*float(c[p+{half}])), sr=T(l*float(s[p+{half}]));
  y[a]=T(float(cl)+float(sl)); y[a+{half}]=T(float(cr)+float(sr));
}}
__global__ void packed_rotary_copy(const T* x,const T* c,const T* s,T* q,T* k,Bits* v) {{
  const int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<{nq}) rotate_packed<{hq},0,{str(orders[0]).lower()}>(x,c,s,q,i);
  else if(i<{nq + nk}) rotate_packed<{hk},{wq},{str(orders[1]).lower()}>(x,c,s,k,i-{nq});
  else if(i<{total}) {{const int j=i-{nq + nk}; v[j]=reinterpret_cast<const Bits*>(x)[(j/{wv})*{width}+{wq + wk}+j%{wv}];}}
}}
extern "C" int launch(void** p,int,void* stream) {{
  packed_rotary_copy<<<{(total + 127) // 128},128,0,(cudaStream_t)stream>>>((const T*)p[0],(const T*)p[1],(const T*)p[2],(T*)p[3],(T*)p[4],(Bits*)p[5]);
  return int(cudaGetLastError());
}}
"""


def fuse_packed_rotary(gm, device, host_backend):
    records = []
    for split in list(gm.graph.nodes):
        mat = split_copies(split)
        if mat is None:
            continue
        x, widths, picks = mat
        v = _tensor(x)
        if v.ndim != 3:
            continue
        b, t, _ = v.shape
        ops = list(gm.graph.nodes)
        pos = {n: i for i, n in enumerate(ops)}
        branches = [br for n in ops if (br := _branch(n)) is not None]
        matches = [
            [
                (br, path)
                for br in branches
                if (path := _from_clone(br, picks[i][1], b, t, widths[i])) is not None
            ]
            for i in (0, 1)
        ]
        if any(len(group) != 1 for group in matches):
            continue
        (q, pq), (k, pk) = [m[0] for m in matches]
        _, hq, _, d = _tensor(q.x).shape
        _, hk, _, dk = _tensor(k.x).shape
        if d != dk or q.c is not k.c or q.s is not k.s or q.members & k.members:
            continue
        if any(_tensor(n).device != v.device for n in (q.c, q.s)):
            continue
        vclone = picks[2][1]
        olds = (q.out, k.out, vclone)
        ext = set().union(*(n.users for n in olds))
        if not ext:
            continue
        anchor = min(ext, key=pos.get)
        phases = (q.c, q.s)
        if pos[x] >= pos[anchor]:
            continue
        # Re-create a metadata-only phase view when an early V user precedes the original
        # unsqueeze. In particular, never read an output of this region as its own phase.
        if any(
            pos[z] >= pos[anchor]
            and (
                not _is(z, A.unsqueeze.default, 2)
                or pos[z.args[0]] >= pos[anchor]
                or z.args[0] in olds
            )
            for z in phases
        ) or any(z in olds for z in phases):
            continue
        orders = (q.bshd, k.bshd)
        vs = [_tensor(br.out) for br in (q, k)]
        shapes = [
            tuple(z.transpose(1, 2).shape if sw else z.shape)
            for z, sw in zip(vs, orders, strict=True)
        ] + [tuple(_tensor(vclone).shape)]
        dt = DT[v.dtype]
        shared = _tensor(q.c).shape[0] == 1
        src = emit(b, t, d, hq, hk, widths[2], orders, shared, dt)
        prog = IndexProgram(
            [tuple(_tensor(z).shape) for z in (x, *phases)], shapes, dt, device, src
        )
        label = f"_sparkfuse_packed_rotary_{len(records)}"
        gm.add_module(label, IndexRegion(prog, host_backend))
        group = (
            q.members | k.members | {*pq, *pk, *(n for pair in picks for n in pair), *phases, split}
        )
        with gm.graph.inserting_before(anchor):
            args = [x]
            moved = {}
            for z in phases:
                if pos[z] >= pos[anchor] and z not in moved:
                    zz = gm.graph.call_function(z.target, z.args)
                    zz.meta = dict(z.meta)
                    moved[z] = zz
                args.append(moved.get(z, z))
            call = gm.graph.call_module(label, tuple(args))
            for j, (old, swap) in enumerate(zip(olds, (*orders, False), strict=True)):
                n = gm.graph.call_function(operator.getitem, (call, j))
                if swap:
                    n.meta["val"] = vs[j].transpose(1, 2)
                    n = gm.graph.call_function(A.transpose.int, (n, 1, 2))
                n.meta = dict(old.meta)
                old.replace_all_uses_with(n)
        records.append(
            dict(
                kind="packed_rotary_copy",
                module=label,
                live_ins=[n.name for n in args],
                live_outs=[n.name for n in olds],
                widths=list(widths),
                shape=list(v.shape),
                head_dim=d,
                physical_order=["BSHD" if z else "BHSD" for z in orders],
                shared_phase=shared,
                rematerialized_phase_views=len(moved),
                host_backend=host_backend,
                schedule=dict(threads=128),
                cuda_sha256=sha256(src.encode()).hexdigest(),
            )
        )
        for n in sorted(group, key=pos.get, reverse=True):
            if n.op not in ("placeholder", "get_attr") and not n.users:
                gm.graph.erase_node(n)
    if records:
        gm.graph.lint()
        gm.recompile()
    return records
