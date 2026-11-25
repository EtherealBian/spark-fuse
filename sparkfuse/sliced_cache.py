"""Fuse Q/K partial RoPE with two functional cache appends. Preserve public packed views.

Q is fresh, K is private to its cat, and the cache outputs remain fresh BH(S+T)D.
Positive-stride read-only history uses exact input-stride descriptors; there is no
backing-storage reinterpretation, prepacking, cache mutation or capacity assumption.
"""

from hashlib import sha256
from math import prod
import operator

import torch
from .index_program import IndexProgram, IndexRegion
from .ir import DT
from .rotary import _is, _tensor
from .sliced_rotary import _last_slice, _part, cuda_header, head_function

A = torch.ops.aten


def _append(n):
    if (
        not _is(n, A.cat.default, 2)
        or n.args[1] not in (-2, 2)
        or not isinstance(n.args[0], (tuple, list))
        or len(n.args[0]) != 2
    ):
        return None
    x, z = n.args[0]
    vals = [_tensor(v) for v in (x, z, n)]
    if any(v is None or v.ndim != 4 for v in vals):
        return None
    old, new, out = vals
    b, h, p, d = old.shape
    t = new.shape[2]
    if (
        old.stride(-1) != 1
        or max(old.stride()) >= 2**31
        or sum((n - 1) * s for n, s in zip(old.shape, old.stride())) >= 2**29
        or tuple(new.shape) != (b, h, t, d)
        or tuple(out.shape) != (b, h, p + t, d)
        or prod(out.shape) > 2**29
        or tuple(out.stride()) != (h * (p + t) * d, (p + t) * d, d, 1)
        or any(v.dtype != old.dtype or v.device != old.device for v in (new, out))
    ):
        return None
    return x, z


def _value(v, wide, h, d):
    if not _is(v, A.transpose.int, 3) or v.args[1:] not in ((1, 2), (2, 1)):
        return None
    vw = v.args[0]
    if not _is(vw, A.view.default, 2):
        return None
    sl = vw.args[0]
    a = _last_slice(sl, 3)
    if a is None or a[0] is not wide:
        return None
    x = _tensor(wide)
    meta = _tensor(vw)
    if (
        meta is None
        or tuple(meta.shape) != (*x.shape[:2], h, d)
        or min(a[2], x.shape[2]) - a[1] != h * d
    ):
        return None
    return a[1], {sl, vw, v}


def emit(b, t, w, d, r, hq, hk, offsets, p, ks, vs, shared, dt, vectorized=True):
    oq, ok, ov = offsets
    elem = 4 if dt == "f32" else 2
    pack = next(v for v in (8, 4, 2, 1) if (v == 1 or vectorized) and d % v == 0 and v * elem <= 16)
    typ = {2: "unsigned short", 4: "unsigned int", 8: "uint2", 16: "uint4"}[pack * elem]
    lanes = d - r // 2
    nq = b * hq * t * lanes
    nk = b * hk * t * lanes
    nv = b * hk * t * d
    no = b * hk * p * (d // pack)
    bq, bk, bv, bo = ((n + 127) // 128 for n in (nq, nk, nv, no))

    def triple(st):
        return ",".join(map(str, st[:3]))

    src = (
        "#include <stdint.h>\n"
        + cuda_header(dt)
        + head_function(t, w, d, r, shared, name="query")
        + head_function(t, w, d, r, shared, name="key", past=p)
    )
    src += f"""using Packet={typ};
template<int BS,int HS,int TS> __device__ __forceinline__ void history(const Bits*x,Bits*y,int i) {{
  constexpr int W={pack}, D={d}, S={p}, T={t}, H={hk}, PK=D/W;
  if(i>={no})return;
  const int row=i/PK, token=row%S, head=(row/S)%H, batch=row/(H*S), col=(i%PK)*W;
  const Bits* from=x+batch*BS+head*HS+token*TS+col;
  Bits* to=y+((batch*H+head)*(S+T)+token)*D+col;
  if(((uintptr_t(from)|uintptr_t(to)) & (sizeof(Packet)-1))==0)
    *reinterpret_cast<Packet*>(to)=*reinterpret_cast<const Packet*>(from);
  else {{
    #pragma unroll
    for(int j=0;j<W;++j)to[j]=from[j];
  }}
}}
__global__ void sliced_cache_update(const T*x,const T*c,const T*s,const Bits*pk,const Bits*pv,T*q,T*k,Bits*v) {{
  int block=blockIdx.x, i=block*128+threadIdx.x;
  if(block<{bq}){{if(i<{nq})query<{hq},{oq}>(x,c,s,q,i);return;}}
  if((block-={bq})<{bk}){{i=block*128+threadIdx.x;if(i<{nk})key<{hk},{ok}>(x,c,s,k,i);return;}}
  if((block-={bk})<{bv}){{
    i=block*128+threadIdx.x;
    if(i<{nv}){{const int row=i/{d},col=i%{d},token=row%{t},head=(row/{t})%{hk},batch=row/({t}*{hk});
      v[((batch*{hk}+head)*{p + t}+{p}+token)*{d}+col]=reinterpret_cast<const Bits*>(x)[(batch*{t}+token)*{w}+{ov}+head*{d}+col];}}
    return;
  }}
  if((block-={bv})<{bo}){{history<{triple(ks)}>(pk,reinterpret_cast<Bits*>(k),block*128+threadIdx.x);return;}}
  history<{triple(vs)}>(pv,v,(block-{bo})*128+threadIdx.x);
}}
extern "C" int launch(void**p,int,void*stream){{
  sliced_cache_update<<<{bq + bk + bv + 2 * bo},128,0,(cudaStream_t)stream>>>((const T*)p[0],(const T*)p[1],(const T*)p[2],(const Bits*)p[3],(const Bits*)p[4],(T*)p[5],(T*)p[6],(Bits*)p[7]);
  return int(cudaGetLastError());
}}
"""
    return src, pack


def _fuse_one(gm, dev, host, ordinal):
    pos = {n: i for i, n in enumerate(gm.graph.nodes)}
    parts = [p for n in pos if (p := _part(n)) is not None]
    cats = {n: a for n in pos if (a := _append(n)) is not None}
    for k in parts:
        if len(k.out.users) != 1:
            continue
        kc = next(iter(k.out.users))
        if kc not in cats or cats[kc][1] is not k.out:
            continue
        pk = cats[kc][0]
        past = _tensor(pk).shape[2]
        x = _tensor(k.wide)
        b, t, w = x.shape
        for vc, (pv, v) in cats.items():
            if (
                vc is kc
                or _tensor(pv).shape != _tensor(pk).shape
                or (val := _value(v, k.wide, k.heads, k.dim)) is None
            ):
                continue
            ov, vnodes = val
            for q in parts:
                if q is k or not (
                    q.wide is k.wide
                    and q.cos is k.cos
                    and q.sin is k.sin
                    and q.dim == k.dim
                    and q.rotary == k.rotary
                ):
                    continue
                seg = sorted(
                    (off, off + h * k.dim)
                    for off, h in ((q.start, q.heads), (k.start, k.heads), (ov, k.heads))
                )
                if any(a[1] > z[0] for a, z in zip(seg, seg[1:])):
                    continue
                outs = (q.out, kc, vc)
                grp = q.members | k.members | vnodes | {kc, vc}
                args = (k.wide, k.cos, k.sin, pk, pv)
                users = set().union(*(n.users for n in outs))
                if not all(n.users for n in outs) or users & grp or set(args) & grp:
                    continue
                anchor = min(users, key=pos.get)
                if any(pos[a] >= pos[anchor] for a in args):
                    continue
                dt = DT[x.dtype]
                st = [tuple(_tensor(z).stride()) for z in (pk, pv)]
                offsets = (q.start, k.start, ov)
                code, packet = emit(
                    b,
                    t,
                    w,
                    k.dim,
                    k.rotary,
                    q.heads,
                    k.heads,
                    offsets,
                    past,
                    *st,
                    _tensor(k.cos).shape[0] == 1,
                    dt,
                )
                pg = IndexProgram(
                    [tuple(_tensor(z).shape) for z in args],
                    [tuple(_tensor(z).shape) for z in outs],
                    dt,
                    dev,
                    code,
                    input_strides=(None, None, None, *st),
                )
                label = f"_sparkfuse_sliced_cache_{ordinal}"
                gm.add_module(label, IndexRegion(pg, host))
                with gm.graph.inserting_before(anchor):
                    call = gm.graph.call_module(label, args)
                    for j, n in enumerate(outs):
                        y = gm.graph.call_function(operator.getitem, (call, j))
                        y.meta = dict(n.meta)
                        n.replace_all_uses_with(y)
                for old in sorted(grp, key=pos.get, reverse=True):
                    if not old.users:
                        gm.graph.erase_node(old)
                return dict(
                    kind="sliced_cache_update",
                    module=label,
                    shape=list(x.shape),
                    prefix=past,
                    heads=[q.heads, k.heads],
                    head_dim=k.dim,
                    rotary_dim=k.rotary,
                    offsets=list(offsets),
                    history_strides=[list(z) for z in st],
                    live_ins=[z.name for z in args],
                    live_outs=[z.name for z in outs],
                    host_backend=host,
                    schedule=dict(threads=128, max_packet_elements=packet),
                    cuda_sha256=sha256(code.encode()).hexdigest(),
                )
    return None


def fuse_sliced_cache(gm, device, host_backend):
    records = []
    while (r := _fuse_one(gm, device, host_backend, len(records))) is not None:
        records.append(r)
    if records:
        gm.graph.lint()
        gm.recompile()
    return records
