"""Static residual + signed symmetric per-token activation quantization (opt in)."""

from dataclasses import dataclass
from hashlib import sha256
from math import prod
import operator

import torch
from torch import fx

from .index_program import IndexProgram, IndexRegion
from .ir import DT
from .rotary import _is

A = torch.ops.aten


def _dense(n, shape, dtype, dev):
    v = n.meta.get("val") if isinstance(n, fx.Node) else None
    return (
        isinstance(v, torch.Tensor)
        and v.layout == torch.strided
        and v.dtype is dtype
        and v.device == dev
        and not v.is_conj()
        and not v.is_neg()
        and all(type(k) is int for k in (*v.shape, *v.stride()))
        and tuple(v.shape) == shape
        and tuple(v.stride()) == tuple(prod(shape[i + 1 :]) for i in range(len(shape)))
    )


def _num(x, value):
    return type(x) in (float, int) and x == value


@dataclass
class Site:
    inputs: tuple
    members: tuple
    guards: list
    outs: tuple
    roles: tuple
    shape: tuple
    dtype: str


def candidate(q, dev):
    if not (_is(q, A.to.dtype, 2) and q.args[1] is torch.int8 and q.users):
        return None
    c = q.args[0]
    if not (_is(c, A.clamp.default, 3) and _num(c.args[1], -127) and _num(c.args[2], 127)):
        return None
    t = c.args[0]
    if not _is(t, A.round.default, 1):
        return None
    d = t.args[0]
    if not _is(d, A.div.Tensor, 2):
        return None
    f, s = d.args
    if not (
        _is(s, A.div.Tensor, 2)
        and _num(s.args[1], 127)
        and _is(f, A.to.dtype, 2)
        and f.args[1] is torch.float32
    ):
        return None
    low, z = s.args[0], f.args[0]
    if not (
        _is(z, A.add.Tensor, 2) and _is(low, A.clamp_min.default, 2) and _num(low.args[1], 1.0e-5)
    ):
        return None
    peak = low.args[0]
    if not (_is(peak, A.amax.default, 3) and peak.args[2] is True):
        return None
    absolute, dims, _ = peak.args
    if not (_is(absolute, A.abs.default, 1) and absolute.args[0] is f):
        return None
    v = z.meta.get("val")
    if (
        not isinstance(v, torch.Tensor)
        or v.dtype not in (torch.float16, torch.bfloat16)
        or v.ndim not in (2, 3)
    ):
        return None
    shape = tuple(v.shape)
    if (
        not all(type(k) is int and k > 0 for k in shape)
        or prod(shape) > 2**28
        or not 1 <= shape[-1] <= 8192
        or not isinstance(dims, (tuple, list))
        or len(dims) != 1
        or type(dims[0]) is not int
        or dims[0] not in (-1, len(shape) - 1)
    ):
        return None
    if (
        not all(_dense(n, shape, v.dtype, dev) for n in (*z.args, z))
        or not all(_dense(n, shape, torch.float32, dev) for n in (f, absolute, d, t, c))
        or not all(_dense(n, (*shape[:-1], 1), torch.float32, dev) for n in (peak, low, s))
        or not _dense(q, shape, torch.int8, dev)
    ):
        return None
    members = (z, f, absolute, peak, low, s, d, t, c, q)
    guards, escapes = [], set()
    for n in members:
        for u in n.users:
            if u in members:
                continue
            if u.op == "call_function" and u.target is A._assert_tensor_metadata.default:
                m = n.meta["val"]
                if (
                    u.args != (n,)
                    or u.users
                    or dict(u.kwargs) != dict(dtype=m.dtype, device=m.device, layout=m.layout)
                ):
                    return None
                guards.append(u)
            elif n in (z, q, s):
                escapes.add(n)
            else:
                return None
    if q not in escapes:
        return None
    live = [(n, role) for n, role in ((z, "residual"), (q, "codes"), (s, "scale")) if n in escapes]
    return Site(
        z.args,
        members,
        guards,
        tuple(n for n, _ in live),
        tuple(k for _, k in live),
        shape,
        DT[v.dtype],
    )


def emit(site):
    rows, w = prod(site.shape[:-1]), site.shape[-1]
    ty, load, store = (
        ("__nv_bfloat16", "__bfloat162float", "__float2bfloat16_rn")
        if site.dtype == "bf16"
        else ("__half", "__half2float", "__float2half_rn")
    )
    sig = {"residual": "N* z", "codes": "signed char* q", "scale": "float* so"}
    ptr = {"residual": "N*", "codes": "signed char*", "scale": "float*"}
    params = ", ".join(sig[k] for k in site.roles)
    args = ", ".join(f"({ptr[k]})p[{i + 2}]" for i, k in enumerate(site.roles))
    return f"""#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
using N = {ty};
constexpr int W = {w}, T = 128;
__global__ void residual_token_quant(const N* x, const N* r, {params}) {{
    const int base=int(blockIdx.x)*W, lane=threadIdx.x%32, warp=threadIdx.x/32;
    __shared__ unsigned peaks[T/32];
    __shared__ float scale;
    unsigned peak=0;
    N keep[(W+T-1)/T];
    #pragma unroll
    for (int k=0; k<(W+T-1)/T; ++k) {{
        int i=threadIdx.x+k*T;
        if (i<W) {{
            N a={store}({load}(x[base+i])+{load}(r[base+i]));
            keep[k]=a;
            {"z[base+i]=a;" if "residual" in site.roles else ""}
            float f={load}(a);
            peak=max(peak, isnan(f)?0x7fffffffu:(__float_as_uint(f)&0x7fffffffu));
        }}
    }}
    #pragma unroll
    for(int d=16;d;d>>=1) peak=max(peak,__shfl_down_sync(0xffffffff,peak,d));
    if(!lane) peaks[warp]=peak;
    __syncthreads();
    if(!warp) {{
        peak=lane<T/32?peaks[lane]:0;
        #pragma unroll
        for(int d=16;d;d>>=1) peak=max(peak,__shfl_down_sync(0xffffffff,peak,d));
        if(!lane) {{
            float f=__uint_as_float(peak);
            scale=isnan(f)?f:fmaxf(f,1.e-5f)*(1.f/127.f);
            {"so[blockIdx.x]=scale;" if "scale" in site.roles else ""}
        }}
    }}
    __syncthreads();
    #pragma unroll
    for(int k=0;k<(W+T-1)/T;++k) {{
        int i=threadIdx.x+k*T;
        if(i<W) {{
            float a=__fdiv_rn({load}(keep[k]),scale);
            int code=isnan(a)?0:__float2int_rn(a);
            q[base+i]=(signed char)min(max(code,-127),127);
        }}
    }}
}}
extern "C" int launch(void** p,int,void* stream) {{
    residual_token_quant<<<{rows},T,0,(cudaStream_t)stream>>>((const N*)p[0],(const N*)p[1],{args});
    return int(cudaGetLastError());
}}
"""


def fuse_token_quant(gm, device, host_backend):
    records = []
    dev = torch.device(device)
    if dev.type != "cuda":
        return records
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    for q in list(gm.graph.nodes):
        site = candidate(q, dev)
        if site is None:
            continue
        source = emit(site)
        types = {"codes": "i8", "scale": "f32", "residual": site.dtype}
        prog = IndexProgram(
            [site.shape] * 2,
            [tuple(n.meta["val"].shape) for n in site.outs],
            site.dtype,
            dev,
            source,
            output_dtypes=[types[k] for k in site.roles],
        )
        label = f"_sparkfuse_token_quant_{len(records)}"
        gm.add_module(label, IndexRegion(prog, host_backend))
        with gm.graph.inserting_before(site.members[0]):
            call = gm.graph.call_module(label, site.inputs)
            for i, n in enumerate(site.outs):
                y = gm.graph.call_function(operator.getitem, (call, i))
                y.meta = dict(n.meta)
                n.replace_all_uses_with(y)
        for n in (*site.guards, *reversed(site.members)):
            gm.graph.erase_node(n)
        records.append(
            dict(
                kind="residual_token_quant",
                module=label,
                live_ins=[n.name for n in site.inputs],
                live_outs=[n.name for n in site.outs],
                output_roles=list(site.roles),
                ops=[n.name for n in site.members],
                dtype=site.dtype,
                shape=list(site.shape),
                host_backend=host_backend,
                scale_floor=1.0e-5,
                divisor=127,
                schedule=dict(threads=128, rows_per_block=1),
                cuda_sha256=sha256(source.encode()).hexdigest(),
            )
        )
    if records:
        gm.graph.lint()
        gm.recompile()
    return records
