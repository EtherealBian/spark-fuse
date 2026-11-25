"""Packed narrow residual storage for an explicit, ordered RMSNorm expression."""

from dataclasses import dataclass
import math
from string import Template

from .ir import Unsupported
from .mean_tree import cohort_eligible
from .vector import HEADER


@dataclass(frozen=True)
class Roles:
    x: int
    skip: int
    weight: int
    result: int
    residual: int | None
    dtype: str
    epsilon: float


def match(p):
    """Return physical argument positions, not FX names or assumed placeholder order."""
    if (
        not cohort_eligible(p)
        or len(p.domain) not in (2, 3)
        or len(p.inputs) != 3
        or len(p.vals) != 13
    ):
        return None
    # Keep every leading extent for legality; only launch indexing flattens the rows.
    vs, full, row = p.vals, p.domain, (*p.domain[:-1], 1)
    (sq,) = vs[p.red].deps
    if vs[sq].op != "square" or vs[sq].dt != "f32":
        return None
    (wide,) = vs[sq].deps
    if vs[wide].op != "cast" or (vs[wide].shape, vs[wide].dt) != (full, "f32"):
        return None
    (add,) = vs[wide].deps
    dt = vs[add].dt
    if vs[add].op != "add" or dt not in ("f16", "bf16") or vs[add].shape != full:
        return None
    x, skip = vs[add].deps
    if x == skip or any(
        i not in p.inputs or (vs[i].shape, vs[i].dt) != (full, dt) for i in (x, skip)
    ):
        return None
    (weight,) = set(p.inputs) - {x, skip}
    if (vs[weight].shape, vs[weight].dt) != ((p.width,), dt):
        return None
    eps = next(
        (i for i, v in enumerate(vs) if v.op == "const" and v.dt == "f32" and v.shape == ()), None
    )
    if eps is None or not math.isfinite(vs[eps].imm) or vs[eps].imm < 0:
        return None

    def take(op, shape, dt, *deps):
        return next(
            (
                i
                for i, v in enumerate(vs)
                if (v.op, v.shape, v.dt) == (op, shape, dt)
                and (v.deps == deps or (op in ("mul", "add") and v.deps == deps[::-1]))
            ),
            None,
        )

    adjusted = take("add", row, "f32", p.red, eps)
    inv = take("rsqrt", row, "f32", adjusted)
    norm = take("mul", full, "f32", wide, inv)
    narrow = take("cast", full, dt, norm)
    result = take("mul", full, dt, narrow, weight)
    if result is None or result not in p.outputs or set(p.outputs) - {result, add}:
        return None
    return Roles(
        p.inputs.index(x),
        p.inputs.index(skip),
        p.inputs.index(weight),
        p.outputs.index(result),
        p.outputs.index(add) if add in p.outputs else None,
        dt,
        vs[eps].imm,
    )


_CUDA = Template(r"""
using T = $narrow;
// Fast path is selected once per launch; the safe path also handles mixed alignments.
template<int V, bool Aligned> __device__ __forceinline__ Pack<T,V> read_pack(const T* p){
    if constexpr (Aligned) return *reinterpret_cast<const Pack<T,V>*>(p);
    return loadv<T,V>(p);
}
template<int V, bool Aligned> __device__ __forceinline__ void write_pack(T* p, Pack<T,V> value){
    if constexpr (Aligned) *reinterpret_cast<Pack<T,V>*>(p) = value;
    else storev<T,V>(p,value);
}
template<bool Aligned> __global__ void fused_packed_mean(
    const T* x, const T* skip, const T* weight, T* y$extra_param){
    constexpr int H=$h, V=$v, NT=$t;
    __shared__ __align__(16) T tile[H];
    __shared__ float groups[32];
    const int t=threadIdx.x, base=blockIdx.x*H, lane=t&31;
    for(int j=t*V; j<H; j+=NT*V){
        auto a=read_pack<V,Aligned>(x+base+j), b=read_pack<V,Aligned>(skip+base+j);
        Pack<T,V> z;
        #pragma unroll
        for(int k=0; k<V; k+=2){
            auto r=__hadd2($pair(a.x[k],a.x[k+1]),$pair(b.x[k],b.x[k+1]));
            z.x[k]=$lo(r); z.x[k+1]=$hi(r);
        }
        write_pack<V,true>(tile+j,z);
        $extra_write
    }
    __syncthreads();
    // The materialized eager mean has 128 interleaved FP32 chains in this size regime.
    if(t<128){
        float sum=0.f;
        #pragma unroll 4
        for(int j=t; j<H; j+=128){
            float f=$wide(tile[j]); sum=__fadd_rn(sum,__fmul_rn(f,f));
        }
        float b=__shfl_down_sync(~0u,sum,1), c=__shfl_down_sync(~0u,sum,2), d=__shfl_down_sync(~0u,sum,3);
        if(!(t&3)) groups[t/4]=__fadd_rn(__fadd_rn(__fadd_rn(sum,b),c),d);
    }
    __syncthreads();
    // Each warp independently finishes the same 32-group tree; no third CTA barrier.
    float mu=groups[lane];
    for(int d=16; d; d>>=1) mu=__fadd_rn(mu,__shfl_down_sync(~0u,mu,d));
    mu=__shfl_sync(~0u,mu,0);
    float inv=rsqrtf(__fadd_rn(__fmul_rn(mu,1.f/H),$eps));
    for(int j=t*V; j<H; j+=V*NT){
        auto a=read_pack<V,true>(tile+j), b=read_pack<V,Aligned>(weight+j);
        Pack<T,V> z;
        #pragma unroll
        for(int k=0; k<V; k+=2){
            auto f=$round2(__fmul_rn(inv,$wide(a.x[k])),__fmul_rn(inv,$wide(a.x[k+1])));
            auto g=__hmul2(f,$pair(b.x[k],b.x[k+1]));
            z.x[k]=$lo(g); z.x[k+1]=$hi(g);
        }
        write_pack<V,Aligned>(y+base+j,z);
    }
}
extern "C" int launch(void** p,int rows,void* stream){
    uintptr_t bits=$bits;
    if((bits & ($v*sizeof(T)-1))==0)
        fused_packed_mean<true><<<rows,$t,0,(cudaStream_t)stream>>>($args);
    else fused_packed_mean<false><<<rows,$t,0,(cudaStream_t)stream>>>($args);
    return (int)cudaGetLastError();
}
""")


def generate_packed_mean(p, s):
    m = match(p)
    if m is None:
        raise Unsupported(
            "mean_packed_tile: expected an explicit residual/FP32 mean/cast/weight graph"
        )
    roles = [m.x, m.skip, m.weight, len(p.inputs) + m.result]
    visible = m.residual is not None
    if visible:
        roles.append(len(p.inputs) + m.residual)
    ops = (
        (
            "__nv_bfloat16",
            "__halves2bfloat162",
            "__low2bfloat16",
            "__high2bfloat16",
            "__bfloat162float",
            "__floats2bfloat162_rn",
        )
        if m.dtype == "bf16"
        else (
            "__half",
            "__halves2half2",
            "__low2half",
            "__high2half",
            "__half2float",
            "__floats2half2_rn",
        )
    )
    return (
        HEADER
        + "\n"
        + _CUDA.substitute(
            **dict(zip(("narrow", "pair", "lo", "hi", "wide", "round2"), ops, strict=True)),
            h=p.width,
            v=s.vec,
            t=s.threads,
            eps=float(m.epsilon).hex() + "f",
            extra_param=", T* residual" if visible else "",
            extra_write="write_pack<V,Aligned>(residual+base+j,z);"
            if visible
            else "// No globally visible residual.",
            bits=" | ".join(f"uintptr_t(p[{i}])" for i in roles),
            args=", ".join(f"(T*)p[{i}]" for i in roles),
        )
    )
