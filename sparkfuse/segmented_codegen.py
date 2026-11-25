"""Consume old/new K/V in place. Only the first query head for each KV head publishes caches."""

import math


def emit_segmented(plan, dtype, strides, mask_dtype, mask_batch, publish):
    b, h, g, s, d, t, c = (
        plan.batch,
        plan.heads,
        plan.kv_heads,
        plan.sequence,
        plan.width,
        plan.tile,
        plan.chunks,
    )
    if not (
        1 <= b <= 32
        and 1 <= g <= h <= 256
        and h % g == 0
        and 2 <= s <= 32768
        and d in (64, 128)
        and t in (32, 64, 128, 256)
        and c == math.ceil(s / t)
        and plan.head_group == 1
        and dtype in ("f16", "bf16")
        and mask_batch in (0, 1, b)
        and mask_dtype in (None, "f32", dtype)
        and bool(mask_batch) == bool(mask_dtype)
        and len(publish) == 2
        and all(type(x) is bool for x in publish)
        and len(strides) == 4
        and 0 < plan.scale <= 32
        and math.prod((b, h, s, d)) < 2**29
    ):
        raise ValueError("unsupported segmented decode schedule")
    for i, st in enumerate(strides):
        shape = (b, g, 1 if i % 2 else s - 1, d)
        if (
            len(st) != 4
            or st[-1] != 1
            or not all(type(v) is int and 0 < v < 2**31 for v in st)
            or sum((n - 1) * v for n, v in zip(shape, st)) >= 2**29
        ):
            raise ValueError("unsupported segmented cache stride")
    ty, read, rounding = {
        "bf16": ("__nv_bfloat16", "__bfloat162float", "__float2bfloat16_rn"),
        "f16": ("__half", "__half2float", "__float2half_rn"),
    }[dtype]
    bias = "float" if mask_dtype == "f32" else "T"
    mask = (
        ""
        if not mask_batch
        else ("+bias[" if mask_dtype == "f32" else "+ld(bias[")
        + ("b*S+" if mask_batch > 1 else "")
        + "key]"
        + ("" if mask_dtype == "f32" else ")")
    )
    base = 5 + bool(mask_batch)

    def addr(i):
        sb, sg, ss, _ = strides[i]
        return f"b*{sb}+g*{sg}+" + (f"key*{ss}+" if i % 2 == 0 else "") + "j"

    def value(i, old, new):
        return f"key<S-1 ? {old}[{addr(i)}] : {new}[{addr(i + 1)}]"

    return f"""#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>
using T={ty};constexpr int H={h},G={g},S={s},D={d},Tile={t},C={c},NT=128;
__device__ __forceinline__ float ld(T x){{return {read}(x);}}
__device__ __forceinline__ T rt(float x){{return {rounding}(x);}}
__device__ __forceinline__ float sumwarp(float x){{
 #pragma unroll
 for(int o=16;o;o>>=1)x+=__shfl_down_sync(0xffffffffu,x,o);return x;}}
__global__ void segmented_partial(const T* q,const T* k0,const T* k1,const T* v0,const T* v1,const {bias}* bias,void* dst,T* ko,T* vo){{
 const int i=threadIdx.x,l=i&31,w=i>>5,h=blockIdx.y,b=blockIdx.z,c=blockIdx.x;
 const int g=h/(H/G),begin=c*Tile,count=min(Tile,S-begin),qi=(b*H+h)*D,basekv=(b*G+g)*S*D;
 const bool owner=h%(H/G)==0;
 __shared__ float score[Tile],rowmax,mass;float qr[D/32];
 #pragma unroll
 for(int j=0;j<D/32;++j)qr[j]=ld(q[qi+j*32+l]);
 for(int t=w;t<count;t+=NT/32){{int key=begin+t;float dot=0.f;
 #pragma unroll
 for(int u=0;u<D/32;u++){{int j=u*32+l;T val={value(0, "k0", "k1")};dot=fmaf(qr[u],ld(val),dot);
 {"if(owner)ko[basekv+key*D+j]=val;" if publish[0] else ""}}}
 dot=sumwarp(dot);if(l==0)score[t]=__fmul_rn(dot,{float(plan.scale).hex()}f){mask};}}
 __syncthreads();if(w==0){{float mx=-INFINITY;for(int t=l;t<count;t+=32)mx=fmaxf(mx,score[t]);
 #pragma unroll
 for(int o=16;o;o>>=1)mx=fmaxf(mx,__shfl_down_sync(0xffffffffu,mx,o));if(!l)rowmax=mx;}}
 __syncthreads();for(int t=i;t<count;t+=NT)score[t]=(rowmax==-INFINITY?0.f:expf(score[t]-rowmax));
 __syncthreads();if(w==0){{float sm=0.f;for(int t=l;t<count;t+=32)sm+=score[t];sm=sumwarp(sm);if(!l)mass=sm;}}
 float acc=0.f;if(i<D){{int j=i;for(int t=0;t<count;t++){{int key=begin+t;T val={value(2, "v0", "v1")};acc=fmaf(score[t],ld(val),acc);
 {"if(owner)vo[basekv+key*D+j]=val;" if publish[1] else ""}}}}}
 __syncthreads();const int base=((b*H+h)*C+c)*(D+2);
 if constexpr(C==1){{if(i<D)((T*)dst)[qi+i]=rt(mass==0.f?0.f:acc/mass);}}
 else{{if(i<D)((float*)dst)[base+i]=acc;if(!i){{((float*)dst)[base+D]=rowmax;((float*)dst)[base+D+1]=mass;}}}}
}}
extern "C" int launch(void** p,int,void* stream){{segmented_partial<<<dim3(C,H,{b}),NT,0,(cudaStream_t)stream>>>((T*)p[0],(T*)p[1],(T*)p[2],(T*)p[3],(T*)p[4],{f"({bias}*)p[5]" if mask_batch else "nullptr"},p[{base}],{f"(T*)p[{base + 1}]" if publish[0] else "nullptr"},{f"(T*)p[{base + 1 + int(publish[0])}]" if publish[1] else "nullptr"});return int(cudaGetLastError());}}
"""
