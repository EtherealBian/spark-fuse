"""Static split-KV decode kernels. No dense score tensor or repeated KV head allocation."""

import math
from .grouped_decode_codegen import grouped_partial


def emit(b, h, g, s, d, dtype, tile, scale, mask_batch=0, *, head_group=1):
    if not (
        all(type(i) is int for i in (b, h, g, s, d, tile, mask_batch, head_group))
        and type(scale) in (int, float)
        and 1 <= b <= 32
        and 1 <= g <= h <= 256
        and head_group in (1, 2, 4)
        and h % (g * head_group) == 0
        and 1 <= s <= 32768
        and d in (64, 128)
        and tile in (32, 64, 128, 256)
        and dtype in ("bf16", "f16")
        and mask_batch in (0, 1, b)
        and math.isfinite(scale)
        and 0 < scale <= 32
    ):
        raise ValueError("unsupported decode geometry")
    chunks = (s + tile - 1) // tile
    if max(b * g * s * d, b * h * chunks * (d + 2), b * h * d) >= 2**29:
        raise ValueError("unsafe decode index span")
    ty, ld, st = {
        "bf16": ("__nv_bfloat16", "__bfloat162float", "__float2bfloat16_rn"),
        "f16": ("__half", "__half2float", "__float2half_rn"),
    }[dtype]
    defines = f"""#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>
using T={ty};
constexpr int H={h},G={g},S={s},D={d},Tile={tile},C={chunks},NT=128;
__device__ __forceinline__ float ld(T x){{return {ld}(x);}}
__device__ __forceinline__ T rt(float x){{return {st}(x);}}
__device__ __forceinline__ float sumwarp(float x){{
 #pragma unroll
 for(int o=16;o;o>>=1)x+=__shfl_down_sync(0xffffffffu,x,o);
 return x;
}}
"""
    mask = "" if not mask_batch else f"+bias[{'b*S+' if mask_batch > 1 else ''}key]"
    # Each warp computes one query-key dot, then value lanes read adjacent features.
    if head_group != 1:
        a = grouped_partial(defines, b, head_group, scale, mask_batch, mask)
    else:
        a = (
            defines
            + f"""__global__ void decode_partial(const T* q,const T* k,const T* v,const float* bias,void* dst){{
 const int i=threadIdx.x,l=i&31,w=i>>5,h=blockIdx.y,b=blockIdx.z,chunk=blockIdx.x;
 const int begin=chunk*Tile, count=min(Tile,S-begin),kv=((b*G+h/(H/G))*S)*D,qi=(b*H+h)*D;
 __shared__ float scores[Tile],rowmax,mass;
 float qreg[D/32];
 #pragma unroll
 for(int j=0;j<D/32;++j)qreg[j]=ld(q[qi+j*32+l]);
 for(int t=w;t<count;t+=NT/32){{
   int key=begin+t;float dot=0.f;
   #pragma unroll
   for(int j=0;j<D/32;++j)dot=fmaf(qreg[j],ld(k[kv+key*D+j*32+l]),dot);
   dot=sumwarp(dot);
   if(l==0)scores[t]=__fmul_rn(dot,{float(scale).hex()}f){mask};
 }}
 __syncthreads();
 if(w==0){{float mx=-INFINITY;for(int t=l;t<count;t+=32)mx=fmaxf(mx,scores[t]);
 #pragma unroll
 for(int o=16;o;o>>=1)mx=fmaxf(mx,__shfl_down_sync(0xffffffffu,mx,o));
 if(l==0)rowmax=mx;}}
 __syncthreads(); float mx=rowmax;
 for(int t=i;t<count;t+=NT) scores[t]=(mx==-INFINITY ? 0.f : expf(scores[t]-mx));
 __syncthreads();
 if(w==0){{float sm=0.f;for(int t=l;t<count;t+=32)sm+=scores[t];sm=sumwarp(sm);if(l==0)mass=sm;}}
 float y=0.f;if(i<D){{for(int t=0;t<count;++t)y=fmaf(scores[t],ld(v[kv+(begin+t)*D+i]),y);}}
 __syncthreads(); const int base=((b*H+h)*C+chunk)*(D+2);
 if constexpr(C==1){{if(i<D)((T*)dst)[qi+i]=rt(mass==0.f ? 0.f : y/mass);}}
 else{{if(i<D)((float*)dst)[base+i]=y;if(i==0){{((float*)dst)[base+D]=mx;((float*)dst)[base+D+1]=mass;}}}}
}}
extern "C" int launch(void** p,int,void* stream){{decode_partial<<<dim3(C,H,{b}),NT,0,(cudaStream_t)stream>>>((T*)p[0],(T*)p[1],(T*)p[2],{("(float*)p[3]" if mask_batch else "nullptr")},p[{4 if mask_batch else 3}]);return int(cudaGetLastError());}}
"""
        )
    if chunks == 1:
        return a, None
    m = (
        defines
        + f"""__global__ void decode_merge(const float* partial,T* out){{
 const int i=threadIdx.x,bh=blockIdx.x, stride=D+2;const float* p=partial+bh*C*stride;
 __shared__ float rowmax,total;__shared__ float weights[C];
 if(i<32){{float mx=-INFINITY;for(int c=i;c<C;c+=32)mx=fmaxf(mx,p[c*stride+D]);
 #pragma unroll
 for(int o=16;o;o>>=1)mx=fmaxf(mx,__shfl_down_sync(0xffffffffu,mx,o));if(i==0)rowmax=mx;}}
 __syncthreads();
 for(int c=i;c<C;c+=NT)weights[c]=(rowmax==-INFINITY?0.f:expf(p[c*stride+D]-rowmax));
 __syncthreads();
 if(i<32){{float sm=0.f;for(int c=i;c<C;c+=32)sm=fmaf(p[c*stride+D+1],weights[c],sm);sm=sumwarp(sm);if(!i)total=sm;}}
 float y=0.f;if(i<D)for(int c=0;c<C;++c)y=fmaf(p[c*stride+i],weights[c],y);
 __syncthreads();if(i<D)out[bh*D+i]=rt(total==0.f?0.f:y/total);
}}
extern "C" int launch(void** p,int,void* stream){{decode_merge<<<{b * h},NT,0,(cudaStream_t)stream>>>((float*)p[0],(T*)p[1]);return int(cudaGetLastError());}}
"""
    )
    return a, m
