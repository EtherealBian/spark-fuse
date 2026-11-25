#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
template<class T,int V> struct alignas(sizeof(T)*V) Pack { T x[V]; };
template<class T,int V> __device__ __forceinline__ Pack<T,V> loadv(const T* p){
  using P=Pack<T,V>;
  if((uintptr_t(p)&(alignof(P)-1))==0) return *reinterpret_cast<const P*>(p);
  P q;
  #pragma unroll
  for(int i=0;i<V;i++)q.x[i]=p[i]; return q;
}
template<class T,int V> __device__ __forceinline__ void storev(T* p,Pack<T,V> q){
  using P=Pack<T,V>;
  if((uintptr_t(p)&(alignof(P)-1))==0){ *reinterpret_cast<P*>(p)=q; return; }
  #pragma unroll
  for(int i=0;i<V;i++)p[i]=q.x[i];
}

using T = __nv_bfloat16;
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
    const T* x, const T* skip, const T* weight, T* y){
    constexpr int H=6144, V=4, NT=256;
    __shared__ __align__(16) T tile[H];
    __shared__ float groups[32];
    const int t=threadIdx.x, base=blockIdx.x*H, lane=t&31;
    for(int j=t*V; j<H; j+=NT*V){
        auto a=read_pack<V,Aligned>(x+base+j), b=read_pack<V,Aligned>(skip+base+j);
        Pack<T,V> z;
        #pragma unroll
        for(int k=0; k<V; k+=2){
            auto r=__hadd2(__halves2bfloat162(a.x[k],a.x[k+1]),__halves2bfloat162(b.x[k],b.x[k+1]));
            z.x[k]=__low2bfloat16(r); z.x[k+1]=__high2bfloat16(r);
        }
        write_pack<V,true>(tile+j,z);
        // No globally visible residual.
    }
    __syncthreads();
    // The materialized eager mean has 128 interleaved FP32 chains in this size regime.
    if(t<128){
        float sum=0.f;
        #pragma unroll 4
        for(int j=t; j<H; j+=128){
            float f=__bfloat162float(tile[j]); sum=__fadd_rn(sum,__fmul_rn(f,f));
        }
        float b=__shfl_down_sync(~0u,sum,1), c=__shfl_down_sync(~0u,sum,2), d=__shfl_down_sync(~0u,sum,3);
        if(!(t&3)) groups[t/4]=__fadd_rn(__fadd_rn(__fadd_rn(sum,b),c),d);
    }
    __syncthreads();
    // Each warp independently finishes the same 32-group tree; no third CTA barrier.
    float mu=groups[lane];
    for(int d=16; d; d>>=1) mu=__fadd_rn(mu,__shfl_down_sync(~0u,mu,d));
    mu=__shfl_sync(~0u,mu,0);
    float inv=rsqrtf(__fadd_rn(__fmul_rn(mu,1.f/H),0x1.0c6f7a0b5ed8dp-20f));
    for(int j=t*V; j<H; j+=V*NT){
        auto a=read_pack<V,true>(tile+j), b=read_pack<V,Aligned>(weight+j);
        Pack<T,V> z;
        #pragma unroll
        for(int k=0; k<V; k+=2){
            auto f=__floats2bfloat162_rn(__fmul_rn(inv,__bfloat162float(a.x[k])),__fmul_rn(inv,__bfloat162float(a.x[k+1])));
            auto g=__hmul2(f,__halves2bfloat162(b.x[k],b.x[k+1]));
            z.x[k]=__low2bfloat16(g); z.x[k+1]=__high2bfloat16(g);
        }
        write_pack<V,Aligned>(y+base+j,z);
    }
}
extern "C" int launch(void** p,int rows,void* stream){
    uintptr_t bits=uintptr_t(p[0]) | uintptr_t(p[1]) | uintptr_t(p[2]) | uintptr_t(p[3]);
    if((bits & (4*sizeof(T)-1))==0)
        fused_packed_mean<true><<<rows,256,0,(cudaStream_t)stream>>>((T*)p[0], (T*)p[1], (T*)p[2], (T*)p[3]);
    else fused_packed_mean<false><<<rows,256,0,(cudaStream_t)stream>>>((T*)p[0], (T*)p[1], (T*)p[2], (T*)p[3]);
    return (int)cudaGetLastError();
}
