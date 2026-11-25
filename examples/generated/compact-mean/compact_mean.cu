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

template<class T,int V> __device__ __forceinline__ Pack<T,V> load_tail(const T* p,int remain){
  if(remain>=V) return loadv<T,V>(p);
  Pack<T,V> value{};
  #pragma unroll
  for(int i=0;i<V;i++) if(i<remain) value.x[i]=p[i];
  return value;
}
template<class T,int V> __device__ __forceinline__ void store_tail(T* p,Pack<T,V> value,int remain){
  if(remain>=V){ storev(p,value); return; }
  #pragma unroll
  for(int i=0;i<V;i++) if(i<remain) p[i]=value.x[i];
}

template<bool Aligned> __global__ void fused_compact(const __nv_bfloat16* __restrict__ in0, __nv_bfloat16* __restrict__ out0, int rows) {
const int row=blockIdx.x, lane=threadIdx.x; if(row>=rows)return; float sums[8]={};
for(unsigned ix=threadIdx.x*8u;ix<14336u;ix+=8192u) {const int first=(int)ix;
Pack<__nv_bfloat16,8> pk0;
if constexpr (Aligned) pk0=*reinterpret_cast<const Pack<__nv_bfloat16,8>*>((in0+row*14336+first));
else pk0=load_tail<__nv_bfloat16,8>((in0+row*14336+first),14336-first);
#pragma unroll
for(int q=0;q<8;q++){const int j=first+q; if(j>=14336)continue; const int pos=row*14336+j;
const float v1 = __bfloat162float(__float2bfloat16_rn(__bfloat162float(pk0.x[q])*__bfloat162float(pk0.x[q])));
sums[q]+=v1;
}}
float acc=sums[0];
#pragma unroll
for(int q=1;q<8;q++)acc+=sums[q];
for(int d=16;d;d>>=1)acc+=__shfl_down_sync(0xffffffff,acc,d);
__shared__ float part[32];
if((lane&31)==0)part[lane/32]=acc; __syncthreads(); if(lane>=32)return;
acc=lane<32?part[lane]:0.f;
for(int d=16;d;d>>=1)acc+=__shfl_down_sync(0xffffffff,acc,d);
if(lane!=0)return;
const int j=0, pos=row*14336;
const float reduced=__bfloat162float(__float2bfloat16_rn(acc*(1.f/14336.f)));

out0[row]=__float2bfloat16_rn(reduced);
}
extern "C" int launch(void**ptr,int rows,void*stream){
auto in0=static_cast<__nv_bfloat16*>(ptr[0]);
auto out0=static_cast<__nv_bfloat16*>(ptr[1]);
if(((uintptr_t(in0) & (alignof(Pack<__nv_bfloat16,8>)-1))==0)) fused_compact<true><<<rows,1024,0,(cudaStream_t)stream>>>(in0,out0,rows); else fused_compact<false><<<rows,1024,0,(cudaStream_t)stream>>>(in0,out0,rows);
return (int)cudaGetLastError();
}
