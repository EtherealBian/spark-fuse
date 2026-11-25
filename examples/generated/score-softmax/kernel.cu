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

__global__ void fused(const __nv_bfloat16* __restrict__ in0, const __nv_bfloat16* __restrict__ in1, __nv_bfloat16* __restrict__ out0, __nv_bfloat16* __restrict__ out1, int rows) {
const int row=blockIdx.x*4+(threadIdx.x>>5); if(row>=rows)return; const int lane=(threadIdx.x&31);
float held[8];
float acc=-INFINITY;
for(int first=lane*4;first<129;first+=128){
auto pk0=load_tail<__nv_bfloat16,4>(in0+((row*129+first)),129-first);
auto pk1=load_tail<__nv_bfloat16,4>(in1+(first),129-first);
#pragma unroll
for(int q=0;q<4;++q){const int j=first+q;
if(j>=129)continue;
const int pos=row*129+j;
const float v2 = 0x1.0000000000000p-3f;
const float v3 = __bfloat162float(__float2bfloat16_rn(__bfloat162float(pk0.x[q])*v2));
const float v4 = __bfloat162float(__float2bfloat16_rn(v3+__bfloat162float(pk1.x[q])));
acc=fmaxf(acc,v4);
held[(first/128)*4+q]=v4;
}
}
for(int d=16;d;d>>=1) acc=fmaxf(acc,__shfl_down_sync(0xffffffff,acc,d));
float rowmax=__shfl_sync(0xffffffff,acc,0);
float mass=0.f;
for(int first=lane*4;first<129;first+=128){
#pragma unroll
for(int q=0;q<4;++q){const int j=first+q;
if(j>=129)continue;
const int pos=row*129+j;
float e=expf(held[(first/128)*4+q]-rowmax); held[(first/128)*4+q]=e; mass+=e;
}
}
for(int d=16;d;d>>=1) mass=mass+__shfl_down_sync(0xffffffff,mass,d);
float rowsum=__shfl_sync(0xffffffff,mass,0);
const float inv=1.f/rowsum;
for(int first=lane*4;first<129;first+=128){
auto pk0=load_tail<__nv_bfloat16,4>(in0+((row*129+first)),129-first);
auto pk1=load_tail<__nv_bfloat16,4>(in1+(first),129-first);
Pack<__nv_bfloat16,4> dst0{};
Pack<__nv_bfloat16,4> dst1{};
#pragma unroll
for(int q=0;q<4;++q){const int j=first+q;
if(j>=129)continue;
const int pos=row*129+j;
const float prob=__bfloat162float(__float2bfloat16_rn(held[(first/128)*4+q]*inv));
const float v2 = 0x1.0000000000000p-3f;
const float v3 = __bfloat162float(__float2bfloat16_rn(__bfloat162float(pk0.x[q])*v2));
const float v4 = __bfloat162float(__float2bfloat16_rn(v3+__bfloat162float(pk1.x[q])));
dst0.x[q]=__float2bfloat16_rn(prob);
dst1.x[q]=__float2bfloat16_rn(v4);
}
store_tail(out0+row*129+first,dst0,129-first);
store_tail(out1+row*129+first,dst1,129-first);
}
}
extern "C" int launch(void **ptr,int rows,void *stream){
fused<<<(rows-1)/4+1,128,0,(cudaStream_t)stream>>>((__nv_bfloat16*)ptr[0],(__nv_bfloat16*)ptr[1],(__nv_bfloat16*)ptr[2],(__nv_bfloat16*)ptr[3],rows);return (int)cudaGetLastError();
}
