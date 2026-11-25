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
__global__ void fused_mean_tree(const __half* __restrict__ in0, const __half* __restrict__ in1, const __half* __restrict__ in2, __half* __restrict__ out0, int rows) {
const int lane=threadIdx.x&31, warp=threadIdx.x/32;
const int row=blockIdx.x*16+warp;
if(row>=rows)return;
float sum=0.f;
{ const int vw=0;
float accum[4]={};
for(int first=(32*vw+lane)*4;first<8160;first+=128){
auto pack0=loadv<__half,4>(in0+row*8160+first);
auto pack1=loadv<__half,4>(in1+row*8160+first);
#pragma unroll
for(int q=0;q<4;q++){ const int j=first+q, pos=row*8160+j;
const float v3 = __half2float(__float2half_rn(__half2float(pack0.x[q])+__half2float(pack1.x[q])));
const float v4 = v3;
const float v5 = v4*v4;
accum[q]=__fadd_rn(accum[q],v5);
}}
float v=__fadd_rn(__fadd_rn(__fadd_rn(accum[0],accum[1]),accum[2]),accum[3]);
for(int d=16;d;d>>=1)v=__fadd_rn(v,__shfl_down_sync(0xffffffff,v,d));
sum=__shfl_sync(0xffffffff,v,0); }
const float reduced=__fmul_rn(sum,1.f/8160.f);
{ const int vw=0;
for(int first=(32*vw+lane)*4;first<8160;first+=128){
auto pack0=loadv<__half,4>(in0+row*8160+first);
auto pack1=loadv<__half,4>(in1+row*8160+first);
auto pack2=loadv<__half,4>(in2+first);
Pack<__half,4> dst0;
#pragma unroll
for(int q=0;q<4;q++){ const int j=first+q, pos=row*8160+j;
const float v3 = __half2float(__float2half_rn(__half2float(pack0.x[q])+__half2float(pack1.x[q])));
const float v4 = v3;
const float v7 = 0x1.0c6f7a0b5ed8dp-20f;
const float v8 = reduced+v7;
const float v9 = rsqrtf(v8);
const float v10 = v4*v9;
const float v11 = __half2float(__float2half_rn(v10));
const float v12 = __half2float(__float2half_rn(__half2float(pack2.x[q])*v11));
dst0.x[q]=__float2half_rn(v12);
}
storev(out0+row*8160+first,dst0);
}}
}
extern "C" int launch(void**ptr,int rows,void*stream){
fused_mean_tree<<<(rows-1)/16+1,512,0,(cudaStream_t)stream>>>((__half*)ptr[0],(__half*)ptr[1],(__half*)ptr[2],(__half*)ptr[3],rows); return (int)cudaGetLastError();}
