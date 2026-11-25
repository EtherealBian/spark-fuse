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
__global__ void fused(const __nv_bfloat16* __restrict__ in0, const __nv_bfloat16* __restrict__ in1, const __nv_bfloat16* __restrict__ in2, __nv_bfloat16* __restrict__ out0, int rows) {
const int row=blockIdx.x; if(row>=rows)return; const int lane=threadIdx.x;
float acc=0.f;
for(int first=lane*4;first<1536;first+=2048) {
auto pack0=loadv<__nv_bfloat16,4>(in0+row*1536+first);
auto pack1=loadv<__nv_bfloat16,4>(in1+row*1536+first);
#pragma unroll
for(int q=0;q<4;q++){ const int j=first+q, pos=row*1536+j;
const float v3 = __bfloat162float(__float2bfloat16_rn(__bfloat162float(pack0.x[q])+__bfloat162float(pack1.x[q])));
const float v4 = v3*v3;
acc+=v4;
}}
for(int d=16;d;d>>=1)acc+=__shfl_down_sync(0xffffffff,acc,d);
__shared__ float parts[16];
if((lane&31)==0)parts[lane/32]=acc; __syncthreads();
if(lane<32){acc=lane<16?parts[lane]:0.f;for(int d=16;d;d>>=1)acc+=__shfl_down_sync(0xffffffff,acc,d);if(lane==0)parts[0]=acc;}__syncthreads();float sum=parts[0];
float reduced=sum*(1.f/1536.f);
for(int first=lane*4;first<1536;first+=2048) {
auto pack0=loadv<__nv_bfloat16,4>(in0+row*1536+first);
auto pack1=loadv<__nv_bfloat16,4>(in1+row*1536+first);
auto pack2=loadv<__nv_bfloat16,4>(in2+first);
Pack<__nv_bfloat16,4> dst0;
#pragma unroll
for(int q=0;q<4;q++){ const int j=first+q, pos=row*1536+j;
const float v3 = __bfloat162float(__float2bfloat16_rn(__bfloat162float(pack0.x[q])+__bfloat162float(pack1.x[q])));
const float v6 = 0x1.0c6f7a0b5ed8dp-20f;
const float v7 = reduced+v6;
const float v8 = rsqrtf(v7);
const float v9 = v3*v8;
const float v10 = __bfloat162float(__float2bfloat16_rn(v9*__bfloat162float(pack2.x[q])));
dst0.x[q]=__float2bfloat16_rn(v10);
}
storev(out0+row*1536+first,dst0);
}
}
extern "C" int launch(void **ptr,int rows,void *stream){
fused<<<rows,512,0,(cudaStream_t)stream>>>((__nv_bfloat16*)ptr[0],(__nv_bfloat16*)ptr[1],(__nv_bfloat16*)ptr[2],(__nv_bfloat16*)ptr[3],rows);return (int)cudaGetLastError();
}
