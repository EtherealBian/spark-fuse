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
__global__ void fused(const __nv_bfloat16* __restrict__ in0, __nv_bfloat16* __restrict__ out0, int rows) {
const int row=blockIdx.x; if(row>=rows)return; const int lane=threadIdx.x;
float acc=0.f;
for(int first=lane*8;first<14336;first+=8192) {
auto pack0=loadv<__nv_bfloat16,8>(in0+row*14336+first);
#pragma unroll
for(int q=0;q<8;q++){ const int j=first+q, pos=row*14336+j;
const float v1 = __bfloat162float(__float2bfloat16_rn(__bfloat162float(pack0.x[q])*__bfloat162float(pack0.x[q])));
acc+=v1;
}}
for(int d=16;d;d>>=1)acc+=__shfl_down_sync(0xffffffff,acc,d);
__shared__ float parts[32];
if((lane&31)==0)parts[lane/32]=acc; __syncthreads();
if(lane<32){acc=lane<32?parts[lane]:0.f;for(int d=16;d;d>>=1)acc+=__shfl_down_sync(0xffffffff,acc,d);if(lane==0)parts[0]=acc;}__syncthreads();float sum=parts[0];
float reduced=__bfloat162float(__float2bfloat16_rn(sum*(1.f/14336.f)));
if(lane==0){const int j=0,pos=row*14336;

out0[row]=__float2bfloat16_rn(reduced);
}
}
extern "C" int launch(void **ptr,int rows,void *stream){
fused<<<rows,1024,0,(cudaStream_t)stream>>>((__nv_bfloat16*)ptr[0],(__nv_bfloat16*)ptr[1],rows);return (int)cudaGetLastError();
}
