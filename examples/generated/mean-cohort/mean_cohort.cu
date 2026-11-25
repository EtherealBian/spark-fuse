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
__global__ void fused_mean_cohort(const __half* __restrict__ in0, const __half* __restrict__ in1, const __half* __restrict__ in2, __half* __restrict__ out0, int rows) {
const int row=blockIdx.x, tid=threadIdx.x;
__shared__ float lanes[32], total;
__shared__ float tile[8160];
for(int j=tid;j<8160;j+=256){const int pos=row*8160+j;
const float v0 = __half2float(in0[pos]);
const float v1 = __half2float(in1[pos]);
const float v3 = __half2float(__float2half_rn(v0+v1));
const float v4 = v3;
tile[j]=v4;} __syncthreads();
if(tid<128){ float sum=0.f;
for(int j=tid;j<8160;j+=128){const int pos=row*8160+j;
const float v5 = tile[j]*tile[j];
sum=__fadd_rn(sum,v5);}
float b=__shfl_down_sync(0xffffffff,sum,1), c=__shfl_down_sync(0xffffffff,sum,2), d=__shfl_down_sync(0xffffffff,sum,3);
if((tid&3)==0)lanes[tid/4]=__fadd_rn(__fadd_rn(__fadd_rn(sum,b),c),d); } __syncthreads();
if(tid<32){float v=lanes[tid];for(int d=16;d;d>>=1)v=__fadd_rn(v,__shfl_down_sync(0xffffffff,v,d));
if(!tid)total=__fmul_rn(v,1.f/8160.f);} __syncthreads();
const float reduced=total;
for(int j=tid;j<8160;j+=256){const int pos=row*8160+j;
const float v2 = __half2float(in2[j]);
const float v7 = 0x1.0c6f7a0b5ed8dp-20f;
const float v8 = reduced+v7;
const float v9 = rsqrtf(v8);
const float v10 = tile[j]*v9;
const float v11 = __half2float(__float2half_rn(v10));
const float v12 = __half2float(__float2half_rn(v2*v11));
out0[pos]=__float2half_rn(v12);
}
}
extern "C" int launch(void**ptr,int rows,void*stream){
fused_mean_cohort<<<rows,256,0,(cudaStream_t)stream>>>((__half*)ptr[0],(__half*)ptr[1],(__half*)ptr[2],(__half*)ptr[3],rows);return (int)cudaGetLastError();}
