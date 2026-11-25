#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>
__device__ __forceinline__ float bf16_quotient(float n,float d,float inv){
    const float approx=__fmul_rn(n,inv);
    const unsigned bits=__float_as_uint(approx);
    const bool near=((bits & 0xffffu)-0x7ffcu)<=8u;
    const bool normal=bits>=0x00800000u && bits<=0x3f800000u;
    return ((normal && !near) || (n==0.f && isfinite(d))) ? approx : __fdiv_rn(n,d);
}
__global__ void fused(const __nv_bfloat16* __restrict__ in0, const __nv_bfloat16* __restrict__ in1, __nv_bfloat16* __restrict__ out0, int rows) {
const int row=blockIdx.x*4+(threadIdx.x>>5); if(row>=rows)return; const int lane=(threadIdx.x&31);
float held[49];
float acc=-INFINITY;
#pragma unroll
for(int slot=0;slot<49;++slot){ const int j=lane+slot*32;
if(j<1537){ const int pos=row*1537+j;
const float v0 = __bfloat162float(in0[pos]);
const float v2 = 0x1.6a09e667f3bcdp-4f;
const float v3 = __bfloat162float(__float2bfloat16_rn(v0*v2));
const float v1 = __bfloat162float(in1[((row/12)%2)*1537+j]);
const float v4 = __bfloat162float(__float2bfloat16_rn(v3+v1));
const float v5 = v4;
acc=fmaxf(acc,v5);
held[slot]=v5;
}}
for(int d=16;d;d>>=1) acc=fmaxf(acc,__shfl_xor_sync(0xffffffff,acc,d));
const float rowmax=acc;
float mass=0.f;
#pragma unroll
for(int slot=0;slot<49;++slot){ const int j=lane+slot*32;
if(j<1537){ const int pos=row*1537+j;
float e=expf(held[slot]-rowmax); held[slot]=e; mass+=e;
}}
for(int d=16;d;d>>=1) mass=mass+__shfl_xor_sync(0xffffffff,mass,d);
const float rowsum=mass;
const float inv=__frcp_rn(rowsum);
#pragma unroll
for(int slot=0;slot<49;++slot){ const int j=lane+slot*32;
if(j<1537){ const int pos=row*1537+j;
const float prob=bf16_quotient(held[slot],rowsum,inv);
const float v7 = __bfloat162float(__float2bfloat16_rn(prob));
out0[pos]=__float2bfloat16_rn(v7);
}}
}
extern "C" int launch(void **ptr,int rows,void *stream){
fused<<<(rows-1)/4+1,128,0,(cudaStream_t)stream>>>((__nv_bfloat16*)ptr[0],(__nv_bfloat16*)ptr[1],(__nv_bfloat16*)ptr[2],rows);return (int)cudaGetLastError();
}
