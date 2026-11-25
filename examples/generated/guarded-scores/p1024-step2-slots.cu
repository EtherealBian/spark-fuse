#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>
__global__ void fused(const __nv_bfloat16* __restrict__ in0, const __nv_bfloat16* __restrict__ in1, __nv_bfloat16* __restrict__ out0, int rows) {
const int row=blockIdx.x*4+(threadIdx.x>>5); if(row>=rows)return; const int lane=(threadIdx.x&31);
float held[33];
float acc=-INFINITY;
#pragma unroll
for(int slot=0;slot<33;++slot){ const int j=lane+slot*32;
if(j<1026){ const int pos=row*1026+j;
const float v0 = __bfloat162float(in0[pos]);
const float v2 = 0x1.6a09e667f3bcdp-4f;
const float v3 = __bfloat162float(__float2bfloat16_rn(v0*v2));
const float v1 = __bfloat162float(in1[((row/12)%2)*1026+j]);
const float v4 = __bfloat162float(__float2bfloat16_rn(v3+v1));
const float v5 = v4;
acc=fmaxf(acc,v5);
held[slot]=v5;
}}
for(int d=16;d;d>>=1) acc=fmaxf(acc,__shfl_xor_sync(0xffffffff,acc,d));
const float rowmax=acc;
float mass=0.f;
#pragma unroll
for(int slot=0;slot<33;++slot){ const int j=lane+slot*32;
if(j<1026){ const int pos=row*1026+j;
float e=expf(held[slot]-rowmax); held[slot]=e; mass+=e;
}}
for(int d=16;d;d>>=1) mass=mass+__shfl_xor_sync(0xffffffff,mass,d);
const float rowsum=mass;
const float inv=1.f/rowsum;
#pragma unroll
for(int slot=0;slot<33;++slot){ const int j=lane+slot*32;
if(j<1026){ const int pos=row*1026+j;
const float prob=__fdiv_rn(held[slot],rowsum);
const float v7 = __bfloat162float(__float2bfloat16_rn(prob));
out0[pos]=__float2bfloat16_rn(v7);
}}
}
extern "C" int launch(void **ptr,int rows,void *stream){
fused<<<(rows-1)/4+1,128,0,(cudaStream_t)stream>>>((__nv_bfloat16*)ptr[0],(__nv_bfloat16*)ptr[1],(__nv_bfloat16*)ptr[2],rows);return (int)cudaGetLastError();
}
