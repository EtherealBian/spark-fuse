#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>
__global__ void fused(const __nv_bfloat16* __restrict__ in0, const __nv_bfloat16* __restrict__ in1, __nv_bfloat16* __restrict__ out0, __nv_bfloat16* __restrict__ out1, float* __restrict__ out2, int rows) {
const int row=blockIdx.x*4+(threadIdx.x>>5); if(row>=rows)return; const int lane=(threadIdx.x&31);
float held[64];
float acc=-INFINITY;
#pragma unroll 1
for(int j=lane;j<2047;j+=32){const int pos=row*2047+j;
const float v0 = __bfloat162float(in0[pos]);
const float v2 = 0x1.6a09e667f3bcdp-4f;
const float v3 = __bfloat162float(__float2bfloat16_rn(v0*v2));
const float v1 = __bfloat162float(in1[j]);
const float v4 = __bfloat162float(__float2bfloat16_rn(v3+v1));
const float v5 = v4;
acc=fmaxf(acc,v5);
held[j/32]=v5;
}
for(int d=16;d;d>>=1) acc=fmaxf(acc,__shfl_xor_sync(0xffffffff,acc,d));
const float rowmax=acc;
float mass=0.f;
#pragma unroll 1
for(int j=lane;j<2047;j+=32){const int pos=row*2047+j;
float e=expf(held[j/32]-rowmax); held[j/32]=e; mass+=e;
}
for(int d=16;d;d>>=1) mass=mass+__shfl_xor_sync(0xffffffff,mass,d);
const float rowsum=mass;
const float inv=1.f/rowsum;
#pragma unroll 1
for(int j=lane;j<2047;j+=32){const int pos=row*2047+j;
const float prob=__fdiv_rn(held[j/32],rowsum);
const float v7 = __bfloat162float(__float2bfloat16_rn(prob));
const float v0 = __bfloat162float(in0[pos]);
const float v2 = 0x1.6a09e667f3bcdp-4f;
const float v3 = __bfloat162float(__float2bfloat16_rn(v0*v2));
const float v1 = __bfloat162float(in1[j]);
const float v4 = __bfloat162float(__float2bfloat16_rn(v3+v1));
out0[pos]=__float2bfloat16_rn(v7);
out1[pos]=__float2bfloat16_rn(v4);
out2[pos]=prob;
}
}
extern "C" int launch(void **ptr,int rows,void *stream){
fused<<<(rows-1)/4+1,128,0,(cudaStream_t)stream>>>((__nv_bfloat16*)ptr[0],(__nv_bfloat16*)ptr[1],(__nv_bfloat16*)ptr[2],(__nv_bfloat16*)ptr[3],(float*)ptr[4],rows);return (int)cudaGetLastError();
}
