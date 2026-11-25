#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>
__global__ void fused(const float* __restrict__ in0, const float* __restrict__ in1, float* __restrict__ out0, float* __restrict__ out1, int rows) {
const int row=blockIdx.x*4+(threadIdx.x>>5); if(row>=rows)return; const int lane=(threadIdx.x&31);
float held[35];
float acc=-INFINITY;
#pragma unroll 1
for(int j=lane;j<1089;j+=32){const int pos=row*1089+j;
const float v0 = in0[pos];
const float v2 = 0x1.6a09e667f3bcdp-4f;
const float v3 = v0*v2;
const float v1 = in1[j];
const float v4 = v3+v1;
acc=fmaxf(acc,v4);
held[j/32]=v4;
}
for(int d=16;d;d>>=1) acc=fmaxf(acc,__shfl_xor_sync(0xffffffff,acc,d));
const float rowmax=acc;
float mass=0.f;
#pragma unroll 1
for(int j=lane;j<1089;j+=32){const int pos=row*1089+j;
float e=expf(held[j/32]-rowmax); held[j/32]=e; mass+=e;
}
for(int d=16;d;d>>=1) mass=mass+__shfl_xor_sync(0xffffffff,mass,d);
const float rowsum=mass;
const float inv=1.f/rowsum;
#pragma unroll 1
for(int j=lane;j<1089;j+=32){const int pos=row*1089+j;
const float prob=__fdiv_rn(held[j/32],rowsum);
const float v0 = in0[pos];
const float v2 = 0x1.6a09e667f3bcdp-4f;
const float v3 = v0*v2;
const float v1 = in1[j];
const float v4 = v3+v1;
out0[pos]=v4;
out1[pos]=prob;
}
}
extern "C" int launch(void **ptr,int rows,void *stream){
fused<<<(rows-1)/4+1,128,0,(cudaStream_t)stream>>>((float*)ptr[0],(float*)ptr[1],(float*)ptr[2],(float*)ptr[3],rows);return (int)cudaGetLastError();
}
