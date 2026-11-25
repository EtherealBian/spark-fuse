#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>
__global__ void fused(const float* __restrict__ in0, const float* __restrict__ in1, float* __restrict__ out0, float* __restrict__ out1, int rows) {
const int row=blockIdx.x; if(row>=rows)return; const int lane=threadIdx.x;
float held[9];
float acc=-INFINITY;
for(int j=lane;j<1089;j+=128){const int pos=row*1089+j;
const float v0 = in0[pos];
const float v2 = 0x1.6a09e667f3bcdp-4f;
const float v3 = v0*v2;
const float v1 = in1[j];
const float v4 = v3+v1;
acc=fmaxf(acc,v4);
held[j/128]=v4;
}
for(int d=16;d;d>>=1) acc=fmaxf(acc,__shfl_down_sync(0xffffffff,acc,d));
__shared__ float part_rowmax[4]; if((lane&31)==0)part_rowmax[lane>>5]=acc; __syncthreads();
if(lane<32){ acc=lane<4?part_rowmax[lane]:-INFINITY;
for(int d=16;d;d>>=1) acc=fmaxf(acc,__shfl_down_sync(0xffffffff,acc,d));
if(lane==0)part_rowmax[0]=acc; } __syncthreads(); const float rowmax=part_rowmax[0];
float mass=0.f;
for(int j=lane;j<1089;j+=128){const int pos=row*1089+j;
float e=expf(held[j/128]-rowmax); held[j/128]=e; mass+=e;
}
for(int d=16;d;d>>=1) mass=mass+__shfl_down_sync(0xffffffff,mass,d);
__shared__ float part_rowsum[4]; if((lane&31)==0)part_rowsum[lane>>5]=mass; __syncthreads();
if(lane<32){ mass=lane<4?part_rowsum[lane]:0.f;
for(int d=16;d;d>>=1) mass=mass+__shfl_down_sync(0xffffffff,mass,d);
if(lane==0)part_rowsum[0]=mass; } __syncthreads(); const float rowsum=part_rowsum[0];
const float inv=1.f/rowsum;
for(int j=lane;j<1089;j+=128){const int pos=row*1089+j;
const float prob=held[j/128]*inv;
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
fused<<<rows,128,0,(cudaStream_t)stream>>>((float*)ptr[0],(float*)ptr[1],(float*)ptr[2],(float*)ptr[3],rows);return (int)cudaGetLastError();
}
