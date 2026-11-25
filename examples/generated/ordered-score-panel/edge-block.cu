#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>
__global__ void fused(const __half* __restrict__ in0, const __half* __restrict__ in1, __half* __restrict__ out0, __half* __restrict__ out1, float* __restrict__ out2, int rows) {
const int row=blockIdx.x; if(row>=rows)return; const int lane=threadIdx.x;
float held[16];
float acc=-INFINITY;
for(int j=lane;j<2048;j+=128){const int pos=row*2048+j;
const float v0 = __half2float(in0[pos]);
const float v2 = 0x1.6a09e667f3bcdp-4f;
const float v3 = __half2float(__float2half_rn(v0*v2));
const float v1 = __half2float(in1[j]);
const float v4 = __half2float(__float2half_rn(v3+v1));
const float v5 = v4;
acc=fmaxf(acc,v5);
held[j/128]=v5;
}
for(int d=16;d;d>>=1) acc=fmaxf(acc,__shfl_down_sync(0xffffffff,acc,d));
__shared__ float part_rowmax[4]; if((lane&31)==0)part_rowmax[lane>>5]=acc; __syncthreads();
if(lane<32){ acc=lane<4?part_rowmax[lane]:-INFINITY;
for(int d=16;d;d>>=1) acc=fmaxf(acc,__shfl_down_sync(0xffffffff,acc,d));
if(lane==0)part_rowmax[0]=acc; } __syncthreads(); const float rowmax=part_rowmax[0];
float mass=0.f;
for(int j=lane;j<2048;j+=128){const int pos=row*2048+j;
float e=expf(held[j/128]-rowmax); held[j/128]=e; mass+=e;
}
for(int d=16;d;d>>=1) mass=mass+__shfl_down_sync(0xffffffff,mass,d);
__shared__ float part_rowsum[4]; if((lane&31)==0)part_rowsum[lane>>5]=mass; __syncthreads();
if(lane<32){ mass=lane<4?part_rowsum[lane]:0.f;
for(int d=16;d;d>>=1) mass=mass+__shfl_down_sync(0xffffffff,mass,d);
if(lane==0)part_rowsum[0]=mass; } __syncthreads(); const float rowsum=part_rowsum[0];
const float inv=1.f/rowsum;
for(int j=lane;j<2048;j+=128){const int pos=row*2048+j;
const float prob=held[j/128]*inv;
const float v7 = __half2float(__float2half_rn(prob));
const float v0 = __half2float(in0[pos]);
const float v2 = 0x1.6a09e667f3bcdp-4f;
const float v3 = __half2float(__float2half_rn(v0*v2));
const float v1 = __half2float(in1[j]);
const float v4 = __half2float(__float2half_rn(v3+v1));
out0[pos]=__float2half_rn(v7);
out1[pos]=__float2half_rn(v4);
out2[pos]=prob;
}
}
extern "C" int launch(void **ptr,int rows,void *stream){
fused<<<rows,128,0,(cudaStream_t)stream>>>((__half*)ptr[0],(__half*)ptr[1],(__half*)ptr[2],(__half*)ptr[3],(float*)ptr[4],rows);return (int)cudaGetLastError();
}
