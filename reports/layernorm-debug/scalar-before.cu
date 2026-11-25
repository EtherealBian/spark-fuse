#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>
__global__ void fused(const __nv_bfloat16* __restrict__ in0, const __nv_bfloat16* __restrict__ in1, const __nv_bfloat16* __restrict__ in2, const __nv_bfloat16* __restrict__ in3, __nv_bfloat16* __restrict__ out0, __nv_bfloat16* __restrict__ out1, int rows) {
const int row=blockIdx.x; if(row>=rows)return; const int lane=threadIdx.x;
float held[6];
float origin; {const int j=0, pos=row*768;
const float v0 = __bfloat162float(in0[pos]);
const float v1 = __bfloat162float(in1[pos]);
const float v4 = __bfloat162float(__float2bfloat16_rn(v0+v1));
origin=v4; }
float moment=0.f;
for(int j=lane;j<768;j+=128){const int pos=row*768+j;
const float v0 = __bfloat162float(in0[pos]);
const float v1 = __bfloat162float(in1[pos]);
const float v4 = __bfloat162float(__float2bfloat16_rn(v0+v1));
const float shifted=v4-origin; moment+=shifted;
held[j/128]=shifted;
}
for(int d=16;d;d>>=1) moment=moment+__shfl_down_sync(0xffffffff,moment,d);
__shared__ float part_sum_shifted[4]; if((lane&31)==0)part_sum_shifted[lane>>5]=moment; __syncthreads();
if(lane<32){ moment=lane<4?part_sum_shifted[lane]:0.f;
for(int d=16;d;d>>=1) moment=moment+__shfl_down_sync(0xffffffff,moment,d);
if(lane==0)part_sum_shifted[0]=moment; } __syncthreads(); const float sum_shifted=part_sum_shifted[0];
const float center=sum_shifted*(1.f/768.f); float variance=0.f;
for(int j=lane;j<768;j+=128){const int pos=row*768+j;
const float centered=held[j/128]-center; variance+=centered*centered; held[j/128]=centered;
}
for(int d=16;d;d>>=1) variance=variance+__shfl_down_sync(0xffffffff,variance,d);
__shared__ float part_sum_square[4]; if((lane&31)==0)part_sum_square[lane>>5]=variance; __syncthreads();
if(lane<32){ variance=lane<4?part_sum_square[lane]:0.f;
for(int d=16;d;d>>=1) variance=variance+__shfl_down_sync(0xffffffff,variance,d);
if(lane==0)part_sum_square[0]=variance; } __syncthreads(); const float sum_square=part_sum_square[0];
const float inv=rsqrtf(sum_square*(1.f/768.f)+0x1.4f8b588e368f1p-17f);
for(int j=lane;j<768;j+=128){const int pos=row*768+j;
const float prob=held[j/128]*inv;
const float v2 = __bfloat162float(in2[j]);
const float v6 = prob*v2;
const float v3 = __bfloat162float(in3[j]);
const float v7 = v6+v3;
const float v8 = __bfloat162float(__float2bfloat16_rn(v7));
const float v0 = __bfloat162float(in0[pos]);
const float v1 = __bfloat162float(in1[pos]);
const float v4 = __bfloat162float(__float2bfloat16_rn(v0+v1));
out0[pos]=__float2bfloat16_rn(v8);
out1[pos]=__float2bfloat16_rn(v4);
}
}
extern "C" int launch(void **ptr,int rows,void *stream){
fused<<<rows,128,0,(cudaStream_t)stream>>>((__nv_bfloat16*)ptr[0],(__nv_bfloat16*)ptr[1],(__nv_bfloat16*)ptr[2],(__nv_bfloat16*)ptr[3],(__nv_bfloat16*)ptr[4],(__nv_bfloat16*)ptr[5],rows);return (int)cudaGetLastError();
}
