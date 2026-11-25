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
__global__ void fused(const float* __restrict__ in0, const float* __restrict__ in1, const float* __restrict__ in2, const float* __restrict__ in3, float* __restrict__ out0, float* __restrict__ out1, int rows) {
const int row=blockIdx.x; if(row>=rows)return; const int lane=threadIdx.x;
float held[8];
float origin; {const int j=0, pos=row*768;
const float v0 = in0[pos];
const float v1 = in1[pos];
const float v4 = v0+v1;
origin=v4; }
float moment=0.f;
for(int first=lane*4;first<768;first+=512){
auto pk0=loadv<float,4>(in0+((row*768+first)));
auto pk1=loadv<float,4>(in1+((row*768+first)));
#pragma unroll
for(int q=0;q<4;++q){const int j=first+q;

const int pos=row*768+j;
const float v4 = pk0.x[q]+pk1.x[q];
const float shifted=v4-origin; moment+=shifted;
held[(first/512)*4+q]=shifted;
}
}
for(int d=16;d;d>>=1) moment=moment+__shfl_down_sync(0xffffffff,moment,d);
__shared__ float part_sum_shifted[4]; if((lane&31)==0)part_sum_shifted[lane>>5]=moment; __syncthreads();
if(lane<32){ moment=lane<4?part_sum_shifted[lane]:0.f;
for(int d=16;d;d>>=1) moment=moment+__shfl_down_sync(0xffffffff,moment,d);
if(lane==0)part_sum_shifted[0]=moment; } __syncthreads(); const float sum_shifted=part_sum_shifted[0];
const float center=sum_shifted*(1.f/768.f); float variance=0.f;
for(int first=lane*4;first<768;first+=512){
#pragma unroll
for(int q=0;q<4;++q){const int j=first+q;

const int pos=row*768+j;
const float centered=held[(first/512)*4+q]-center; variance+=centered*centered; held[(first/512)*4+q]=centered;
}
}
for(int d=16;d;d>>=1) variance=variance+__shfl_down_sync(0xffffffff,variance,d);
__shared__ float part_sum_square[4]; if((lane&31)==0)part_sum_square[lane>>5]=variance; __syncthreads();
if(lane<32){ variance=lane<4?part_sum_square[lane]:0.f;
for(int d=16;d;d>>=1) variance=variance+__shfl_down_sync(0xffffffff,variance,d);
if(lane==0)part_sum_square[0]=variance; } __syncthreads(); const float sum_square=part_sum_square[0];
const float inv=rsqrtf(sum_square*(1.f/768.f)+0x1.4f8b588e368f1p-17f);
for(int first=lane*4;first<768;first+=512){
auto pk0=loadv<float,4>(in0+((row*768+first)));
auto pk1=loadv<float,4>(in1+((row*768+first)));
auto pk2=loadv<float,4>(in2+(first));
auto pk3=loadv<float,4>(in3+(first));
Pack<float,4> dst0{};
Pack<float,4> dst1{};
#pragma unroll
for(int q=0;q<4;++q){const int j=first+q;

const int pos=row*768+j;
const float prob=held[(first/512)*4+q]*inv;
const float v6 = prob*pk2.x[q];
const float v7 = v6+pk3.x[q];
const float v8 = v7;
const float v4 = pk0.x[q]+pk1.x[q];
dst0.x[q]=v8;
dst1.x[q]=v4;
}
storev(out0+row*768+first,dst0);
storev(out1+row*768+first,dst1);
}
}
extern "C" int launch(void **ptr,int rows,void *stream){
fused<<<rows,128,0,(cudaStream_t)stream>>>((float*)ptr[0],(float*)ptr[1],(float*)ptr[2],(float*)ptr[3],(float*)ptr[4],(float*)ptr[5],rows);return (int)cudaGetLastError();
}
