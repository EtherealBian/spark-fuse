#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
__device__ __forceinline__ float bf16_quotient(float n,float d,float inv){
    const float approx=__fmul_rn(n,inv);
    const unsigned bits=__float_as_uint(approx);
    const bool near=((bits & 0xffffu)-0x7ffcu)<=8u;
    const bool normal=bits>=0x00800000u && bits<=0x3f800000u;
    return ((normal && !near) || (n==0.f && isfinite(d))) ? approx : __fdiv_rn(n,d);
}
__global__ void audit(const float* n,const float* d,unsigned short* o,unsigned short* fast,unsigned short* safe,unsigned char* branches,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 float v=n[i], m=d[i], inv=__frcp_rn(m), app=__fmul_rn(v,inv);
 unsigned bits=__float_as_uint(app);
 bool near=((bits & 65535u)-32764u)<=8u;
 bool normal=bits>=0x00800000u && bits<=0x3f800000u;
 bool zero=v==0.f && isfinite(m);
 float q=__fdiv_rn(v,m);
 o[i]=__bfloat16_as_ushort(__float2bfloat16_rn(q));
 fast[i]=__bfloat16_as_ushort(__float2bfloat16_rn(app));
 safe[i]=__bfloat16_as_ushort(__float2bfloat16_rn(bf16_quotient(v,m,inv)));
 branches[i]=zero?0:!normal?1:near?2:3;
}
extern "C" int launch(void **ptr,int count,void *stream){audit<<<(count+255)/256,256,0,(cudaStream_t)stream>>>((float*)ptr[0],(float*)ptr[1],(unsigned short*)ptr[2],(unsigned short*)ptr[3],(unsigned short*)ptr[4],(unsigned char*)ptr[5],count);return (int)cudaGetLastError();}