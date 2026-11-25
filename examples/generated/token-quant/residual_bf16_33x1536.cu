#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
using N = __nv_bfloat16;
constexpr int W = 1536, T = 128;
__global__ void residual_token_quant(const N* x, const N* r, N* z, signed char* q, float* so) {
    const int base=int(blockIdx.x)*W, lane=threadIdx.x%32, warp=threadIdx.x/32;
    __shared__ unsigned peaks[T/32];
    __shared__ float scale;
    unsigned peak=0;
    N keep[(W+T-1)/T];
    #pragma unroll
    for (int k=0; k<(W+T-1)/T; ++k) {
        int i=threadIdx.x+k*T;
        if (i<W) {
            N a=__float2bfloat16_rn(__bfloat162float(x[base+i])+__bfloat162float(r[base+i]));
            keep[k]=a;
            z[base+i]=a;
            float f=__bfloat162float(a);
            peak=max(peak, isnan(f)?0x7fffffffu:(__float_as_uint(f)&0x7fffffffu));
        }
    }
    #pragma unroll
    for(int d=16;d;d>>=1) peak=max(peak,__shfl_down_sync(0xffffffff,peak,d));
    if(!lane) peaks[warp]=peak;
    __syncthreads();
    if(!warp) {
        peak=lane<T/32?peaks[lane]:0;
        #pragma unroll
        for(int d=16;d;d>>=1) peak=max(peak,__shfl_down_sync(0xffffffff,peak,d));
        if(!lane) {
            float f=__uint_as_float(peak);
            scale=isnan(f)?f:fmaxf(f,1.e-5f)*(1.f/127.f);
            so[blockIdx.x]=scale;
        }
    }
    __syncthreads();
    #pragma unroll
    for(int k=0;k<(W+T-1)/T;++k) {
        int i=threadIdx.x+k*T;
        if(i<W) {
            float a=__fdiv_rn(__bfloat162float(keep[k]),scale);
            int code=isnan(a)?0:__float2int_rn(a);
            q[base+i]=(signed char)min(max(code,-127),127);
        }
    }
}
extern "C" int launch(void** p,int,void* stream) {
    residual_token_quant<<<33,T,0,(cudaStream_t)stream>>>((const N*)p[0],(const N*)p[1],(N*)p[2], (signed char*)p[3], (float*)p[4]);
    return int(cudaGetLastError());
}
