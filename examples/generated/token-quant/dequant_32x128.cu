#include <cuda_runtime.h>
__global__ void ordered_int32_epilogue(const int* acc,const float* token,const float* channel,float* output) {
    const int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=4096) return;
    const float a=__int2float_rn(acc[i]);
    const float r=__fmul_rn(a,token[i/128]);
    // cast is register-only
    // row-scaled value is register-only
    output[i]=__fmul_rn(r,channel[i%128]);
}
extern "C" int launch(void** p,int,void* stream) {
    ordered_int32_epilogue<<<32,128,0,(cudaStream_t)stream>>>((int*)p[0],(float*)p[1],(float*)p[2],(float*)p[3]);
    return int(cudaGetLastError());
}
