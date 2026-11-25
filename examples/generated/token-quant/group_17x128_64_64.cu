#include <cuda_runtime.h>
__global__ void grouped_int32_epilogue(const float* row,const int* a0,const float* c0,const int* a1,const float* c1,const int* a2,const float* c2,float* output0,float* output1,float* output2) {
    if (blockIdx.x < 17u) {
        const int i=(int(blockIdx.x)-0)*128 + int(threadIdx.x);
        if (i >= 2176) return;
        const float a=__int2float_rn(a0[i]);
        const float r=__fmul_rn(a,row[i/128]);
        output0[i]=__fmul_rn(r,c0[i%128]);
        return;
    }
    if (blockIdx.x < 26u) {
        const int i=(int(blockIdx.x)-17)*128 + int(threadIdx.x);
        if (i >= 1088) return;
        const float a=__int2float_rn(a1[i]);
        const float r=__fmul_rn(a,row[i/64]);
        output1[i]=__fmul_rn(r,c1[i%64]);
        return;
    }
    if (blockIdx.x < 35u) {
        const int i=(int(blockIdx.x)-26)*128 + int(threadIdx.x);
        if (i >= 1088) return;
        const float a=__int2float_rn(a2[i]);
        const float r=__fmul_rn(a,row[i/64]);
        output2[i]=__fmul_rn(r,c2[i%64]);
        return;
    }
}
extern "C" int launch(void** p,int,void* stream) {
    grouped_int32_epilogue<<<35,128,0,(cudaStream_t)stream>>>((const float*)p[0],(const int*)p[1],(const float*)p[2],(const int*)p[3],(const float*)p[4],(const int*)p[5],(const float*)p[6],(float*)p[7],(float*)p[8],(float*)p[9]);
    return int(cudaGetLastError());
}
