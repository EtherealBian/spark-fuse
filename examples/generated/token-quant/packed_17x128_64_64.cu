#include <cuda_runtime.h>
__global__ void packed_int32_fanout(const float* row,const int* packed,const float* c0,const float* c1,const float* c2,int* acc0,float* output0,int* acc1,float* output1,int* acc2,float* output2) {
    if (blockIdx.x < 17u) {
        const int i=(int(blockIdx.x)-0)*128+int(threadIdx.x);
        if (i >= 2176) return;
        const int r=i/128, c=i%128;
        const int v=packed[r*256+0+c];
        const float a=__int2float_rn(v);
        const float b=__fmul_rn(a,row[r]);
        acc0[i]=v;
        output0[i]=__fmul_rn(b,c0[c]);
        return;
    }
    if (blockIdx.x < 26u) {
        const int i=(int(blockIdx.x)-17)*128+int(threadIdx.x);
        if (i >= 1088) return;
        const int r=i/64, c=i%64;
        const int v=packed[r*256+128+c];
        const float a=__int2float_rn(v);
        const float b=__fmul_rn(a,row[r]);
        acc1[i]=v;
        output1[i]=__fmul_rn(b,c1[c]);
        return;
    }
    if (blockIdx.x < 35u) {
        const int i=(int(blockIdx.x)-26)*128+int(threadIdx.x);
        if (i >= 1088) return;
        const int r=i/64, c=i%64;
        const int v=packed[r*256+192+c];
        const float a=__int2float_rn(v);
        const float b=__fmul_rn(a,row[r]);
        acc2[i]=v;
        output2[i]=__fmul_rn(b,c2[c]);
        return;
    }
}
extern "C" int launch(void** p,int,void* stream) {
    packed_int32_fanout<<<35,128,0,(cudaStream_t)stream>>>((const float*)p[0],(const int*)p[1],(const float*)p[2],(const float*)p[3],(const float*)p[4],(int*)p[5],(float*)p[6],(int*)p[7],(float*)p[8],(int*)p[9],(float*)p[10]);
    return int(cudaGetLastError());
}
