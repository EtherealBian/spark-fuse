#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
__global__ void packed_affine_projection(const float* row,const int* packed,const float* c0,const float* b0,const float* c1,const float* b1,const float* c2,const float* b2,__nv_bfloat16* narrow0,__nv_bfloat16* narrow1,__nv_bfloat16* narrow2) {
    if (blockIdx.x < 17u) {
        const int i=(int(blockIdx.x)-0)*128+int(threadIdx.x);
        if (i >= 2176) return;
        const int v=packed[(i/128)*256+0+i%128];
        const float a=__int2float_rn(v);
        const float r=__fmul_rn(a,row[i/128]);
        const float o=__fmul_rn(r,c0[i%128]);
        const float b=__fadd_rn(o,b0[i%128]);
        narrow0[i]=__float2bfloat16_rn(b);
        return;
    }
    if (blockIdx.x < 26u) {
        const int i=(int(blockIdx.x)-17)*128+int(threadIdx.x);
        if (i >= 1088) return;
        const int v=packed[(i/64)*256+128+i%64];
        const float a=__int2float_rn(v);
        const float r=__fmul_rn(a,row[i/64]);
        const float o=__fmul_rn(r,c1[i%64]);
        const float b=__fadd_rn(o,b1[i%64]);
        narrow1[i]=__float2bfloat16_rn(b);
        return;
    }
    if (blockIdx.x < 35u) {
        const int i=(int(blockIdx.x)-26)*128+int(threadIdx.x);
        if (i >= 1088) return;
        const int v=packed[(i/64)*256+192+i%64];
        const float a=__int2float_rn(v);
        const float r=__fmul_rn(a,row[i/64]);
        const float o=__fmul_rn(r,c2[i%64]);
        const float b=__fadd_rn(o,b2[i%64]);
        narrow2[i]=__float2bfloat16_rn(b);
        return;
    }
}
extern "C" int launch(void** p,int,void* stream) {
    packed_affine_projection<<<35,128,0,(cudaStream_t)stream>>>((const float*)p[0],(const int*)p[1],(const float*)p[2],(const float*)p[3],(const float*)p[4],(const float*)p[5],(const float*)p[6],(const float*)p[7],(__nv_bfloat16*)p[8],(__nv_bfloat16*)p[9],(__nv_bfloat16*)p[10]);
    return int(cudaGetLastError());
}
