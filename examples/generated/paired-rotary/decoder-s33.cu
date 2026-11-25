#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
using T=__nv_bfloat16;
template<int H, bool token_major>
__device__ __forceinline__ void rotate(const T* x,const T* c,const T* s,T* y,int i) {
  const int row=i/64, j=i%64, batch=row/(H*33);
  const int token=token_major ? (row/H)%33 : row%33;
  const int p=((batch*33+token)*128)+j, a=row*128+j;
  const float l=float(x[a]), r=float(x[a+64]);
  const T cl=T(l*float(c[p])), sl=T((-r)*float(s[p]));
  const T cr=T(r*float(c[p+64])), sr=T(l*float(s[p+64]));
  y[a]=T(float(cl)+float(sl)); y[a+64]=T(float(cr)+float(sr));
}
__global__ void rotary_pair(const T* q,const T* k,const T* c,const T* s,T* oq,T* ok) {
  int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<50688) rotate<12,true>(q,c,s,oq,i);
  else if(i<59136) rotate<2,true>(k,c,s,ok,i-50688);
}
extern "C" int launch(void** p,int,void* stream) {
  rotary_pair<<<231,256,0,(cudaStream_t)stream>>>((const T*)p[0],(const T*)p[1],(const T*)p[2],(const T*)p[3],(T*)p[4],(T*)p[5]);
  return int(cudaGetLastError());
}
