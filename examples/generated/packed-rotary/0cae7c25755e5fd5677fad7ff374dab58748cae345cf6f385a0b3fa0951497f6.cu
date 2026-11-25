#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
using T=__nv_bfloat16; using Bits=unsigned short;
template<int H,int Offset,bool TokenMajor>
__device__ __forceinline__ void rotate_packed(const T* x,const T* c,const T* s,T* y,int i) {
  const int row=i/64, j=i%64, batch=row/(H*1024);
  const int token=TokenMajor?(row/H)%1024:row%1024, head=TokenMajor?row%H:(row/1024)%H;
  const int p=((batch*1024+token)*128)+j, a=row*128+j;
  const int in=(batch*1024+token)*2048+Offset+head*128+j;
  const float l=float(x[in]), r=float(x[in+64]);
  const T cl=T(l*float(c[p])), sl=T((-r)*float(s[p]));
  const T cr=T(r*float(c[p+64])), sr=T(l*float(s[p+64]));
  y[a]=T(float(cl)+float(sl)); y[a+64]=T(float(cr)+float(sr));
}
__global__ void packed_rotary_copy(const T* x,const T* c,const T* s,T* q,T* k,Bits* v) {
  const int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<1572864) rotate_packed<12,0,true>(x,c,s,q,i);
  else if(i<1835008) rotate_packed<2,1536,true>(x,c,s,k,i-1572864);
  else if(i<2359296) {const int j=i-1835008; v[j]=reinterpret_cast<const Bits*>(x)[(j/256)*2048+1792+j%256];}
}
extern "C" int launch(void** p,int,void* stream) {
  packed_rotary_copy<<<18432,128,0,(cudaStream_t)stream>>>((const T*)p[0],(const T*)p[1],(const T*)p[2],(T*)p[3],(T*)p[4],(Bits*)p[5]);
  return int(cudaGetLastError());
}
