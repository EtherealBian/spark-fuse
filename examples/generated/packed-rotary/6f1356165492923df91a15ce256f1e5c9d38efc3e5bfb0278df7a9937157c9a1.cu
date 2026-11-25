#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
using T=__nv_bfloat16; using Bits=unsigned short;
template<int H,int Offset,bool TokenMajor>
__device__ __forceinline__ void rotate_packed(const T* x,const T* c,const T* s,T* y,int i) {
  const int row=i/64, j=i%64, batch=row/(H*127);
  const int token=TokenMajor?(row/H)%127:row%127, head=TokenMajor?row%H:(row/127)%H;
  const int p=((batch*127+token)*128)+j, a=row*128+j;
  const int in=(batch*127+token)*2048+Offset+head*128+j;
  const float l=float(x[in]), r=float(x[in+64]);
  const T cl=T(l*float(c[p])), sl=T((-r)*float(s[p]));
  const T cr=T(r*float(c[p+64])), sr=T(l*float(s[p+64]));
  y[a]=T(float(cl)+float(sl)); y[a+64]=T(float(cr)+float(sr));
}
__global__ void packed_rotary_copy(const T* x,const T* c,const T* s,T* q,T* k,Bits* v) {
  const int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<195072) rotate_packed<12,0,true>(x,c,s,q,i);
  else if(i<227584) rotate_packed<2,1536,true>(x,c,s,k,i-195072);
  else if(i<292608) {const int j=i-227584; v[j]=reinterpret_cast<const Bits*>(x)[(j/256)*2048+1792+j%256];}
}
extern "C" int launch(void** p,int,void* stream) {
  packed_rotary_copy<<<2286,128,0,(cudaStream_t)stream>>>((const T*)p[0],(const T*)p[1],(const T*)p[2],(T*)p[3],(T*)p[4],(Bits*)p[5]);
  return int(cudaGetLastError());
}
