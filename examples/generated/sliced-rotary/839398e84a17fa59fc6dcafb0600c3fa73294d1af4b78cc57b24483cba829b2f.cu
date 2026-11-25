#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
using T=__nv_bfloat16; using Bits=unsigned short;
template<int H,int Off> __device__ __forceinline__ void sliced_head(const T*x,const T*c,const T*s,T*y,int i){
  const int row=i/96, j=i%96, batch=row/(H*1), token=row%1, head=(row/1)%H;
  const int in=(batch*1+token)*2048+Off+head*128, out=row*128;
  if(j<32){
    const int p=((batch*1+token)*64)+j;
    const float l=float(x[in+j]), h=float(x[in+j+32]);
    const T lc=T(l*float(c[p])),ls=T((-h)*float(s[p]));
    const T hc=T(h*float(c[p+32])),hs=T(l*float(s[p+32]));
    y[out+j]=T(float(lc)+float(ls)); y[out+j+32]=T(float(hc)+float(hs));
  } else {const int k=j+32;reinterpret_cast<Bits*>(y)[out+k]=reinterpret_cast<const Bits*>(x)[in+k];}
}
__global__ void sliced_rotary_pair(const T*x,const T*c,const T*s,T*q,T*k){
  const int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<2304) sliced_head<12,0>(x,c,s,q,i);
  else if(i<2688) sliced_head<2,1536>(x,c,s,k,i-2304);
}
extern "C" int launch(void**p,int,void*stream){
  sliced_rotary_pair<<<21,128,0,(cudaStream_t)stream>>>((const T*)p[0],(const T*)p[1],(const T*)p[2],(T*)p[3],(T*)p[4]);
  return int(cudaGetLastError());
}
