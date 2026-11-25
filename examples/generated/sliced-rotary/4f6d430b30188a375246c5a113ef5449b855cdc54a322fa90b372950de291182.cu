#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
using T=__nv_bfloat16; using Bits=unsigned short;
template<int H,int Off> __device__ __forceinline__ void sliced_head(const T*x,const T*c,const T*s,T*y,int i){
  const int row=i/64, j=i%64, batch=row/(H*127), token=row%127, head=(row/127)%H;
  const int in=(batch*127+token)*2048+Off+head*128, out=row*128;
  if(j<64){
    const int p=((batch*127+token)*128)+j;
    const float l=float(x[in+j]), h=float(x[in+j+64]);
    const T lc=T(l*float(c[p])),ls=T((-h)*float(s[p]));
    const T hc=T(h*float(c[p+64])),hs=T(l*float(s[p+64]));
    y[out+j]=T(float(lc)+float(ls)); y[out+j+64]=T(float(hc)+float(hs));
  } else {const int k=j+64;reinterpret_cast<Bits*>(y)[out+k]=reinterpret_cast<const Bits*>(x)[in+k];}
}
__global__ void sliced_rotary_pair(const T*x,const T*c,const T*s,T*q,T*k){
  const int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<195072) sliced_head<12,0>(x,c,s,q,i);
  else if(i<227584) sliced_head<2,1536>(x,c,s,k,i-195072);
}
extern "C" int launch(void**p,int,void*stream){
  sliced_rotary_pair<<<1778,128,0,(cudaStream_t)stream>>>((const T*)p[0],(const T*)p[1],(const T*)p[2],(T*)p[3],(T*)p[4]);
  return int(cudaGetLastError());
}
