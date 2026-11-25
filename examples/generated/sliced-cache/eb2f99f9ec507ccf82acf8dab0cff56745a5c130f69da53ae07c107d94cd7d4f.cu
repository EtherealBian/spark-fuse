#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
using T=__nv_bfloat16; using Bits=unsigned short;
template<int H,int Off> __device__ __forceinline__ void query(const T*x,const T*c,const T*s,T*y,int i){
  const int row=i/64, j=i%64, batch=row/(H*3), token=row%3, head=(row/3)%H;
  const int in=(batch*3+token)*2048+Off+head*128, out=row*128;
  if(j<64){
    const int p=((batch*3+token)*128)+j;
    const float l=float(x[in+j]), h=float(x[in+j+64]);
    const T lc=T(l*float(c[p])),ls=T((-h)*float(s[p]));
    const T hc=T(h*float(c[p+64])),hs=T(l*float(s[p+64]));
    y[out+j]=T(float(lc)+float(ls)); y[out+j+64]=T(float(hc)+float(hs));
  } else {const int k=j+64;reinterpret_cast<Bits*>(y)[out+k]=reinterpret_cast<const Bits*>(x)[in+k];}
}
template<int H,int Off> __device__ __forceinline__ void key(const T*x,const T*c,const T*s,T*y,int i){
  const int row=i/64, j=i%64, batch=row/(H*3), token=row%3, head=(row/3)%H;
  const int in=(batch*3+token)*2048+Off+head*128, out=((row/3)*134+131+row%3)*128;
  if(j<64){
    const int p=((batch*3+token)*128)+j;
    const float l=float(x[in+j]), h=float(x[in+j+64]);
    const T lc=T(l*float(c[p])),ls=T((-h)*float(s[p]));
    const T hc=T(h*float(c[p+64])),hs=T(l*float(s[p+64]));
    y[out+j]=T(float(lc)+float(ls)); y[out+j+64]=T(float(hc)+float(hs));
  } else {const int k=j+64;reinterpret_cast<Bits*>(y)[out+k]=reinterpret_cast<const Bits*>(x)[in+k];}
}
using Packet=uint4;
template<int BS,int HS,int TS> __device__ __forceinline__ void history(const Bits*x,Bits*y,int i) {
  constexpr int W=8, D=128, S=131, T=3, H=2, PK=D/W;
  if(i>=8384)return;
  const int row=i/PK, token=row%S, head=(row/S)%H, batch=row/(H*S), col=(i%PK)*W;
  const Bits* from=x+batch*BS+head*HS+token*TS+col;
  Bits* to=y+((batch*H+head)*(S+T)+token)*D+col;
  if(((uintptr_t(from)|uintptr_t(to)) & (sizeof(Packet)-1))==0)
    *reinterpret_cast<Packet*>(to)=*reinterpret_cast<const Packet*>(from);
  else {
    #pragma unroll
    for(int j=0;j<W;++j)to[j]=from[j];
  }
}
__global__ void sliced_cache_update(const T*x,const T*c,const T*s,const Bits*pk,const Bits*pv,T*q,T*k,Bits*v) {
  int block=blockIdx.x, i=block*128+threadIdx.x;
  if(block<36){if(i<4608)query<12,0>(x,c,s,q,i);return;}
  if((block-=36)<6){i=block*128+threadIdx.x;if(i<768)key<2,1536>(x,c,s,k,i);return;}
  if((block-=6)<12){
    i=block*128+threadIdx.x;
    if(i<1536){const int row=i/128,col=i%128,token=row%3,head=(row/3)%2,batch=row/(3*2);
      v[((batch*2+head)*134+131+token)*128+col]=reinterpret_cast<const Bits*>(x)[(batch*3+token)*2048+1792+head*128+col];}
    return;
  }
  if((block-=12)<66){history<33536,16768,128>(pk,reinterpret_cast<Bits*>(k),block*128+threadIdx.x);return;}
  history<33536,16768,128>(pv,v,(block-66)*128+threadIdx.x);
}
extern "C" int launch(void**p,int,void*stream){
  sliced_cache_update<<<186,128,0,(cudaStream_t)stream>>>((const T*)p[0],(const T*)p[1],(const T*)p[2],(const Bits*)p[3],(const Bits*)p[4],(T*)p[5],(T*)p[6],(Bits*)p[7]);
  return int(cudaGetLastError());
}
