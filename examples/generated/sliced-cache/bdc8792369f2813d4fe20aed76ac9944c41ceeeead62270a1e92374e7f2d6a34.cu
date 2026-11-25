#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
using T=__nv_bfloat16; using Bits=unsigned short;
template<int H,int Off> __device__ __forceinline__ void query(const T*x,const T*c,const T*s,T*y,int i){
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
template<int H,int Off> __device__ __forceinline__ void key(const T*x,const T*c,const T*s,T*y,int i){
  const int row=i/96, j=i%96, batch=row/(H*1), token=row%1, head=(row/1)%H;
  const int in=(batch*1+token)*2048+Off+head*128, out=((row/1)*1025+1024+row%1)*128;
  if(j<32){
    const int p=((batch*1+token)*64)+j;
    const float l=float(x[in+j]), h=float(x[in+j+32]);
    const T lc=T(l*float(c[p])),ls=T((-h)*float(s[p]));
    const T hc=T(h*float(c[p+32])),hs=T(l*float(s[p+32]));
    y[out+j]=T(float(lc)+float(ls)); y[out+j+32]=T(float(hc)+float(hs));
  } else {const int k=j+32;reinterpret_cast<Bits*>(y)[out+k]=reinterpret_cast<const Bits*>(x)[in+k];}
}
using Packet=uint4;
template<int BS,int HS,int TS> __device__ __forceinline__ void history(const Bits*x,Bits*y,int i) {
  constexpr int W=8, D=128, S=1024, T=1, H=2, PK=D/W;
  if(i>=65536)return;
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
  if(block<18){if(i<2304)query<12,0>(x,c,s,q,i);return;}
  if((block-=18)<3){i=block*128+threadIdx.x;if(i<384)key<2,1536>(x,c,s,k,i);return;}
  if((block-=3)<4){
    i=block*128+threadIdx.x;
    if(i<512){const int row=i/128,col=i%128,token=row%1,head=(row/1)%2,batch=row/(1*2);
      v[((batch*2+head)*1025+1024+token)*128+col]=reinterpret_cast<const Bits*>(x)[(batch*1+token)*2048+1792+head*128+col];}
    return;
  }
  if((block-=4)<512){history<262144,131072,128>(pk,reinterpret_cast<Bits*>(k),block*128+threadIdx.x);return;}
  history<2097152,128,2048>(pv,v,(block-512)*128+threadIdx.x);
}
extern "C" int launch(void**p,int,void*stream){
  sliced_cache_update<<<1049,128,0,(cudaStream_t)stream>>>((const T*)p[0],(const T*)p[1],(const T*)p[2],(const Bits*)p[3],(const Bits*)p[4],(T*)p[5],(T*)p[6],(Bits*)p[7]);
  return int(cudaGetLastError());
}
