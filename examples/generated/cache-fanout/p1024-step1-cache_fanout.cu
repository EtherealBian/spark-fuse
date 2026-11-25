#include <cuda_runtime.h>
#include <stdint.h>
// Copy raw packets, not floating-point values. Old and incoming cache layouts may differ.
template<int W,typename P,bool old_sw,bool new_sw>
__device__ __forceinline__ void copy_one(const P* old,const P* next,P* cache,P* repeated,int i) {
  constexpr int packs=128/W, plane=1025*packs;
  int bh=i/plane, batch=bh/2, head=bh%2, tok=(i%plane)/packs, j=i%packs;
  P value;
  if(tok<1024) { int row=old_sw ? (batch*1024+tok)*2+head : bh*1024+tok; value=old[row*packs+j]; }
  else { int q=tok-1024, row=new_sw ? (batch*1+q)*2+head : bh*1+q; value=next[row*packs+j]; }
  cache[i]=value;
  P* dst=repeated+bh*(6*plane)+i%plane;
  #pragma unroll 6
  for(int group=0;group<6;++group) dst[group*plane]=value;
}
template<int W,typename P>
__global__ void cache_fanout(const P* k,const P* kt,const P* v,const P* vt,P* ko,P* vo,P* kr,P* vr) {
  int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=524800/W) return;
  if(blockIdx.y==0)copy_one<W,P,true,false>(k,kt,ko,kr,i);
  else copy_one<W,P,true,false>(v,vt,vo,vr,i);
}
extern "C" int launch(void** p,int,void* stream) {
uintptr_t addr=0; for(int j=0;j<8;++j)addr|=uintptr_t(p[j]);
if((addr & 15)==0){cache_fanout<8,uint4><<<dim3(513,2),128,0,(cudaStream_t)stream>>>((const uint4*)p[0],(const uint4*)p[1],(const uint4*)p[2],(const uint4*)p[3],(uint4*)p[4],(uint4*)p[5],(uint4*)p[6],(uint4*)p[7]);return int(cudaGetLastError());}
if((addr & 7)==0){cache_fanout<4,uint2><<<dim3(1025,2),128,0,(cudaStream_t)stream>>>((const uint2*)p[0],(const uint2*)p[1],(const uint2*)p[2],(const uint2*)p[3],(uint2*)p[4],(uint2*)p[5],(uint2*)p[6],(uint2*)p[7]);return int(cudaGetLastError());}
if((addr & 3)==0){cache_fanout<2,unsigned int><<<dim3(2050,2),128,0,(cudaStream_t)stream>>>((const unsigned int*)p[0],(const unsigned int*)p[1],(const unsigned int*)p[2],(const unsigned int*)p[3],(unsigned int*)p[4],(unsigned int*)p[5],(unsigned int*)p[6],(unsigned int*)p[7]);return int(cudaGetLastError());}
{cache_fanout<1,unsigned short><<<dim3(4100,2),128,0,(cudaStream_t)stream>>>((const unsigned short*)p[0],(const unsigned short*)p[1],(const unsigned short*)p[2],(const unsigned short*)p[3],(unsigned short*)p[4],(unsigned short*)p[5],(unsigned short*)p[6],(unsigned short*)p[7]);return int(cudaGetLastError());}
}
