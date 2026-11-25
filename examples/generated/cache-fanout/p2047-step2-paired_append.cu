#include <cuda_runtime.h>
#include <stdint.h>
using E=unsigned short;
template<int W,typename P,bool guarded,int S,bool swapped>
__device__ __forceinline__ void move_bits(const E* x,E* out,int batch,int head,int token,int j) {
  const int row=swapped ? (batch*S+token)*2+head : (batch*2+head)*S+token;
  const E* p=x+row*128+j;
  if constexpr (guarded) {
    if((uintptr_t(p)|uintptr_t(out)) % sizeof(P)) {
      #pragma unroll
      for(int k=0;k<W;++k)out[k]=p[k];
      return;
    }
  }
  *reinterpret_cast<P*>(out)=*reinterpret_cast<const P*>(p);
}
template<int W,typename P,bool guarded,bool old_sw,bool new_sw>
__device__ __forceinline__ void append(const E* a,const E* b,E* y,int i) {
  constexpr int packets=128/W;
  int row=i/packets,j=(i%packets)*W,token=row%2049,head=(row/2049)%2,batch=row/(2*2049);
  if(token<2048) move_bits<W,P,guarded,2048,old_sw>(a,y+i*W,batch,head,token,j);
  else move_bits<W,P,guarded,1,new_sw>(b,y+i*W,batch,head,token-2048,j);
}
template<int W,typename P,bool guarded>
__global__ void paired_append(const E* k,const E* kt,const E* v,const E* vt,E* ko,E* vo) {
  int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=1049088/W)return;
  if(blockIdx.y==0)append<W,P,guarded,false,false>(k,kt,ko,i);
  else append<W,P,guarded,false,false>(v,vt,vo,i);
}
extern "C" int launch(void** p,int,void* stream) {
  uintptr_t addr=0; for(int i=0;i<6;++i) addr|=uintptr_t(p[i]);
  if((addr & 15)==0) { paired_append<8,uint4,false><<<dim3(1025,2),128,0,(cudaStream_t)stream>>>((const E*)p[0],(const E*)p[1],(const E*)p[2],(const E*)p[3],(E*)p[4],(E*)p[5]); return int(cudaGetLastError()); }
  if((addr & 7)==0) { paired_append<4,uint2,false><<<dim3(2049,2),128,0,(cudaStream_t)stream>>>((const E*)p[0],(const E*)p[1],(const E*)p[2],(const E*)p[3],(E*)p[4],(E*)p[5]); return int(cudaGetLastError()); }
  if((addr & 3)==0) { paired_append<2,unsigned int,false><<<dim3(4098,2),128,0,(cudaStream_t)stream>>>((const E*)p[0],(const E*)p[1],(const E*)p[2],(const E*)p[3],(E*)p[4],(E*)p[5]); return int(cudaGetLastError()); }
  { paired_append<1,unsigned short,false><<<dim3(8196,2),128,0,(cudaStream_t)stream>>>((const E*)p[0],(const E*)p[1],(const E*)p[2],(const E*)p[3],(E*)p[4],(E*)p[5]); return int(cudaGetLastError()); }
}
