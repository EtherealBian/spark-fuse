#include <cuda_runtime.h>
#include <stdint.h>
template<int W,typename P> __global__ void paired_head_repeat(const P* a,const P* b,P* ya,P* yb) {
  constexpr int plane=262144/W;
  int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=4*plane)return;
  P value=(blockIdx.y?b:a)[i];
  P* dst=(blockIdx.y?yb:ya)+(i/plane)*(6*plane)+i%plane;
  #pragma unroll 6
  for(int j=0;j<6;++j)dst[j*plane]=value;
}
extern "C" int launch(void** p,int,void* stream) {
  uintptr_t addr=0;for(int i=0;i<4;++i)addr|=uintptr_t(p[i]);
  if((addr & 15)==0){ paired_head_repeat<8,uint4><<<dim3(1024,2),128,0,(cudaStream_t)stream>>>((const uint4*)p[0],(const uint4*)p[1],(uint4*)p[2],(uint4*)p[3]);return int(cudaGetLastError()); }
  if((addr & 7)==0){ paired_head_repeat<4,uint2><<<dim3(2048,2),128,0,(cudaStream_t)stream>>>((const uint2*)p[0],(const uint2*)p[1],(uint2*)p[2],(uint2*)p[3]);return int(cudaGetLastError()); }
  if((addr & 3)==0){ paired_head_repeat<2,unsigned int><<<dim3(4096,2),128,0,(cudaStream_t)stream>>>((const unsigned int*)p[0],(const unsigned int*)p[1],(unsigned int*)p[2],(unsigned int*)p[3]);return int(cudaGetLastError()); }
  { paired_head_repeat<1,unsigned short><<<dim3(8192,2),128,0,(cudaStream_t)stream>>>((const unsigned short*)p[0],(const unsigned short*)p[1],(unsigned short*)p[2],(unsigned short*)p[3]);return int(cudaGetLastError()); }
}
