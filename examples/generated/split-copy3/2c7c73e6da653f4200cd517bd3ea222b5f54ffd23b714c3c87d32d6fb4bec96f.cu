#include <cuda_runtime.h>
#include <stdint.h>
using E=unsigned short;
template <int K, class P> __global__ void split_copy3(const E* __restrict__ src, E* __restrict__ a, E* __restrict__ b, E* __restrict__ c) {
  int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=4194304/K)return;
  int row=i/(2048/K),col=(i%(2048/K))*K;
  E* dst;
  if(col<1536) dst=a+row*1536+col;
  else if(col<1792) dst=b+(row*256+(col-1536));
  else dst=c+(row*256+(col-1792));
  *reinterpret_cast<P*>(dst)=reinterpret_cast<const P*>(src)[i];
}
extern "C" int launch(void** p,int,void* stream) {
  auto *src=(const E*)p[0]; auto *a=(E*)p[1], *b=(E*)p[2], *c=(E*)p[3];
  uintptr_t addr=uintptr_t(src)|uintptr_t(a)|uintptr_t(b)|uintptr_t(c);
  if(!(addr & 15)) { split_copy3<8,uint4><<<4096,128,0,(cudaStream_t)stream>>>(src,a,b,c); return int(cudaGetLastError()); }
if(!(addr & 7)) { split_copy3<4,uint2><<<8192,128,0,(cudaStream_t)stream>>>(src,a,b,c); return int(cudaGetLastError()); }
if(!(addr & 3)) { split_copy3<2,unsigned int><<<16384,128,0,(cudaStream_t)stream>>>(src,a,b,c); return int(cudaGetLastError()); }
{ split_copy3<1,unsigned short><<<32768,128,0,(cudaStream_t)stream>>>(src,a,b,c); return int(cudaGetLastError()); }
}
