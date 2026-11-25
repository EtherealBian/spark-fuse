#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
using E=__nv_bfloat16;
template<int V> struct alignas(2*V) Packet { E lane[V]; };
template<int V> __global__ void packed_swiglu(const E* __restrict__ x,E* __restrict__ y,E* __restrict__ act) {
  const int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=2275840/V) return;
  const int row=i/(8960/V),col=(i%(8960/V))*V;
  using P=Packet<V>;const E* base=x+row*17920+col;
  const P g=*reinterpret_cast<const P*>(base+0);
  const P u=*reinterpret_cast<const P*>(base+8960); P z,a;
  #pragma unroll
  for(int j=0;j<V;++j) {const float f=__bfloat162float(g.lane[j]);
    a.lane[j]=__float2bfloat16_rn(f/(1.f+expf(-f)));
    z.lane[j]=__float2bfloat16_rn(__bfloat162float(a.lane[j])*__bfloat162float(u.lane[j]));
  }
  reinterpret_cast<P*>(y)[i]=z;
  if constexpr (false) reinterpret_cast<P*>(act)[i]=a;
}
extern "C" int launch(void** p,int,void* stream) {
  auto* x=(const E*)p[0]; auto* y=(E*)p[1];uintptr_t addr=uintptr_t(x)|uintptr_t(y); E* a=nullptr;
  {packed_swiglu<1><<<17780,128,0,(cudaStream_t)stream>>>(x,y,a); return int(cudaGetLastError());}
}
