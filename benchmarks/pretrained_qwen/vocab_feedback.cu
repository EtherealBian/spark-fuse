// Batch-one BF16 vocabulary reduction. Exact ties return the smaller token ID.
#include <cuda_runtime.h>
#include <cstdint>
using U = unsigned long long;
constexpr int block = 128, tile = 512;
__device__ __forceinline__ unsigned rank_bits(unsigned x) {
    if ((x & 0x7fffu) > 0x7f80u) return 65536u; // all NaNs share the highest rank
    if ((x & 0x7fffu) == 0) return 0x8000u;     // signed zeros compare equal
    return x & 0x8000u ? (x ^ 0xffffu) : (x ^ 0x8000u);
}
__device__ __forceinline__ U warpmax(U v) {
    for (int d=16; d; d>>=1) { U alt=__shfl_down_sync(0xffffffffu, v,d); v=alt>v?alt:v; }
    return v;
}
__device__ __forceinline__ U reduce(U v, U *shared) {
    v=warpmax(v); const int lane=threadIdx.x & 31;
    if (!lane) shared[threadIdx.x>>5]=v;
    __syncthreads();
    if (threadIdx.x<32) v=warpmax(lane<block/32?shared[lane]:0ull);
    return v; // only thread zero consumes this value
}
__global__ void partial(const uint16_t *__restrict__ logits,U *__restrict__ scratch,int n) {
    __shared__ U groups[block/32]; U best=0;
    for (int j=0;j<tile;j+=block) { int i=blockIdx.x*tile+threadIdx.x+j;
        if(i<n){U k=(U(rank_bits(logits[i]))<<32) | (0xffffffffu-unsigned(i));best=k>best?k:best;}}
    best=reduce(best,groups);if(!threadIdx.x)scratch[blockIdx.x]=best;
}
__global__ void commit(const U *scratch,int tiles,long long *token,long long *pos,long long *dst,long long *history,int capacity) {
    __shared__ U groups[block/32]; U best=0;
    for(int i=threadIdx.x;i<tiles;i+=blockDim.x){U x=scratch[i];best=x>best?x:best;}
    best=reduce(best,groups);
    if(!threadIdx.x){long long d=*dst,p=*pos;
        if(d<0||d>=capacity||p<0||p==0x7fffffffffffffffLL){*dst=-1;return;}
        long long id=0xffffffffu-unsigned(best); *token=id;history[d]=id;*pos=p+1;*dst=d+1;
    }
}
extern "C" int step(void **p,int n,int capacity,cudaStream_t stream){
    if(!p||n<1||n>1048576||capacity<1)return (int)cudaErrorInvalidValue;
    int blocks=(n+tile-1)/tile;partial<<<blocks,block,0,stream>>>((const uint16_t*)p[0],(U*)p[1],n);
    auto rc=cudaGetLastError();if(rc!=cudaSuccess)return (int)rc;
    commit<<<1,block,0,stream>>>((const U*)p[1],blocks,(long long*)p[2],(long long*)p[3],(long long*)p[4],(long long*)p[5],capacity);return (int)cudaGetLastError();
}
// Symbol expected by SparkFuse's toolchain loader; use the capacity-aware `step` ABI above.
extern "C" int launch(void **,int,cudaStream_t){return (int)cudaErrorNotSupported;}
