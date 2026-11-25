#include <cuda_runtime.h>
#include <cstdint>
namespace {
constexpr int count = 21, tile = 4096;
__device__ __constant__ int ends[count] = {25,38,39,56,73,90,107,124,133,142,151,160,169,178,183,192,201,210,219,228,233};
__device__ __constant__ size_t bytes[count] = {101376ULL,50688ULL,132ULL,67584ULL,67584ULL,67584ULL,67584ULL,67584ULL,33792ULL,33792ULL,33792ULL,33792ULL,33792ULL,33792ULL,16896ULL,33792ULL,33792ULL,33792ULL,33792ULL,33792ULL,16896ULL};
struct Pointers { void* p[count * 2]; };
#if __CUDA_ARCH__ >= 700
#define SF_GRID_CONST __grid_constant__
#else
#define SF_GRID_CONST
#endif
__global__ void sparkfuse_slot_copy(const SF_GRID_CONST Pointers ptr) {
  int i = 0;
  while (blockIdx.x >= ends[i]) ++i;  // The launch grid ends exactly at ends[count-1].
  const size_t offset = size_t(blockIdx.x - (i ? ends[i-1] : 0)) * tile;
  const size_t remain = bytes[i] - offset;
  const unsigned n = remain < tile ? static_cast<unsigned>(remain) : tile;
  auto* dst = static_cast<unsigned char*>(ptr.p[i]) + offset;
  auto* src = static_cast<const unsigned char*>(ptr.p[count + i]) + offset;
  if (((reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src)) & 15) == 0) {
    auto* d = reinterpret_cast<uint4*>(dst);
    auto* s = reinterpret_cast<const uint4*>(src);
    for (unsigned j = threadIdx.x; j < n / 16; j += blockDim.x) d[j] = s[j];
    if (threadIdx.x < n % 16) dst[n / 16 * 16 + threadIdx.x] = src[n / 16 * 16 + threadIdx.x];
  } else {
    for (unsigned j = threadIdx.x; j < n; j += blockDim.x) dst[j] = src[j];
  }
}
}
extern "C" int launch(void** ptr, int n, void* stream) {
  if (n != count) return int(cudaErrorInvalidValue);
  Pointers args;
  for (int j = 0; j < 2*count; ++j) args.p[j] = ptr[j];
  sparkfuse_slot_copy<<<233, 128, 0, static_cast<cudaStream_t>(stream)>>>(args);
  return int(cudaGetLastError());
}
