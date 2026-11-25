"""Generate one byte-preserving gather/scatter launch for fixed, independent resident outputs."""

import ctypes
from hashlib import sha256
from itertools import accumulate

from .ir import Unsupported
from .runtime import build

TILE = 4096
MAX_OUTPUTS = 32


def generate(sizes):
    sizes = tuple(sizes)
    if not 1 <= len(sizes) <= MAX_OUTPUTS or any(type(n) is not int or n <= 0 for n in sizes):
        raise Unsupported("CUDA slot copy needs 1..32 nonempty whole outputs")
    ends = list(accumulate((n - 1) // TILE + 1 for n in sizes))
    if ends[-1] > 2**31 - 1:
        raise Unsupported("CUDA slot copy exceeds the block-index limit")
    return (
        """#include <cuda_runtime.h>
#include <cstdint>
namespace {
constexpr int count = COUNT, tile = TILE;
__device__ __constant__ int ends[count] = {ENDS};
__device__ __constant__ size_t bytes[count] = {BYTES};
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
  sparkfuse_slot_copy<<<BLOCKS, 128, 0, static_cast<cudaStream_t>(stream)>>>(args);
  return int(cudaGetLastError());
}
""".replace("COUNT", str(len(sizes)))
        .replace("TILE", str(TILE))
        .replace("ENDS", ",".join(map(str, ends)))
        .replace("BYTES", ",".join(f"{n}ULL" for n in sizes))
        .replace("BLOCKS", str(ends[-1]))
    )


def prepare(outputs, device):
    sizes = [t.nbytes for t in outputs]
    source = generate(sizes)
    lib, _ = build(source, device)
    return (
        lib,
        ctypes.cast(lib.launch, ctypes.c_void_p).value,
        dict(output_bytes=sizes, tile_bytes=TILE, cuda_sha256=sha256(source.encode()).hexdigest()),
    )
