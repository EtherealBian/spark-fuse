"""Packed memory schedule; unaligned (but contiguous) tensor views remain legal."""

from .codegen import CT, LOAD, STORE

HEADER = """#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
template<class T,int V> struct alignas(sizeof(T)*V) Pack { T x[V]; };
template<class T,int V> __device__ __forceinline__ Pack<T,V> loadv(const T* p){
  using P=Pack<T,V>;
  if((uintptr_t(p)&(alignof(P)-1))==0) return *reinterpret_cast<const P*>(p);
  P q;
  #pragma unroll
  for(int i=0;i<V;i++)q.x[i]=p[i]; return q;
}
template<class T,int V> __device__ __forceinline__ void storev(T* p,Pack<T,V> q){
  using P=Pack<T,V>;
  if((uintptr_t(p)&(alignof(P)-1))==0){ *reinterpret_cast<P*>(p)=q; return; }
  #pragma unroll
  for(int i=0;i<V;i++)p[i]=q.x[i];
}"""


TAIL_HEADER = """
template<class T,int V> __device__ __forceinline__ Pack<T,V> load_tail(const T* p,int remain){
  if(remain>=V) return loadv<T,V>(p);
  Pack<T,V> value{};
  #pragma unroll
  for(int i=0;i<V;i++) if(i<remain) value.x[i]=p[i];
  return value;
}
template<class T,int V> __device__ __forceinline__ void store_tail(T* p,Pack<T,V> value,int remain){
  if(remain>=V){ storev(p,value); return; }
  #pragma unroll
  for(int i=0;i<V;i++) if(i<remain) p[i]=value.x[i];
}
"""


def generate_vector(p, s, emit, index):
    h = p.width
    v = s.vec
    tail = h % v != 0
    packets = (h + v - 1) // v
    t = s.threads
    warp = s.layout == "warp"
    flat = s.layout == "flat"
    # If all tensors are scalars or span the full domain, vector packets can
    # cross row boundaries. This avoids one partial pack per odd-width row.
    linear = flat and tail and all(p.vals[i].shape in ((), p.domain) for i in p.inputs)
    extent = f"rows*{h}" if linear else str(h)
    stride = 32 if warp else t
    params = [f"const {CT[p.vals[i].dt]}* __restrict__ in{k}" for k, i in enumerate(p.inputs)] + [
        f"{CT[p.vals[i].dt]}* __restrict__ out{k}" for k, i in enumerate(p.outputs)
    ]
    src = [
        HEADER + (TAIL_HEADER if tail else ""),
        "__global__ void fused(" + ", ".join(params + ["int rows"]) + ") {",
    ]
    if linear:
        src += [
            f"const int packet=blockIdx.x*blockDim.x+threadIdx.x; if(packet>=(rows*{h}-1)/{v}+1)return; const int row=0, first=packet*{v};"
        ]
    elif flat:
        src += [
            f"const int packet=blockIdx.x*blockDim.x+threadIdx.x;if(packet>=rows*{packets})return; const int row=packet/{packets}, first=(packet%{packets})*{v};"
        ]
    else:
        row = "blockIdx.x*(blockDim.x/32)+threadIdx.x/32" if warp else "blockIdx.x"
        lane = "threadIdx.x&31" if warp else "threadIdx.x"
        src += [f"const int row={row}; if(row>=rows)return; const int lane={lane};"]
    loop = f"for(int first=lane*{v};first<{h};first+={stride * v}) {{"
    inner = f"#pragma unroll\nfor(int q=0;q<{v};q++){{ const int j=first+q, pos=row*{h}+j;"
    if tail:
        inner = f"#pragma unroll\nfor(int q=0;q<{v};q++){{ const int j=first+q; if(j>={extent})continue; const int pos=row*{h}+j;"
    pre = {}

    def packed(ids, known):
        active = set()

        def walk(i):
            if i in known or i in active:
                return
            active.add(i)
            for d in p.vals[i].deps:
                walk(d)

        for i in ids:
            walk(i)
        lines = []
        extended = dict(known)
        for i in p.inputs:
            n = p.vals[i]
            ix = index(n)
            if i in active and ix in ("pos", "j"):
                offset = "first" if ix == "j" else f"row*{h}+first"
                k = n.imm
                load = "load_tail" if tail else "loadv"
                extra = f",{extent}-first" if tail else ""
                lines.append(f"auto pack{i}={load}<{CT[n.dt]},{v}>(in{k}+{offset}{extra});")
                extended[i] = LOAD[n.dt](f"pack{i}.x[q]")
        return lines, extended

    if p.red is not None:
        operand = p.vals[p.red].deps[0]
        op = p.vals[operand]
        saved = op.deps[0] if op.op == "square" else operand
        targets = [operand, saved] if s.cache else [operand]
        loads, known = packed(targets, {})
        off = f"(first/{stride * v})*{v}+q"
        if s.cache:
            src += [f"float held[{((h + stride * v - 1) // (stride * v)) * v}];"]
        text, names = emit(targets, known)
        src += ["float acc=0.f;", loop, *loads, inner, text, "acc+=" + names[0] + ";"]
        if s.cache:
            src += [f"held[{off}]={names[1]};"]
            pre[saved] = f"held[{off}]"
        src += ["}}", "for(int d=16;d;d>>=1)acc+=__shfl_down_sync(0xffffffff,acc,d);"]
        if warp:
            src += ["float sum=__shfl_sync(0xffffffff,acc,0);"]
        else:
            src += [
                f"__shared__ float parts[{t // 32}];",
                "if((lane&31)==0)parts[lane/32]=acc; __syncthreads();",
                f"if(lane<32){{acc=lane<{t // 32}?parts[lane]:0.f;for(int d=16;d;d>>=1)acc+=__shfl_down_sync(0xffffffff,acc,d);if(lane==0)parts[0]=acc;}}__syncthreads();float sum=parts[0];",
            ]
        dt = p.vals[p.red].dt
        src += ["float reduced=" + LOAD[dt](STORE[dt](f"sum*(1.f/{h}.f)")) + ";"]
        pre[p.red] = "reduced"
    full = []
    small = []
    for k, i in enumerate(p.outputs):
        (full if p.vals[i].shape == p.domain else small).append((k, i))
    if small:
        text, names = emit([i for k, i in small], {p.red: "reduced"} if p.red is not None else {})
        src += [f"if(lane==0){{const int j=0,pos=row*{h};", text]
        for (k, i), nm in zip(small, names):
            src += ["out" + str(k) + "[row]=" + STORE[p.vals[i].dt](nm) + ";"]
        src += ["}"]
    if full:
        ids = [i for k, i in full]
        loads, known = packed(ids, pre)
        text, names = emit(ids, known)
        src += ([] if flat else [loop]) + loads
        for k, i in full:
            init = "{}" if tail else ""
            src += [f"Pack<{CT[p.vals[i].dt]},{v}> dst{k}{init};"]
        src += [inner, text]
        for (k, i), nm in zip(full, names):
            src += [f"dst{k}.x[q]=" + STORE[p.vals[i].dt](nm) + ";"]
        src += ["}"]
        for k, i in full:
            store = "store_tail" if tail else "storev"
            extra = f",{extent}-first" if tail else ""
            src += [f"{store}(out{k}+row*{h}+first,dst{k}{extra});"]
        if not flat:
            src += ["}"]
    blocks = f"(rows*{packets}-1)/{t}+1" if flat else f"(rows-1)/{t // 32}+1" if warp else "rows"
    if linear:
        blocks = f"((rows*{h}-1)/{v})/{t}+1"
    args = [f"({CT[p.vals[i].dt]}*)ptr[{k}]" for k, i in enumerate(p.inputs + p.outputs)]
    src += [
        "}",
        'extern "C" int launch(void **ptr,int rows,void *stream){',
        f"fused<<<{blocks},{t},0,(cudaStream_t)stream>>>("
        + ",".join(args + ["rows"])
        + ");return (int)cudaGetLastError();",
        "}",
    ]
    return "\n".join(src) + "\n"
