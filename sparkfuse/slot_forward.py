"""Eliminate dead functional cache scatters and forward matching per-row writes into gathers."""

from dataclasses import dataclass
import torch
from torch.utils import _pytree as tree
from .decode_attention import _require
from .index_program import IndexProgram
from .native import native_dispatch


@dataclass(frozen=True)
class SlotPlan:
    batch: int
    groups: int
    capacity: int
    width: int
    visible: tuple
    dtype: torch.dtype
    device: torch.device

    @property
    def bank(self):
        return (self.batch, self.groups, self.capacity, self.width)

    @property
    def token(self):
        return (self.batch, self.groups, 1, self.width)

    @property
    def inputs(self):
        return (self.bank, self.bank, self.token, self.token, (self.batch,), (self.batch,))

    @property
    def outputs(self):
        return (self.token, self.token) + tuple(self.bank for v in self.visible if v)


def recognize(ep):
    ns = list(ep.graph.nodes)
    ins = [n for n in ns if n.op == "placeholder"]
    _require(
        len(ins) == 6
        and ns[:6] == ins
        and not ep.range_constraints
        and len(ep.graph_signature.input_specs) == 6
        and all(x.kind.name == "USER_INPUT" for x in ep.graph_signature.input_specs)
        and all(x.kind.name == "USER_OUTPUT" for x in ep.graph_signature.output_specs)
        and ep.call_spec.in_spec == tree.tree_flatten((tuple(range(6)), {}))[1],
        "expected static functional six-operand export",
    )
    meta = [x.meta.get("val") for x in ins]
    _require(
        all(
            isinstance(m, torch.Tensor)
            and all(type(v) is int and v > 0 for v in m.shape)
            and m.is_contiguous()
            and m.layout == torch.strided
            and not m.is_neg()
            and not m.is_conj()
            and m.device == meta[0].device
            for m in meta
        ),
        "dense static input metadata required",
    )
    k = meta[0]
    _require(
        k.ndim == 4
        and k.device.type == "cuda"
        and k.dtype in (torch.bfloat16, torch.float16)
        and all(m.dtype == k.dtype for m in meta[:4])
        and all(m.dtype == torch.int32 for m in meta[4:]),
        "expected CUDA BF16/FP16 banks and int32 positions",
    )
    b, g, s, d = k.shape
    _require(
        b <= 256 and g <= 128 and s <= 32768 and d in (32, 64, 128, 256) and b * g * s * d < 2**29,
        "unsupported slot geometry",
    )
    p = SlotPlan(b, g, s, d, (), k.dtype, k.device)
    _require(
        tuple(tuple(m.shape) for m in meta) == p.inputs, "bank, token or position shape mismatch"
    )
    ops = iter(ns[6:])
    aten = torch.ops.aten

    def take(fn, args, kwargs=None):
        n = next(ops, None)
        _require(
            n is not None
            and n.op == "call_function"
            and n.target == fn
            and n.args == args
            and n.kwargs == (kwargs or {}),
            "expected " + str(fn),
        )
        return n

    def index(x):
        z = take(aten.clamp.default, (x, 0, s - 1))
        take(
            aten._assert_tensor_metadata.default,
            (z,),
            {"dtype": torch.int32, "device": k.device, "layout": torch.strided},
        )
        z = take(aten.to.dtype, (z, torch.int64))
        z = take(aten.reshape.default, (z, [b, 1, 1, 1]))
        return take(aten.expand.default, (z, [b, g, 1, d]))

    wi, ri = map(index, ins[4:])
    banks = [
        take(aten.scatter.src, (old, 2, wi, new))
        for old, new in zip(ins[:2], ins[2:4], strict=True)
    ]
    reads = tuple(take(aten.gather.default, (x, 2, ri)) for x in banks)
    out = next(ops, None)
    _require(
        out is not None
        and out.op == "output"
        and len(out.args) == 1
        and type(out.args[0]) is tuple
        and not out.kwargs
        and next(ops, None) is None,
        "expected flat returned tensors",
    )
    ret = out.args[0]
    vis = tuple(x in ret for x in banks)
    _require(
        ret == reads + tuple(x for x, used in zip(banks, vis, strict=True) if used)
        and ep.call_spec.out_spec == tree.tree_flatten(tuple(range(len(ret))))[1],
        "unsupported output order, escape or alias",
    )
    return SlotPlan(b, g, s, d, vis, k.dtype, k.device)


def emit(p):
    # Pointers are to raw storage; no floating-point conversion may canonicalize NaN payloads.
    b, g, s, d = p.bank
    sections = []
    for j, visible in enumerate(p.visible):
        if visible:
            sections.append(f"""if(i<N){{int row=i/(S*P), token=i/P%S, off=i%P, w=clamp_slot(wi[row/G]);
      bank{j}[i]=token==w ? t{j}[row*P+off] : a{j}[i];return;}}i-=N;""")
    fmt = lambda j: f"PT({8 + sum(p.visible[:j])})" if p.visible[j] else "nullptr"
    return f"""#include <cuda_runtime.h>
#include <stdint.h>
namespace {{
constexpr int B={b},G={g},S={s},D={d};
__device__ __forceinline__ int clamp_slot(int v){{return min(S-1,max(0,v));}}
template<class T> __global__ void slots(const T* a0,const T* a1,const T* t0,const T* t1,const int* wi,const int* ri,T* y0,T* y1,T* bank0,T* bank1){{
  constexpr int P=D*2/sizeof(T), R=B*G*P,N=R*S; int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<R){{int row=i/P,r=clamp_slot(ri[row/G]),w=clamp_slot(wi[row/G]);int base=(row*S+r)*P+i%P;
    y0[i]=w==r?t0[i]:a0[base];y1[i]=w==r?t1[i]:a1[base];return;}}i-=R;
  {"".join(sections)}
}}
template<class T> void run(void** p,cudaStream_t stream){{constexpr int N=B*G*(D*2/sizeof(T))*(1+S*{sum(p.visible)});
  #define PT(i) static_cast<T*>(p[i])
  slots<T><<<(N+127)/128,128,0,stream>>>(PT(0),PT(1),PT(2),PT(3),static_cast<int*>(p[4]),static_cast<int*>(p[5]),PT(6),PT(7),{fmt(0)},{fmt(1)});
  #undef PT
}}
}}
extern "C" int launch(void** p,int,void* stream){{uintptr_t addr=0;for(int i=0;i<{6 + len(p.outputs)};++i)if(i!=4&&i!=5)addr|=uintptr_t(p[i]);
  if(!(addr&15))run<uint4>(p,(cudaStream_t)stream);else if(!(addr&3))run<unsigned int>(p,(cudaStream_t)stream);else run<unsigned short>(p,(cudaStream_t)stream);return int(cudaGetLastError());}}
"""


class SlotForward:
    def __call__(self, *xs):
        if not torch.is_inference_mode_enabled() or any(type(x) is not torch.Tensor for x in xs):
            raise ValueError("inference mode and exact tensors required")
        return self.run(*xs)


def compile_slot_forward(ep, example, audit):
    p = recognize(ep)
    _require(torch.is_inference_mode_enabled(), "inference mode required")
    a, b = tuple(example), tuple(audit)
    _require(
        len(a) == len(b) == 6
        and all(
            type(v) is torch.Tensor
            and v.device == p.device
            and v.dtype == (p.dtype if i < 4 else torch.int32)
            and v.shape == sh
            and v.layout == torch.strided
            and v.is_contiguous()
            and not v.is_conj()
            and not v.is_neg()
            for xs in (a, b)
            for i, (v, sh) in enumerate(zip(xs, p.inputs, strict=True))
        ),
        "invalid qualification operands",
    )
    _require(
        {v.untyped_storage().data_ptr() for v in a}.isdisjoint(
            v.untyped_storage().data_ptr() for v in b
        ),
        "two independent qualifications required",
    )
    dt = "bf16" if p.dtype == torch.bfloat16 else "f16"
    z = SlotForward()
    z.plan = p
    z.program = IndexProgram(
        p.inputs, p.outputs, dt, p.device, emit(p), input_dtypes=[dt] * 4 + ["i32", "i32"]
    )
    z.run = native_dispatch(z.program)
    m = ep.module()
    for xs in (a, b):
        ref = m(*xs)
        ys = z(*xs)
        ptrs = {x.untyped_storage().data_ptr() for x in xs}
        _require(
            all(
                y.untyped_storage().data_ptr() not in ptrs
                and y._base is None
                and y.storage_offset() == 0
                and y.stride() == t.stride()
                and y.untyped_storage().nbytes() == y.numel() * y.element_size()
                and torch.equal(y.view(torch.int16), t.view(torch.int16))
                for y, t in zip(ys, ref, strict=True)
            )
            and len({y.untyped_storage().data_ptr() for y in ys}) == len(ys),
            "slot-forward bytes or storage mismatch",
        )
    return z
