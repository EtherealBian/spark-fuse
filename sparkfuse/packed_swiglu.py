"""SwiGLU on a single live packed projection; split views retain their original aliasing."""

from hashlib import sha256
from math import prod
import operator

import torch
from torch import nn
import torch.nn.functional as F
from .codegen import CT, LOAD, STORE
from .index_program import IndexProgram, IndexRegion
from .ir import DT
from .rotary import _is, _tensor

A = torch.ops.aten


class PackedSwiGLUProjection(nn.Module):
    """Explicit inference checkpoint: one authoritative [gate; up] weight, no retained sources.

    Forward returns the gated intermediate (the down projection is not part of this module).
    Editing the source linears after conversion intentionally has no effect. Update this
    module's registered weight, or call reload_weights_ between synchronized requests.
    """

    @staticmethod
    def _pack(gate, up):
        if (
            not all(
                isinstance(w, torch.Tensor) and w.layout == torch.strided and w.ndim == 2
                for w in (gate, up)
            )
            or gate.shape != up.shape
            or gate.dtype != up.dtype
            or gate.device != up.device
            or not gate.is_floating_point()
            or min(gate.shape) <= 0
        ):
            raise ValueError("expected two matching nonempty floating weight matrices")
        return torch.cat((gate.detach(), up.detach()), dim=0)

    def __init__(self, gate_weight, up_weight):
        super().__init__()
        self.weight = nn.Parameter(self._pack(gate_weight, up_weight), requires_grad=False)

    @classmethod
    def from_linears(cls, gate, up):
        if not all(isinstance(m, nn.Linear) and m.bias is None for m in (gate, up)):
            raise ValueError("bias-free gate and up linears required")
        return cls(gate.weight, up.weight)

    @torch.no_grad()
    def reload_weights_(self, gate, up):
        shape = (self.weight.shape[0] // 2, self.weight.shape[1])
        if not all(
            isinstance(w, torch.Tensor)
            and w.shape == shape
            and w.device == self.weight.device
            and w.dtype == self.weight.dtype
            for w in (gate, up)
        ):
            raise ValueError("checkpoint shape/dtype/device mismatch")
        self.weight.copy_(self._pack(gate, up))
        return self

    def forward(self, x):
        gate, up = F.linear(x, self.weight).chunk(2, dim=-1)
        return F.silu(gate) * up


def candidate(out, tensor_meta=_tensor):
    if not _is(out, A.mul.Tensor, 2):
        return None
    for act, other in (out.args, out.args[::-1]):
        if not _is(act, A.silu.default, 1):
            continue
        gate = act.args[0]
        if not all(_is(n, operator.getitem, 2) for n in (gate, other)):
            continue
        split = gate.args[0]
        if split is not other.args[0] or not any(
            _is(split, t, 3) for t in (A.chunk.default, A.split.Tensor, A.split_with_sizes.default)
        ):
            continue
        x, spec, axis = split.args
        v = tensor_meta(x)
        if (
            v is None
            or not v.ndim
            or v.dtype not in (torch.bfloat16, torch.float16)
            or not v.is_contiguous()
            or axis not in (-1, v.ndim - 1)
            or v.shape[-1] % 2
            or v.numel() > 2**30
        ):
            continue
        width = v.shape[-1] // 2
        shape = (*v.shape[:-1], width)
        if (
            split.target == A.chunk.default
            and spec != 2
            or split.target == A.split.Tensor
            and spec != width
            or split.target == A.split_with_sizes.default
            and (not isinstance(spec, (tuple, list)) or list(spec) != [width, width])
        ):
            continue
        indices = [n.args[1] for n in (gate, other)]
        if (
            not all(type(k) is int and -2 <= k <= 1 for k in indices)
            or indices[0] % 2 == indices[1] % 2
        ):
            continue
        vals = [tensor_meta(n) for n in (gate, other, act, out)]
        if any(
            z is None or z.shape != shape or z.dtype != v.dtype or z.device != v.device
            for z in vals
        ):
            continue
        canonical = tuple(prod(shape[j + 1 :]) for j in range(len(shape)))
        if any(tuple(z.stride()) != canonical for z in vals[2:]):
            continue
        return x, split, gate, other, act, shape, indices[0] % 2
    return None


def emit(shape, dt, gate_half, visible, vectorize=True, dynamic_rows=False):
    w = shape[-1]
    rows = prod(shape[:-1])
    widths = [i for i in (8, 4, 2, 1) if w % i == 0 and (vectorize or i == 1)]
    extra = "auto* a=(E*)p[2]; addr|=uintptr_t(a);" if visible else "E* a=nullptr;"
    extra_arg = ",rows" if dynamic_rows else ""
    count = f"(rows*{w})" if dynamic_rows else str(rows * w)
    launch = []
    for width in widths:
        cond = f"if(!(addr & {2 * width - 1})) " if width > 1 else ""
        blocks = (
            f"((rows*{w}/{width}+127)/128)" if dynamic_rows else (rows * w // width + 127) // 128
        )
        launch.append(
            f"{cond}{{packed_swiglu<{width}><<<{blocks},128,0,(cudaStream_t)stream>>>(x,y,a{extra_arg}); return int(cudaGetLastError());}}"
        )
    return (
        f"""#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
using E={CT[dt]};
template<int V> struct alignas(2*V) Packet {{ E lane[V]; }};
template<int V> __global__ void packed_swiglu(const E* __restrict__ x,E* __restrict__ y,E* __restrict__ act{",int rows" if dynamic_rows else ""}) {{
  const int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>={count}/V) return;
  const int row=i/({w}/V),col=(i%({w}/V))*V;
  using P=Packet<V>;const E* base=x+row*{2 * w}+col;
  const P g=*reinterpret_cast<const P*>(base+{gate_half * w});
  const P u=*reinterpret_cast<const P*>(base+{(1 - gate_half) * w}); P z,a;
  #pragma unroll
  for(int j=0;j<V;++j) {{const float f={LOAD[dt]("g.lane[j]")};
    a.lane[j]={STORE[dt]("f/(1.f+expf(-f))")};
    z.lane[j]={STORE[dt](LOAD[dt]("a.lane[j]") + "*" + LOAD[dt]("u.lane[j]"))};
  }}
  reinterpret_cast<P*>(y)[i]=z;
  if constexpr ({str(visible).lower()}) reinterpret_cast<P*>(act)[i]=a;
}}
extern "C" int launch(void** p,int{" rows" if dynamic_rows else ""},void* stream) {{
  auto* x=(const E*)p[0]; auto* y=(E*)p[1];uintptr_t addr=uintptr_t(x)|uintptr_t(y); {extra}
  {chr(10).join(launch)}
}}
""",
        widths[0],
    )


def fuse_packed_swiglu(gm, device, host_backend, vectorize=True):
    records = []
    for out in list(gm.graph.nodes):
        if not out.users:
            continue
        match = candidate(out)
        if match is None:
            continue
        x, split, gate, up, act, shape, half = match
        visible = any(user is not out for user in act.users)
        v = _tensor(x)
        source, width = emit(shape, DT[v.dtype], half, visible, vectorize)
        olds = (out, act) if visible else (out,)
        prog = IndexProgram([tuple(v.shape)], [shape] * len(olds), DT[v.dtype], device, source)
        label = f"_sparkfuse_packed_swiglu_{len(records)}"
        gm.add_module(label, IndexRegion(prog, host_backend))
        with gm.graph.inserting_before(act):
            call = gm.graph.call_module(label, (x,))
            for j, n in enumerate(olds):
                y = gm.graph.call_function(operator.getitem, (call, j))
                y.meta = dict(n.meta)
                n.replace_all_uses_with(y)
        rec = dict(
            kind="packed_swiglu",
            module=label,
            ops=[split.name, gate.name, up.name, act.name, out.name],
            live_ins=[x.name],
            live_outs=[n.name for n in olds],
            shape=shape,
            gate_half=half,
            visible_activation=visible,
            host_backend=host_backend,
            schedule=dict(max_packet_elements=width, threads=128),
            cuda_sha256=sha256(source.encode()).hexdigest(),
        )
        records.append(rec)
        for n in (out, act, gate, up, split):
            if not n.users:
                gm.graph.erase_node(n)
    if records:
        gm.graph.lint()
        gm.recompile()
    return records
