"""Join the two functional cache appends feeding an already-qualified symbolic attention site."""

import ctypes as C
from hashlib import sha256
import operator
import torch
from torch import fx
from .decode_sites import _signature
from .runtime import build
from .native import _extension


def emit(groups, width):
    # One grid row per batch/head/branch: no division by a runtime context length.
    return f"""#include <cuda_runtime.h>
#include <stdint.h>
template<int W,typename Packet>
__global__ void joined(const unsigned short* k,const unsigned short* kt,const unsigned short* v,const unsigned short* vt,unsigned short* ko,unsigned short* vo,int s,int flags) {{
  constexpr int G={groups},D={width},P=D/W;
  const int i=blockIdx.x*blockDim.x+threadIdx.x, bh=blockIdx.y, branch=blockIdx.z;
  if(i>=(s+1)*P)return;
  int token=i/P,packet=i%P;
  auto old=branch?v:k, fresh=branch?vt:kt;auto out=branch?vo:ko;
  int row=(flags&(1<<branch)) ? ((bh/G*s+token)*G+bh%G) : bh*s+token;
  auto src=token==s ? fresh+bh*D+packet*W : old+row*D+packet*W;
  *reinterpret_cast<Packet*>(out+(bh*(s+1)*P+i)*W)=*reinterpret_cast<const Packet*>(src);
}}
extern "C" int launch(void** p,int encoded,void* stream) {{
  int s=encoded>>2,flags=encoded&3;
  uintptr_t addr=0;for(int i=0;i<6;++i)addr|=uintptr_t(p[i]);
  #define COPY(W,T) joined<W,T><<<dim3(((s+1)*({width}/W)+127)/128,BATCH_HEADS,2),128,0,(cudaStream_t)stream>>>((const unsigned short*)p[0],(const unsigned short*)p[1],(const unsigned short*)p[2],(const unsigned short*)p[3],(unsigned short*)p[4],(unsigned short*)p[5],s,flags)
  if((addr&15)==0){{COPY(8,uint4);}}else if((addr&3)==0){{COPY(2,unsigned int);}}else{{COPY(1,unsigned short);}}
  return int(cudaGetLastError());
}}
"""


class Append(torch.nn.Module):
    def __init__(self, plan):
        super().__init__()
        b, g, d = plan["batch"], plan["kv_heads"], plan["width"]
        self.source = emit(g, d).replace("BATCH_HEADS", str(b * g))
        self.lib, self.path = build(self.source, torch.device(plan["device"]))
        self.run = (
            _extension()
            ._AppendIO(
                self.lib,
                C.cast(self.lib.launch, C.c_void_p).value,
                torch.device(plan["device"]).index,
                plan["lower"] - 1,
                plan["upper"] - 1,
                b,
                g,
                d,
                plan["dtype"] == "torch.bfloat16",
            )
            .run
        )

    def forward(self, *args):
        return self.run(*args)


def _inputs(n):
    if (
        not isinstance(n, fx.Node)
        or n.op != "call_function"
        or n.target is not torch.ops.aten.cat.default
        or n.kwargs
        or len(n.args) != 2
        or n.args[1] not in (2, -2)
    ):
        return None
    xs = n.args[0]
    return (
        tuple(xs)
        if isinstance(xs, (list, tuple))
        and len(xs) == 2
        and all(isinstance(x, fx.Node) for x in xs)
        else None
    )


def fuse_dynamic_appends(gm, sites, traces, *, excluded=()):
    from .hybrid_decode import _bits

    records = []
    for s in sites:
        if s["offset_from_symbol"] != 1:
            continue
        (att,) = [n for n in gm.graph.nodes if n.op == "call_module" and n.target == s["module"]]
        k, v = att.args[1:3]
        ka, va = _inputs(k), _inputs(v)
        if k in excluded or v in excluded or not ka or not va or k is v:
            continue
        args = (*ka, *va)
        pos = {n: i for i, n in enumerate(gm.graph.nodes)}
        anchor = min((*k.users, *v.users), key=pos.get)
        if max(pos[x] for x in args) >= pos[anchor] or k in args or v in args:
            continue
        plan = s["plan"]
        from .symbolic_sites import _shape

        b, g, d = plan["batch"], plan["kv_heads"], plan["width"]
        token = (b, g, 1, d)
        full = _shape(k)
        old = (b, g, full[2] - 1, d)
        if (
            full != _shape(v)
            or any(
                not isinstance(x.meta.get("val"), torch.Tensor)
                or str(x.meta["val"].dtype) != plan["dtype"]
                or str(x.meta["val"].device) != plan["device"]
                for x in args
            )
            or tuple(map(_shape, args)) != (old, token, old, token)
        ):
            continue
        rep = Append(plan)
        rec = dict(ops=[k.name, v.name], live_ins=[n.name for n in args])
        records.append(rec)
        for e in traces:
            xs = tuple(e[n] for n in args)
            ref = e[k], e[v]
            if (
                any(
                    x.ndim != 4
                    or tuple(x.shape[:2]) != (plan["batch"], plan["kv_heads"])
                    or x.shape[3] != plan["width"]
                    for x in xs
                )
                or xs[1].shape[2] != 1
                or xs[3].shape[2] != 1
                or xs[0].shape != xs[2].shape
            ):
                break
            yy = rep(*xs)
            if _signature(yy, xs) != _signature(ref, xs) or not all(
                a._base is None and b._base is None and _bits(a, b)
                for a, b in zip(yy, ref, strict=True)
            ):
                break
        else:
            name = f"_sparkfuse_runtime_append_{len(records) - 1}"
            gm.add_module(name, rep)
            with gm.graph.inserting_before(anchor):
                call = gm.graph.call_module(name, args)
                for j, n in enumerate((k, v)):
                    y = gm.graph.call_function(operator.getitem, (call, j))
                    y.meta = dict(n.meta)
                    n.replace_all_uses_with(y)
            gm.graph.erase_node(k)
            gm.graph.erase_node(v)
            rec.update(
                module=name,
                source_sha256=sha256(rep.source.encode()).hexdigest(),
                past=[plan["lower"] - 1, plan["upper"] - 1],
                layouts=["BHSD", "BSHD"],
                other_strides="aten",
                returns="two fresh caches",
            )
            continue
        rec["fallback"] = "cache layout or bytes changed on qualification"
    gm.graph.lint()
    gm.recompile()
    return records
