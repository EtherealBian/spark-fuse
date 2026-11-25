"""One CTA per reduced row. Only lane zero consumes the final sum; no CTA broadcast."""

from .codegen import CT, LOAD, STORE
from .vector import HEADER, TAIL_HEADER


def generate_compact_mean(p, s, emit, index):
    h, v, t = p.width, s.vec, s.threads
    vals = p.vals
    params = [f"const {CT[vals[i].dt]}* __restrict__ in{k}" for k, i in enumerate(p.inputs)] + [
        f"{CT[vals[i].dt]}* __restrict__ out{k}" for k, i in enumerate(p.outputs)
    ]
    operand = vals[p.red].deps[0]
    active = set()

    def mark(i):
        if i not in active:
            active.add(i)
            for d in vals[i].deps:
                mark(d)

    mark(operand)
    packable = [i for i in p.inputs if i in active and index(vals[i]) in ("pos", "j")]
    src = [
        HEADER,
        TAIL_HEADER,
        "template<bool Aligned> __global__ void fused_compact("
        + ", ".join(params + ["int rows"])
        + ") {",
        f"const int row=blockIdx.x, lane=threadIdx.x; if(row>=rows)return; float sums[{v}]={{}};",
        f"for(unsigned ix=threadIdx.x*{v}u;ix<{h}u;ix+={t * v}u) {{const int first=(int)ix;",
    ]
    known = {}
    guards = []
    for i in packable:
        n = vals[i]
        dt = CT[n.dt]
        ptr = f"in{n.imm}"
        offset = f"row*{h}+first" if index(n) == "pos" else "first"
        addr = f"({ptr}+{offset})"
        pack = f"Pack<{dt},{v}>"
        guards.append(f"((uintptr_t({ptr}) & (alignof({pack})-1))==0)")
        # Only select this specialization if every vector operand is aligned for every row.
        src.extend(
            [
                f"{pack} pk{i};",
                f"if constexpr (Aligned) pk{i}=*reinterpret_cast<const {pack}*>({addr});",
                f"else pk{i}=load_tail<{dt},{v}>({addr},{h}-first);",
            ]
        )
        known[i] = LOAD[n.dt](f"pk{i}.x[q]")
    text, names = emit([operand], known)
    src += [
        "#pragma unroll",
        f"for(int q=0;q<{v};q++){{const int j=first+q; if(j>={h})continue; const int pos=row*{h}+j;",
        text,
        f"sums[q]+={names[0]};",
        "}}",
        "float acc=sums[0];",
        "#pragma unroll",
        f"for(int q=1;q<{v};q++)acc+=sums[q];",
        "for(int d=16;d;d>>=1)acc+=__shfl_down_sync(0xffffffff,acc,d);",
    ]
    if t > 32:
        src += [
            f"__shared__ float part[{t // 32}];",
            "if((lane&31)==0)part[lane/32]=acc; __syncthreads(); if(lane>=32)return;",
            f"acc=lane<{t // 32}?part[lane]:0.f;",
            f"for(int d={t // 64};d;d>>=1)acc+=__shfl_down_sync(0xffffffff,acc,d);",
        ]
    typ = vals[p.red].dt
    src += [
        "if(lane!=0)return;",
        f"const int j=0, pos=row*{h};",
        "const float reduced=" + LOAD[typ](STORE[typ](f"acc*(1.f/{h}.f)")) + ";",
    ]
    text, results = emit(p.outputs, {p.red: "reduced"})
    src += [text] + [
        f"out{k}[row]=" + STORE[vals[i].dt](nm) + ";"
        for k, (i, nm) in enumerate(zip(p.outputs, results))
    ]
    src += ["}", 'extern "C" int launch(void**ptr,int rows,void*stream){']
    args = []
    for k, i in enumerate(p.inputs + p.outputs):
        var = f"in{k}" if k < len(p.inputs) else f"out{k - len(p.inputs)}"
        src.append(f"auto {var}=static_cast<{CT[vals[i].dt]}*>(ptr[{k}]);")
        args.append(var)

    def launch(aligned):
        return (
            f"fused_compact<{aligned}><<<rows,{t},0,(cudaStream_t)stream>>>("
            + ",".join(args + ["rows"])
            + ");"
        )

    if h % v == 0 and guards and v > 1:
        src += ["if(" + " && ".join(guards) + ") " + launch("true") + " else " + launch("false")]
    else:
        src += [launch("false")]
    return "\n".join(src + ["return (int)cudaGetLastError();", "}"]) + "\n"
