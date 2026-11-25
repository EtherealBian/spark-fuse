"""Fixed-order FP32 mean of a materialized row expression, then a fused epilogue.

Logical addition order and physical launch geometry are distinct: a small CTA can
visit multiple virtual warps without changing the sum. Input views need not be aligned.
"""

from .codegen import CT, LOAD, STORE
from .vector import HEADER


def eligible(p):
    if p.red is None:
        return False
    red = p.vals[p.red]
    if red.op != "mean" or red.imm != "exported_mean" or red.dt != "f32":
        return False
    src = p.vals[red.deps[0]]
    # The eager *temporary* being reduced starts aligned, even when its source views don't.
    # Do not assume this for a placeholder or an aliasing no-copy cast.
    return (
        src.dt == "f32"
        and src.op in ("square", "mul", "add", "sub")
        and p.rows >= 16
        and 128 <= p.width <= 65536
        and p.width % 4 == 0
        and p.rows * p.width <= 2**24
    )


def sensitive(p):
    if not eligible(p):
        return False
    after, rounded = {p.red}, set()
    for i, v in enumerate(p.vals):
        if not after.intersection(v.deps):
            continue
        after.add(i)
        if (
            v.op == "cast"
            and v.dt in ("f16", "bf16")
            and p.vals[v.deps[0]].dt == "f32"
            and v.shape == p.domain
        ):
            rounded.add(i)
        elif (
            rounded.intersection(v.deps)
            and v.op in ("mul", "add", "sub", "div")
            and v.dt in ("f16", "bf16")
        ):
            return True
    return False


def generate_tree(p, s, emit, index):
    h, t, vals, red = p.width, s.threads, p.vals, p.red
    warps = t // 32
    wide = (h + 31) // 32 >= 256
    logical_threads = 512 if wide else 32
    steps = (h + 4 * logical_threads - 1) // (4 * logical_threads)
    slots = steps * (16 // warps if wide else 1) * 4
    operand = vals[red].deps[0]
    saved = vals[operand].deps[0] if vals[operand].op == "square" else operand
    names = [f"in{k}" for k in range(len(p.inputs))] + [f"out{k}" for k in range(len(p.outputs))]
    ids = p.inputs + p.outputs
    params = [
        f"{'const ' if k < len(p.inputs) else ''}{CT[vals[i].dt]}* __restrict__ {name}"
        for k, (name, i) in enumerate(zip(names, ids))
    ]
    src = [
        HEADER,
        "__global__ void fused_mean_tree(" + ", ".join(params + ["int rows"]) + ") {",
        "const int lane=threadIdx.x&31, warp=threadIdx.x/32;",
        "const int row=blockIdx.x;" if wide else f"const int row=blockIdx.x*{warps}+warp;",
        "if(row>=rows)return;",
    ]
    if wide:
        src += ["__shared__ float partial[16], total;"]
    if s.cache:
        src += [f"float held[{slots}];"]
    per_warp = f"for(int vw=warp;vw<16;vw+={warps}){{" if wide else "{ const int vw=0;"
    packet = f"for(int first=(32*vw+lane)*4;first<{h};first+={logical_threads * 4}){{"
    each = f"#pragma unroll\nfor(int q=0;q<4;q++){{ const int j=first+q, pos=row*{h}+j;"
    slot = f"((vw/{warps})*{steps}+first/{logical_threads * 4})*4+q" if wide else "first/128*4+q"

    def loads(targets, pre):
        seen = set()

        def walk(i):
            if i in seen or i in pre:
                return
            seen.add(i)
            for j in vals[i].deps:
                walk(j)

        for i in targets:
            walk(i)
        lines = []
        known = dict(pre)
        for k, i in enumerate(p.inputs):
            dtype = vals[i].dt
            ix = index(vals[i])
            if i in seen and ix in ("pos", "j"):
                off = f"row*{h}+first" if ix == "pos" else "first"
                lines.append(f"auto pack{i}=loadv<{CT[dtype]},4>(in{k}+{off});")
                known[i] = LOAD[dtype](f"pack{i}.x[q]")
        return lines, known

    targets = [operand, saved] if s.cache else [operand]
    ld, known = loads(targets, {})
    ex, out = emit(targets, known)
    src += [
        "float sum=0.f;",
        per_warp,
        "float accum[4]={};",
        packet,
        *ld,
        each,
        ex,
        f"accum[q]=__fadd_rn(accum[q],{out[0]});",
    ]
    if s.cache:
        src += [f"held[{slot}]={out[1]};"]
    src += [
        "}}",
        "float v=__fadd_rn(__fadd_rn(__fadd_rn(accum[0],accum[1]),accum[2]),accum[3]);",
        "for(int d=16;d;d>>=1)v=__fadd_rn(v,__shfl_down_sync(0xffffffff,v,d));",
    ]
    if wide:
        src += [
            "if(!lane)partial[vw]=v; } __syncthreads();",
            "if(warp==0){float v=lane<16?partial[lane]:0.f; for(int d=8;d;d>>=1)v=__fadd_rn(v,__shfl_down_sync(0xffffffff,v,d));if(!lane)total=v;} __syncthreads(); sum=total;",
        ]
    else:
        src += ["sum=__shfl_sync(0xffffffff,v,0); }"]
    src += [f"const float reduced=__fmul_rn(sum,1.f/{h}.f);"]
    small = [(k, i) for k, i in enumerate(p.outputs) if vals[i].shape != p.domain]
    full = [(k, i) for k, i in enumerate(p.outputs) if vals[i].shape == p.domain]
    if small:
        ex, out = emit([i for _, i in small], {red: "reduced"})
        src += [
            ("if(threadIdx.x==0){" if wide else "if(lane==0){") + f"const int j=0,pos=row*{h};",
            ex,
        ]
        src += [f"out{k}[row]=" + STORE[vals[i].dt](v) + ";" for (k, i), v in zip(small, out)] + [
            "}"
        ]
    if full:
        pre = {red: "reduced"}
        if s.cache:
            pre[saved] = f"held[{slot}]"
        targets = [i for _, i in full]
        ld, known = loads(targets, pre)
        ex, out = emit(targets, known)
        src += (
            [per_warp, packet, *ld]
            + [f"Pack<{CT[vals[i].dt]},4> dst{k};" for k, i in full]
            + [each, ex]
        )
        src += [f"dst{k}.x[q]=" + STORE[vals[i].dt](v) + ";" for (k, i), v in zip(full, out)] + [
            "}"
        ]
        src += [f"storev(out{k}+row*{h}+first,dst{k});" for k, _ in full] + ["}}"]
    blocks = "rows" if wide else f"(rows-1)/{warps}+1"
    args = [f"({CT[vals[i].dt]}*)ptr[{k}]" for k, i in enumerate(ids)] + ["rows"]
    src += [
        "}",
        'extern "C" int launch(void**ptr,int rows,void*stream){',
        f"fused_mean_tree<<<{blocks},{t},0,(cudaStream_t)stream>>>("
        + ",".join(args)
        + "); return (int)cudaGetLastError();}",
    ]
    return "\n".join(src) + "\n"
