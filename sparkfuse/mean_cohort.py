"""Narrow fixed-order FP32 mean: one lane per independent scalar recurrence, one CTA per row."""

from .codegen import CT, STORE
from .vector import HEADER


def generate_cohort(p, s, emit):
    h, t, vals, red = p.width, s.threads, p.vals, p.red
    operand = vals[red].deps[0]
    saved = vals[operand].deps[0] if vals[operand].op == "square" else operand
    ids = p.inputs + p.outputs
    names = [f"in{k}" for k in range(len(p.inputs))] + [f"out{k}" for k in range(len(p.outputs))]
    params = [
        f"{'const ' if k < len(p.inputs) else ''}{CT[vals[i].dt]}* __restrict__ {name}"
        for k, (name, i) in enumerate(zip(names, ids))
    ]
    src = [
        HEADER,
        "__global__ void fused_mean_cohort(" + ", ".join(params + ["int rows"]) + ") {",
        "const int row=blockIdx.x, tid=threadIdx.x;",
        "__shared__ float lanes[32], total;",
    ]
    if s.cache:
        # Save a prefix, not necessarily the entire reduction operand: the epilogue may reuse it.
        ex, a = emit([saved])
        src += [
            f"__shared__ float tile[{h}];",
            f"for(int j=tid;j<{h};j+={t}){{const int pos=row*{h}+j;",
            ex,
            f"tile[j]={a[0]};}} __syncthreads();",
        ]
    one = {saved: "tile[j]"} if s.cache else {}
    ex, a = emit([operand], one)
    src += [
        "if(tid<128){ float sum=0.f;",
        f"for(int j=tid;j<{h};j+=128){{const int pos=row*{h}+j;",
        ex,
        f"sum=__fadd_rn(sum,{a[0]});}}",
        "float b=__shfl_down_sync(0xffffffff,sum,1), c=__shfl_down_sync(0xffffffff,sum,2), d=__shfl_down_sync(0xffffffff,sum,3);",
        "if((tid&3)==0)lanes[tid/4]=__fadd_rn(__fadd_rn(__fadd_rn(sum,b),c),d); } __syncthreads();",
        "if(tid<32){float v=lanes[tid];for(int d=16;d;d>>=1)v=__fadd_rn(v,__shfl_down_sync(0xffffffff,v,d));",
        f"if(!tid)total=__fmul_rn(v,1.f/{h}.f);}} __syncthreads();",
        "const float reduced=total;",
    ]
    for full in (False, True):
        live = [(k, i) for k, i in enumerate(p.outputs) if (vals[i].shape == p.domain) == full]
        if not live:
            continue
        pre = {red: "reduced"}
        pre.update(one if full else {})
        ex, a = emit([i for k, i in live], pre)
        src += [
            f"for(int j=tid;j<{h};j+={t}){{const int pos=row*{h}+j;"
            if full
            else f"if(!tid){{ const int j=0,pos=row*{h};",
            ex,
        ]
        src += [
            f"out{k}[{'pos' if full else 'row'}]=" + STORE[vals[i].dt](v) + ";"
            for (k, i), v in zip(live, a)
        ] + ["}"]
    args = [f"({CT[vals[i].dt]}*)ptr[{k}]" for k, i in enumerate(ids)] + ["rows"]
    src += [
        "}",
        'extern "C" int launch(void**ptr,int rows,void*stream){',
        f"fused_mean_cohort<<<rows,{t},0,(cudaStream_t)stream>>>("
        + ",".join(args)
        + ");return (int)cudaGetLastError();}",
    ]
    return "\n".join(src) + "\n"
