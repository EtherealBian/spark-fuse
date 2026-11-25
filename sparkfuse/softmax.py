"""Stable row softmax micro-schedule; arbitrary supported pointwise DAGs on both sides."""

from .codegen import CT, LOAD, STORE


def generate_softmax(p, s, emit, index):
    h = p.width
    t = s.threads
    warp = s.layout == "warp"
    stride = 32 if warp else t
    inp = p.vals[p.red].deps[0]
    params = [f"const {CT[p.vals[i].dt]}* __restrict__ in{k}" for k, i in enumerate(p.inputs)] + [
        f"{CT[p.vals[i].dt]}* __restrict__ out{k}" for k, i in enumerate(p.outputs)
    ]
    args = [f"({CT[p.vals[i].dt]}*)ptr[{k}]" for k, i in enumerate(p.inputs + p.outputs)]
    src = [
        "#include <cuda_runtime.h>",
        "#include <cuda_fp16.h>",
        "#include <cuda_bf16.h>",
        "#include <math.h>",
        "__global__ void fused(" + ", ".join(params + ["int rows"]) + ") {",
    ]
    row = f"blockIdx.x*{t // 32}+(threadIdx.x>>5)" if warp else "blockIdx.x"
    lane = "(threadIdx.x&31)" if warp else "threadIdx.x"
    src += [f"const int row={row}; if(row>=rows)return; const int lane={lane};"]
    loop = f"for(int j=lane;j<{h};j+={stride}){{ const int pos=row*{h}+j;"
    text, (operand,) = emit([inp])
    if s.cache:
        src.append(f"float held[{(h + stride - 1) // stride}];")

    def collective(acc, mode, name):
        neutral = "-INFINITY" if mode == "max" else "0.f"

        def op(x, y):
            return f"fmaxf({x},{y})" if mode == "max" else f"{x}+{y}"

        shuffle = (
            f"for(int d=16;d;d>>=1) {acc}=" + op(acc, f"__shfl_down_sync(0xffffffff,{acc},d)") + ";"
        )
        if warp:
            return [shuffle, f"float {name}=__shfl_sync(0xffffffff,{acc},0);"]
        # Distinct scratch arrays for the two phases: no scratch overwrite race.
        sh = "part_" + name
        return [
            shuffle,
            f"__shared__ float {sh}[{t // 32}]; if((lane&31)==0){sh}[lane>>5]={acc}; __syncthreads();",
            f"if(lane<32){{ {acc}=lane<{t // 32}?{sh}[lane]:{neutral};",
            shuffle,
            f"if(lane==0){sh}[0]={acc}; }} __syncthreads(); const float {name}={sh}[0];",
        ]

    src += ["float acc=-INFINITY;", loop, text, f"acc=fmaxf(acc,{operand});"]
    if s.cache:
        src += [f"held[j/{stride}]={operand};"]
    src += ["}", *collective("acc", "max", "rowmax"), "float mass=0.f;", loop]
    if s.cache:
        src += [f"float e=expf(held[j/{stride}]-rowmax); held[j/{stride}]=e;"]
    else:
        src += [text, f"float e=expf({operand}-rowmax);"]
    src += ["mass+=e;}", *collective("mass", "sum", "rowsum"), "const float inv=1.f/rowsum;"]
    full = [(k, i) for k, i in enumerate(p.outputs) if p.vals[i].shape == p.domain]
    small = [(k, i) for k, i in enumerate(p.outputs) if p.vals[i].shape != p.domain]
    if small:
        src += [f"if(lane==0){{ const int j=0,pos=row*{h};"]
        tx, names = emit([i for _, i in small])
        src += [tx]
        for (k, i), nm in zip(small, names):
            src += [f"out{k}[row]=" + STORE[p.vals[i].dt](nm) + ";"]
        src += ["}"]
    if full:
        src.append(loop)
        known = {}
        if s.cache:
            prob = f"held[j/{stride}]*inv"
        else:
            # Scope temporaries from the operand DAG separately from the epilogue.
            src += ["float logit; {", text, f"logit={operand}; }}"]
            known[inp] = "logit"
            prob = "expf(logit-rowmax)*inv"
        dt = p.vals[p.red].dt
        src += ["const float prob=" + LOAD[dt](STORE[dt](prob)) + ";"]
        known[p.red] = "prob"
        tx, names = emit([i for _, i in full], known)
        src += [tx]
        for (k, i), nm in zip(full, names):
            src += [f"out{k}[pos]=" + STORE[p.vals[i].dt](nm) + ";"]
        src += ["}"]
    blocks = f"(rows-1)/{t // 32}+1" if warp else "rows"
    src += [
        "}",
        'extern "C" int launch(void **ptr,int rows,void *stream){',
        f"fused<<<{blocks},{t},0,(cudaStream_t)stream>>>("
        + ",".join(args + ["rows"])
        + ");return (int)cudaGetLastError();",
        "}",
    ]
    return "\n".join(src) + "\n"
