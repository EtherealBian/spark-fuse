"""Schedule-independent scalar DAG emission and CUDA row/warp scheduling."""

from dataclasses import dataclass
from .ir import Unsupported

CT = {"f32": "float", "f16": "__half", "bf16": "__nv_bfloat16"}
LOAD = {
    "f32": lambda a: a,
    "f16": lambda a: f"__half2float({a})",
    "bf16": lambda a: f"__bfloat162float({a})",
}
STORE = {
    "f32": lambda a: a,
    "f16": lambda a: f"__float2half_rn({a})",
    "bf16": lambda a: f"__float2bfloat16_rn({a})",
}


@dataclass(frozen=True)
class Schedule:
    threads: int = 128
    layout: str = "block"  # block | warp (multiple independent rows) | flat (pointwise only)
    cache: bool = False  # save reduction operand's input expression across phases
    vec: int = 1

    def validate(self, p):
        if self.vec not in (1, 2, 4, 8) or any(
            self.vec * (4 if p.vals[i].dt == "f32" else 2) > 16 for i in p.inputs + p.outputs
        ):
            raise Unsupported("illegal vector pack for width/dtype")
        if self.threads not in (32, 64, 128, 256, 512, 1024) or self.layout not in (
            "block",
            "warp",
            "flat",
        ):
            raise Unsupported("unknown schedule")
        if self.layout == "flat" and (
            p.red is not None or any(p.vals[i].shape != p.domain for i in p.outputs)
        ):
            raise Unsupported("flat requires full-sized pointwise outputs")
        if self.cache and (p.red is None or self.layout == "flat"):
            raise Unsupported("cache requires a reduction")


def generate(p, s):
    s.validate(p)
    h = p.width
    t = s.threads
    warp = s.layout == "warp"
    stride = 32 if warp else t

    def index(v):
        if v.shape == p.domain:
            return "pos"
        if v.shape == (h,):
            return "j"
        if v.shape == ():
            return "0"
        return "row"

    def emit(targets, pre=None):
        known = dict(pre or {})
        lines = []

        def rec(i):
            if i in known:
                return known[i]
            v = p.vals[i]
            args = [rec(k) for k in v.deps]
            a = args[0] if args else ""
            b = args[1] if len(args) > 1 else ""
            if v.op == "const":
                expr = float(v.imm).hex() + "f"
            elif v.op == "input":
                expr = LOAD[v.dt](f"in{v.imm}[{index(v)}]")
            elif v.op == "mean":
                raise AssertionError("unbound reduction")
            else:
                expr = {
                    "add": f"{a}+{b}",
                    "sub": f"{a}-{b}",
                    "mul": f"{a}*{b}",
                    "div": f"{a}/{b}",
                    "square": f"{a}*{a}",
                    "rsqrt": f"rsqrtf({a})",
                    "relu": f"(isnan({a}) ? {a} : fmaxf({a},0.f))",
                    "silu": f"{a}/(1.f+expf(-{a}))",
                    "sigmoid": f"1.f/(1.f+expf(-{a}))",
                    "neg": f"-{a}",
                    "tanh": f"tanhf({a})",
                    "cast": a,
                }[v.op]
                expr = LOAD[v.dt](STORE[v.dt](expr))
            name = f"v{i}"
            lines.append(f"const float {name} = {expr};")
            known[i] = name
            return name

        names = [rec(i) for i in targets]
        return "\n".join(lines), names

    if s.vec > 1:
        from .vector import generate_vector

        return generate_vector(p, s, emit, index)
    src = [
        "#include <cuda_runtime.h>",
        "#include <cuda_fp16.h>",
        "#include <cuda_bf16.h>",
        "#include <math.h>",
    ]
    params = [f"const {CT[p.vals[i].dt]}* __restrict__ in{k}" for k, i in enumerate(p.inputs)] + [
        f"{CT[p.vals[i].dt]}* __restrict__ out{k}" for k, i in enumerate(p.outputs)
    ]
    src += ["__global__ void fused(" + ", ".join(params + ["int rows"]) + ") {"]
    rowid = "blockIdx.x * (blockDim.x/32) + threadIdx.x/32" if warp else "blockIdx.x"
    lane = "(threadIdx.x & 31)" if warp else "threadIdx.x"
    flat = s.layout == "flat"
    full = []
    rowout = []
    for k, i in enumerate(p.outputs):
        (full if p.vals[i].shape == p.domain else rowout).append((k, i))
    pre = {}
    if flat:
        src += [
            f"const int pos=blockIdx.x*blockDim.x+threadIdx.x; if(pos>=rows*{h})return; const int j=pos%{h}, row=pos/{h};"
        ]
    else:
        src += [f"const int row={rowid}; if(row>=rows)return; const int lane={lane};"]
    loop = f"for(int j=lane;j<{h};j+={stride}) {{ const int pos=row*{h}+j;"
    if p.red is not None:
        operand = p.vals[p.red].deps[0]
        ov = p.vals[operand]
        saved = ov.deps[0] if ov.op == "square" else operand
        if s.cache:
            src.append(f"float held[{(h + stride - 1) // stride}];")
        src += ["float acc=0.f;", loop]
        text, names = emit([operand, saved] if s.cache else [operand])
        src += [text, "acc += " + names[0] + ";"]
        if s.cache:
            src.append(f"held[j/{stride}]={names[1]};")
            pre[saved] = f"held[j/{stride}]"
        src += ["}", "for(int d=16;d;d>>=1) acc += __shfl_down_sync(0xffffffff,acc,d);"]
        if warp:
            src += ["float sum=__shfl_sync(0xffffffff,acc,0);"]
        else:
            src += [
                f"__shared__ float parts[{t // 32}];",
                "if((threadIdx.x&31)==0)parts[threadIdx.x/32]=acc; __syncthreads();",
                f"if(threadIdx.x<32){{ acc=threadIdx.x<{t // 32}?parts[threadIdx.x]:0.f; for(int d=16;d;d>>=1)acc+=__shfl_down_sync(0xffffffff,acc,d); if(threadIdx.x==0)parts[0]=acc; }} __syncthreads(); float sum=parts[0];",
            ]
        v = p.vals[p.red]
        src += ["const float reduced=" + LOAD[v.dt](STORE[v.dt](f"sum*(1.f/{h}.f)")) + ";"]
        pre[p.red] = "reduced"

    def outputs(pairs, prefix):
        text, names = emit([i for _, i in pairs], prefix)
        out = [text]
        for (k, i), name in zip(pairs, names):
            out.append(f"out{k}[{index(p.vals[i])}] = " + STORE[p.vals[i].dt](name) + ";")
        return out

    if rowout:
        src += (
            ["if(lane==0){ const int j=0, pos=row*" + str(h) + ";"]
            + outputs(rowout, {p.red: "reduced"} if p.red is not None else {})
            + ["}"]
        )
    if full:
        src += ([] if flat else [loop]) + outputs(full, pre) + ([] if flat else ["}"])
    src += ["}", 'extern "C" int launch(void **ptr, int rows, void *stream) {']
    args = [f"({CT[p.vals[i].dt]}*)ptr[{k}]" for k, i in enumerate(p.inputs + p.outputs)]
    blocks = f"(rows*{h}-1)/{t}+1" if flat else (f"(rows-1)/{t // 32}+1" if warp else "rows")
    src += [
        f"fused<<<{blocks},{t},0,(cudaStream_t)stream>>>("
        + ", ".join(args + ["rows"])
        + ");return (int)cudaGetLastError();",
        "}",
    ]
    return "\n".join(src) + "\n"
