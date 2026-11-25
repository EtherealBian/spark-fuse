"""Shared row scheduling for stable Softmax and centered LayerNorm; pointwise DAGs on both sides."""

from .codegen import CT, LOAD, STORE


def generate_rowwise(p, s, emit, index):
    h = p.width
    t = s.threads
    online = s.layout == "block_welford"
    xor = s.layout == "warp_xor"
    warp = xor or s.layout == "warp"
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
    v = s.vec
    tail = h % v != 0
    if v > 1:
        from .vector import HEADER, TAIL_HEADER

        src[:4] = [HEADER + (TAIL_HEADER if tail else "")]
    if online:
        from .layernorm import WELFORD_SRC

        src.insert(
            next(i for i, line in enumerate(src) if line.startswith("__global__")), WELFORD_SRC
        )
    off = f"j/{stride}" if v == 1 else f"(first/{stride * v})*{v}+q"
    held_count = ((h + stride * v - 1) // (stride * v)) * v
    if s.cache:
        src.append(f"float held[{held_count}];")
    # Keep a wide cached score row in a real strip-mined loop, too. The GB10
    # overnight search recorded two address faults and one failed launch in these layouts.
    bounded_loop = p.vals[p.red].op == "layer_norm" or (s.cache and held_count >= 32)

    def phase(targets, pre, body, full=()):
        # Gather only input packs used by this phase. Broadcast-only dimensions
        # retain scalar loads, while an unexpanded key dimension permits vectors.
        if v == 1:
            return (["#pragma unroll 1"] if bounded_loop else []) + [
                f"for(int j=lane;j<{h};j+={stride}){{const int pos=row*{h}+j;",
                *body(dict(pre)),
                "}",
            ]
        live = set()

        def walk(i):
            if i in pre or i in live:
                return
            live.add(i)
            for dep in p.vals[i].deps:
                walk(dep)

        for i in targets:
            walk(i)
        # For centered LayerNorm keep the packet traversal as a real loop. A GB10
        # CUDA 13.0 build of the peeled 768-wide second packet step produced an
        # invalid global address under memcheck. Inner packet lanes remain unrolled.
        lines = (["#pragma unroll 1"] if bounded_loop else []) + [
            f"for(int first=lane*{v};first<{h};first+={stride * v}){{"
        ]
        known = dict(pre)
        for i in p.inputs:
            x = p.vals[i]
            if i not in live or not x.shape or x.shape[-1] != h:
                continue
            base = index(x).replace("pos", f"(row*{h}+first)").replace("j", "first")
            load = "load_tail" if tail else "loadv"
            remain = f",{h}-first" if tail else ""
            lines.append(f"auto pk{i}={load}<{CT[x.dt]},{v}>(in{x.imm}+({base}){remain});")
            known[i] = LOAD[x.dt](f"pk{i}.x[q]")
        for k, i in full:
            lines.append(f"Pack<{CT[p.vals[i].dt]},{v}> dst{k}{{}};")
        lines += [
            "#pragma unroll",
            f"for(int q=0;q<{v};++q){{const int j=first+q;",
            f"if(j>={h})continue;" if tail else "",
            f"const int pos=row*{h}+j;",
            *body(known),
            "}",
        ]
        for k, i in full:
            store = "store_tail" if tail else "storev"
            remain = f",{h}-first" if tail else ""
            lines.append(f"{store}(out{k}+row*{h}+first,dst{k}{remain});")
        return lines + ["}"]

    def collective(acc, mode, name):
        neutral = "-INFINITY" if mode == "max" else "0.f"

        def op(x, y):
            return f"fmaxf({x},{y})" if mode == "max" else f"{x}+{y}"

        shuffle = (
            f"for(int d=16;d;d>>=1) {acc}=" + op(acc, f"__shfl_down_sync(0xffffffff,{acc},d)") + ";"
        )
        if xor:
            # Keep each lane's butterfly association and its own result. Broadcasting
            # lane zero would impose a different floating-point evaluation order.
            shuffle = (
                f"for(int d=16;d;d>>=1) {acc}="
                + op(acc, f"__shfl_xor_sync(0xffffffff,{acc},d)")
                + ";"
            )
            return [shuffle, f"const float {name}={acc};"]
        # A 32-thread block is already a complete row warp: no shared scratch
        # or second-stage read/write is needed for a score reduction.
        if warp or (t == 32 and not norm):
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

    def maxbody(known):
        text, (operand,) = emit([inp], known)
        return [text, f"acc=fmaxf(acc,{operand});"] + (
            [f"held[{off}]={operand};"] if s.cache else []
        )

    def sumbody(known):
        if s.cache:
            return [f"float e=expf(held[{off}]-rowmax); held[{off}]=e; mass+=e;"]
        text, (operand,) = emit([inp], known)
        return [text, f"mass+=expf({operand}-rowmax);"]

    norm = p.vals[p.red].op == "layer_norm"
    if online:
        from .layernorm import online_statistics

        src += online_statistics(h, inp, p.vals[p.red].imm, s.cache, off, emit, phase)
    elif norm:
        from .layernorm import statistics

        src += statistics(h, inp, p.vals[p.red].imm, s.cache, off, emit, phase, collective)
    else:
        src += [
            "float acc=-INFINITY;",
            *phase([inp], {}, maxbody),
            *collective("acc", "max", "rowmax"),
            "float mass=0.f;",
            *phase([] if s.cache else [inp], {}, sumbody),
            *collective("mass", "sum", "rowsum"),
            "const float inv=1.f/rowsum;",
        ]
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
        ids = [i for _, i in full]

        def epilogue(known):
            out = []
            if s.cache:
                prob = f"held[{off}]*inv"
            else:
                text, (operand,) = emit([inp], known)
                out += ["float logit; {", text, f"logit={operand}; }}"]
                known[inp] = "logit"
                prob = "((logit-origin)-center)*inv" if norm else "expf(logit-rowmax)*inv"
            if online:
                sample = f"held[{off}]" if s.cache else "logit"
                prob = f"({sample}-center)*inv"
            if xor:
                numerator = f"held[{off}]" if s.cache else "expf(logit-rowmax)"
                prob = f"__fdiv_rn({numerator},rowsum)"
            dt = p.vals[p.red].dt
            out += ["const float prob=" + LOAD[dt](STORE[dt](prob)) + ";"]
            known[p.red] = "prob"
            text, names = emit(ids, known)
            out.append(text)
            for (k, i), nm in zip(full, names):
                dst = f"out{k}[pos]" if v == 1 else f"dst{k}.x[q]"
                out.append(dst + "=" + STORE[p.vals[i].dt](nm) + ";")
            return out

        src += phase(ids + ([] if s.cache else [inp]), {p.red: "prob"}, epilogue, full)
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


# Compatible with workers that loaded codegen just before a source update.
generate_softmax = generate_rowwise
