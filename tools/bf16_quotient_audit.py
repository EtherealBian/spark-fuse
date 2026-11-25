"""Pre-admission audit of observable BF16 rounding; NOT an FP32 division replacement."""

import argparse
import ctypes as C
import json
from pathlib import Path
import torch
from sparkfuse.runtime import build
from hashlib import sha256

from provenance import capture
from sparkfuse.score_rounding import GUARDED_QUOTIENT

ROOT = Path(__file__).resolve().parents[1]
src = (
    """#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
"""
    + GUARDED_QUOTIENT
    + """
__global__ void audit(const float* n,const float* d,unsigned short* o,unsigned short* fast,unsigned short* safe,unsigned char* branches,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 float v=n[i], m=d[i], inv=__frcp_rn(m), app=__fmul_rn(v,inv);
 unsigned bits=__float_as_uint(app);
 bool near=((bits & 65535u)-32764u)<=8u;
 bool normal=bits>=0x00800000u && bits<=0x3f800000u;
 bool zero=v==0.f && isfinite(m);
 float q=__fdiv_rn(v,m);
 o[i]=__bfloat16_as_ushort(__float2bfloat16_rn(q));
 fast[i]=__bfloat16_as_ushort(__float2bfloat16_rn(app));
 safe[i]=__bfloat16_as_ushort(__float2bfloat16_rn(bf16_quotient(v,m,inv)));
 branches[i]=zero?0:!normal?1:near?2:3;
}
extern "C" int launch(void **ptr,int count,void *stream){audit<<<(count+255)/256,256,0,(cudaStream_t)stream>>>((float*)ptr[0],(float*)ptr[1],(unsigned short*)ptr[2],(unsigned short*)ptr[3],(unsigned short*)ptr[4],(unsigned char*)ptr[5],count);return (int)cudaGetLastError();}"""
)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--emit", type=Path)
    args = p.parse_args()
    lib, _ = build(src, torch.device("cuda"))
    result = {}
    gen = torch.Generator(device="cuda").manual_seed(82913)

    def check(label, n, d):
        n = n.contiguous().flatten()
        d = d.expand(n.shape).contiguous()
        outs = [torch.empty_like(n, dtype=torch.int16) for _ in range(3)]
        branch = torch.empty_like(n, dtype=torch.uint8)
        a = [n, d, *outs, branch]
        ptrs = (C.c_void_p * len(a))(*(x.data_ptr() for x in a))
        assert lib.launch(ptrs, n.numel(), torch.cuda.current_stream().cuda_stream) == 0
        ref, fast, guard = outs
        eager = (n / d).bfloat16().view(torch.int16)
        different = ref != fast
        fail = ref != guard
        rec = result.setdefault(
            label,
            dict(
                elements=0,
                eager_reference_differences=0,
                unguarded_differences=0,
                guarded_differences=0,
                branches=[0] * 4,
                witness=[],
            ),
        )
        rec["elements"] += n.numel()
        rec["eager_reference_differences"] += int((ref != eager).sum())
        rec["unguarded_differences"] += int(different.sum())
        rec["guarded_differences"] += int(fail.sum())
        counts = torch.bincount(branch.int(), minlength=4).tolist()
        rec["branches"] = [x + y for x, y in zip(rec["branches"], counts, strict=True)]
        nex = max(0, 4 - len(rec["witness"]))
        if nex:
            for (i,) in different.nonzero()[:nex].tolist():
                rec["witness"].append(
                    dict(
                        n=f"{int(n.view(torch.int32)[i]) & 0xFFFFFFFF:08x}",
                        den=f"{int(d.view(torch.int32)[i]) & 0xFFFFFFFF:08x}",
                        exact=f"{int(ref[i]) & 65535:04x}",
                        mul=f"{int(fast[i]) & 65535:04x}",
                        guard=f"{int(guard[i]) & 65535:04x}",
                        branch=int(branch[i]),
                    )
                )

    # All positive BF16 intervals up to one, including subnormals; generate
    # nearby representable numerators at delta -4..4 FP32 ULP. Vary the divisor.
    mid = (
        ((torch.arange(0x3F80, device="cuda", dtype=torch.int32) << 16) + 0x8000)
        .view(torch.float32)
        .double()
    )
    delta = torch.arange(-4, 5, device="cuda", dtype=torch.int64)
    den0 = torch.tensor(
        [1.0, 1.0 + 2.0**-23, 1.5, 2.0, 3.0, 3.125, 7.0, 32.0, 33.0, 257.0, 1024.0, 2048.0],
        device="cuda",
    )
    for div in torch.cat((den0, torch.exp2(11 * torch.rand(160, device="cuda", generator=gen)))):
        center = (mid * div.double()).float().view(torch.int32).long()
        bits = center[:, None] + delta
        bits = bits[(bits >= 0) & (bits <= 0x3F800000)].int()
        n = bits.view(torch.float32)
        check("bf16_midpoints", n, div)
    for _ in range(48):
        n = torch.randint(
            0, 0x3F800001, (131072,), device="cuda", dtype=torch.int32, generator=gen
        ).view(torch.float32)
        d = torch.exp2(11 * torch.rand(n.shape, device="cuda", generator=gen))
        check("random_positive_bits", n, d)
    z = torch.tensor([0.0, -0.0, 2.0**-149, 2.0**-126, 1.0, float("nan")], device="cuda").repeat(3)
    d = torch.tensor([1.0, 2048.0, float("nan")], device="cuda").repeat_interleave(6)
    check("special", z, d)
    doc = dict(
        scope="The exact production helper on nonnegative Softmax terms and positive row sums; compares BF16 encodings, not FP32 quotients. Midpoint-centered stress samples are not a performance distribution. Counts include repeated boundary values.",
        seed=82913,
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        source_sha256=sha256(src.encode()).hexdigest(),
        provenance=capture(ROOT / ".runtime/round-guard/sources"),
        cases=result,
    )
    doc["passed"] = (
        all(
            not (v["guarded_differences"] or v["eager_reference_differences"])
            for v in result.values()
        )
        and result["bf16_midpoints"]["unguarded_differences"] > 0
    )
    if args.emit:
        args.emit.write_text(src)
    tmp = args.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    tmp.replace(args.output)
    return 0 if doc["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
