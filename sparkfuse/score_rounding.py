"""Dividing nonnegative Softmax terms when only a BF16-rounded result can escape."""


def only_bf16_probability(p):
    if p.red is None or p.red in p.outputs:
        return False
    red = p.vals[p.red]
    if red.op != "softmax" or red.dt != "f32":
        return False
    consumers = [v for v in p.vals if p.red in v.deps]
    return bool(consumers) and all(
        v.op == "cast" and v.dt == "bf16" and v.deps == (p.red,) for v in consumers
    )


# Numerator ∈ [0,1], denominator ∈ [1,width] on finite Softmax rows. Each of
# RN(1/d) and RN(n*inv) contributes at most one unit roundoff on normal
# results. Allow four neighboring FP32 codes on either side of a BF16
# midpoint (also covers crossing an FP32 binade). There, or for subnormal
# products / nonfinite rows, use the original RN division. Do NOT expose the
# interior approximate FP32 value, even if another use also rounds to BF16.
GUARDED_QUOTIENT = """__device__ __forceinline__ float bf16_quotient(float n,float d,float inv){
    const float approx=__fmul_rn(n,inv);
    const unsigned bits=__float_as_uint(approx);
    const bool near=((bits & 0xffffu)-0x7ffcu)<=8u;
    const bool normal=bits>=0x00800000u && bits<=0x3f800000u;
    return ((normal && !near) || (n==0.f && isfinite(d))) ? approx : __fdiv_rn(n,d);
}"""
