"""Variance for LayerNorm with an explicit centering step (not E[x²] - E[x]²)."""


def statistics(width, operand, epsilon, cache, off, emit, phase, collective):
    # The row origin shifts the first sum close to zero for large common offsets.
    # Accumulate biased variance (divide by width), per the ATen LayerNorm contract.
    text, (origin,) = emit([operand])
    src = [f"float origin; {{const int j=0, pos=row*{width};", text, f"origin={origin}; }}"]

    def center(known):
        text, (op,) = emit([operand], known)
        return [text, f"const float shifted={op}-origin; moment+=shifted;"] + (
            [f"held[{off}]=shifted;"] if cache else []
        )

    def variance(known):
        if cache:
            return [
                f"const float centered=held[{off}]-center; variance+=centered*centered; held[{off}]=centered;"
            ]
        text, (op,) = emit([operand], known)
        return [text, f"const float centered=({op}-origin)-center; variance+=centered*centered;"]

    return [
        *src,
        "float moment=0.f;",
        *phase([operand], {}, center),
        *collective("moment", "sum", "sum_shifted"),
        f"const float center=sum_shifted*(1.f/{width}.f); float variance=0.f;",
        *phase([] if cache else [operand], {}, variance),
        *collective("variance", "sum", "sum_square"),
        f"const float inv=rsqrtf(sum_square*(1.f/{width}.f)+{float(epsilon).hex()}f);",
    ]


# Parallel online moments. Merge orientation and the fixed four-warp tree are part
# of this schedule's contract, not a promise about arbitrary ATen versions.
WELFORD_SRC = r"""
struct Moment3 {float mu, ss, n;};
__device__ __forceinline__ Moment3 join_moments(Moment3 lo, Moment3 hi) {
    float n=hi.n+lo.n;
    if(n==0.f)return {0.f,0.f,0.f};
    float inv_n=__frcp_rn(n), lo_share=lo.n*inv_n, hi_share=hi.n*inv_n;
    float d=lo.mu-hi.mu;
    return {fmaf(hi.mu,hi_share,lo.mu*lo_share),
            fmaf((d*d)*hi.n,lo_share,hi.ss+lo.ss),n};
}
"""


def online_statistics(width, operand, epsilon, cache, off, emit, phase):
    def update(known):
        code, (sample,) = emit([operand], known)
        return [
            code,
            f"float diff={sample}-m.mu; m.n+=1.f; m.mu=fmaf(diff,__frcp_rn(m.n),m.mu); m.ss=fmaf(diff,{sample}-m.mu,m.ss);",
        ] + ([f"held[{off}]={sample};"] if cache else [])

    return [
        "Moment3 m={0.f,0.f,0.f};",
        *phase([operand], {}, update),
        "for(int d=16;d;d>>=1){Moment3 other={__shfl_down_sync(0xffffffff,m.mu,d),__shfl_down_sync(0xffffffff,m.ss,d),__shfl_down_sync(0xffffffff,m.n,d)}; m=join_moments(m,other);}",
        "__shared__ Moment3 parts[4]; if((lane&31)==0)parts[lane>>5]=m; __syncthreads();",
        "if(lane==0){Moment3 a=join_moments(parts[0],parts[2]); Moment3 b=join_moments(parts[1],parts[3]); parts[0]=join_moments(a,b);} __syncthreads();",
        f"const float center=parts[0].mu; const float inv=rsqrtf(__fdiv_rn(parts[0].ss,{width}.f)+{float(epsilon).hex()}f);",
    ]
