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
