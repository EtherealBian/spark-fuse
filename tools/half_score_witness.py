"""Does implicit CUDA Half->Float Softmax use the same reduction as a separate cast?"""

import argparse
import json
from pathlib import Path

import torch
from kernel_trace import kernel_census
from ordered_score_panel import EXPORT, ROOT, SEEDS, badbits, inputs
from provenance import capture
from sparkfuse import Schedule, compile_export


class Visible(torch.nn.Module):
    def forward(self, x, mask):
        scaled = x * 0.08838834764831845
        z = scaled + mask
        p = z.float().softmax(-1)
        return scaled, z, p, p.half()


def bit_examples(a, b):
    lhs, rhs = a.view(torch.int32), b.view(torch.int32)
    indices = (lhs != rhs).nonzero()[:4].tolist()
    return [
        dict(
            index=idx,
            implicit_fp32_hex=f"{int(lhs[tuple(idx)]) & 0xFFFFFFFF:08x}",
            explicit_fp32_hex=f"{int(rhs[tuple(idx)]) & 0xFFFFFFFF:08x}",
        )
        for idx in indices
    ]


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    mod = Visible()
    cases = []
    seeds = [*SEEDS, 80021, 80107]
    for w in (1023, 1024, 1025, 1537, 2047, 2048):
        ep = torch.export.export(mod, inputs(24, w, torch.float16, EXPORT, 0))
        k = compile_export(
            ep, schedule=Schedule(128, "warp_slots" if w > 1024 else "warp_xor", True)
        )
        samples = []
        for sd in seeds:
            for off in (0, 1):
                x, mask = inputs(24, w, torch.float16, sd, off)
                scaled, z, p_explicit, h = ref = mod(x, mask)
                # Both reference reductions receive exactly the same masked score values.
                p_implicit = torch.softmax(z, -1, dtype=torch.float32)
                actual = k(x, mask)
                assert actual[2].dtype == p_implicit.dtype == torch.float32
                samples.append(
                    dict(
                        seed=sd,
                        offset_elements=[off, 3 * off],
                        fused_vs_explicit_bits=[
                            badbits(v, r) for v, r in zip(actual, ref, strict=True)
                        ],
                        fused_vs_implicit_fp32_bits=badbits(actual[2], p_implicit),
                        implicit_vs_explicit_fp32_bits=badbits(p_implicit, p_explicit),
                        implicit_vs_explicit_half_bits=badbits(p_implicit.half(), h),
                        examples=bit_examples(p_implicit, p_explicit),
                    )
                )
        f = z.float()
        cases.append(
            dict(
                width=w,
                samples=samples,
                profiles={
                    "implicit_half_to_float": kernel_census(
                        lambda: z.softmax(-1, dtype=torch.float32)
                    ),
                    "on_explicit_float_input": kernel_census(lambda: f.softmax(-1)),
                },
            )
        )
    doc = dict(
        scope="Identical materialized z, same row layout. Fused explicit graph returns scaled FP16, masked FP16, FP32 probabilities, rounded FP16. Counts are elements with unequal bit patterns.",
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        torch_git=torch.version.git_version,
        seeds=seeds,
        export_seed=EXPORT,
        provenance=capture(ROOT / ".runtime/wide-xor-lab/sources"),
        cases=cases,
    )
    doc["passed"] = all(
        not any(s["fused_vs_explicit_bits"])
        and s["fused_vs_implicit_fp32_bits"] == s["implicit_vs_explicit_fp32_bits"]
        and (s["implicit_vs_explicit_fp32_bits"] > 0) == (c["width"] > 1024)
        for c in cases
        for s in c["samples"]
    )
    tmp = a.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    tmp.replace(a.output)
    return 0 if doc["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
