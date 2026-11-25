"""All 65,536 low-precision encodings; signed zero is significant, NaN payloads are not."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from provenance import capture
from sparkfuse import Schedule, compile_export
from sparkfuse.tuning import backend_id


class Activations(nn.Module):
    def forward(self, x):
        return (
            torch.nn.functional.gelu(x),
            torch.nn.functional.gelu(x, approximate="tanh"),
            x.pow(3.0),
        )


def inspect(x, ref, out):
    names = ("gelu_erf", "gelu_tanh", "scalar_cube")
    rows = {}
    for name, a, b in zip(names, ref, out):
        ab = a.view(torch.int16).flatten()
        bb = b.view(torch.int16).flatten()
        xn = x.view(torch.int16).flatten()
        both_nan = (a.isnan() & b.isnan()).flatten()
        mask = (ab != bb) & ~both_nan
        cnt = int(mask.sum())
        ids = mask.nonzero().flatten()[:12].cpu().tolist()

        def bits(t, i):
            return f"{int(t[i]) & 0xFFFF:04x}"

        rows[name] = {
            "encodings": x.numel(),
            "matched_nan_results": int(both_nan.sum()),
            "mismatches": cnt,
            "first_mismatches": [
                {"input": bits(xn, i), "reference": bits(ab, i), "actual": bits(bb, i)} for i in ids
            ],
        }
    return rows


@torch.inference_mode()
def audit():
    result = {
        "torch": str(torch.__version__),
        "device": torch.cuda.get_device_name(),
        "compiler": backend_id(),
        "provenance": capture(
            Path(__file__).resolve().parents[1] / ".runtime/activation-audit/sources"
        ),
        "scope": "All low-precision bit patterns, stable device/software version, elementwise only. Ignore payloads if both results are NaN; compare all other bits, including signed zero.",
        "cases": [],
    }
    for dtype in (torch.float16, torch.bfloat16):
        x = (
            torch.arange(65536, device="cuda", dtype=torch.int32)
            .to(torch.int16)
            .view(dtype)
            .reshape(256, 256)
        )
        model = Activations()
        ep = torch.export.export(model, (x,))
        for vec, offset in [(1, 0), (4, 1)]:
            src = x if not offset else torch.cat([x.new_zeros(1), x.flatten()])[1:].reshape_as(x)
            schedule = Schedule(128, "flat", False, vec)
            kernel = compile_export(ep, schedule=schedule)
            r = inspect(src, model(src), kernel(src))
            result["cases"].append(
                {
                    "dtype": str(dtype),
                    "offset_elements": offset,
                    "schedule": vars(schedule),
                    "outputs": r,
                }
            )
            print(dtype, vec, {k: v["mismatches"] for k, v in r.items()}, flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = audit()
    a.output.write_text(json.dumps(r, indent=2) + "\n")
    raise SystemExit(
        2 if any(v["mismatches"] for c in r["cases"] for v in c["outputs"].values()) else 0
    )
