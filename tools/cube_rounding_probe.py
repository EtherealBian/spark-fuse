"""Compare ATen x.pow(3) with two possible element-type rounding rules."""

import argparse
import json
from pathlib import Path
import torch


def probe(seed=610032, n=98304):
    out = {
        "torch": str(torch.__version__),
        "device": torch.cuda.get_device_name(),
        "seed": seed,
        "n": n,
        "cases": [],
    }
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        x = torch.randn(
            n,
            dtype=dtype,
            device="cuda",
            generator=torch.Generator(device="cuda").manual_seed(seed),
        )
        y, fp = x.pow(3.0), x.float()
        stages, wide = (x * x) * x, (fp * fp * fp).to(dtype)
        out["cases"].append(
            {
                "dtype": str(dtype),
                "unequal_vs_round_each_product": int((y != stages).sum()),
                "unequal_vs_round_only_at_end": int((y != wide).sum()),
            }
        )
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    text = json.dumps(probe(), indent=2) + "\n"
    if a.output:
        a.output.write_text(text)
    else:
        print(text, end="")
