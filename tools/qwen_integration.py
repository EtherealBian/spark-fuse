"""Real library modules with random weights; no checkpoint or token-generation benchmark."""

import argparse, json
from pathlib import Path
import torch, transformers
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RMSNorm
from transformers import Qwen2Config
from sparkfuse.partition import partition_export
from sparkfuse.measure import compare, device_time


@torch.inference_mode()
def run(rows=32):
    torch.manual_seed(711)
    cfg = Qwen2Config(
        hidden_size=1536, intermediate_size=8960, num_attention_heads=12, num_key_value_heads=2
    )
    dtype = torch.bfloat16
    a = (torch.randn(1, rows, 1536, device="cuda", dtype=dtype),)
    result = {}
    for label, mod in [("rms_norm", Qwen2RMSNorm(1536, eps=1e-6)), ("mlp", Qwen2MLP(cfg))]:
        mod = mod.eval().to(device="cuda", dtype=dtype)
        ep = torch.export.export(mod, a)
        gm, regions = partition_export(ep)
        if sum("module" in r for r in regions) != 1:
            raise AssertionError(regions)
        ref = mod(*a)
        fn = lambda: gm(*a)
        errors = compare(fn(), ref)
        result[label] = {
            "regions": regions,
            "original": str(ep.graph),
            "partitioned": str(gm.graph),
            "errors": errors,
            "eager_device": device_time(lambda: mod(*a)),
            "mixed_device": device_time(fn),
        }
    return {
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "random_seed": 711,
        "hidden_size": 1536,
        "intermediate_size": 8960,
        "rows": rows,
        "dtype": str(dtype),
        "modules": result,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rows", type=int, default=32)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a.rows), indent=2) + "\n")
