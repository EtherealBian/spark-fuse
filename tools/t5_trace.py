"""Attribute real T5 module launch counts; never use profiler wall time as a benchmark."""

import argparse
from collections import Counter
import json
from pathlib import Path
import torch
import transformers
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5DenseGatedActDense
from bench import profile
from provenance import capture
from sparkfuse.measure import compare
from sparkfuse.partition import partition_export


@torch.inference_mode()
def trace(seq):
    out = {
        "seq": seq,
        "dtype": "bfloat16",
        "weight_seed": 71339,
        "input_seed": 97843,
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "device": torch.cuda.get_device_name(),
        "provenance": capture(Path(__file__).resolve().parents[1] / ".runtime/t5-trace/sources"),
        "cases": [],
    }
    for variant in ["gated-gelu", "gated-gelu_pytorch_tanh"]:
        cfg = T5Config(d_model=768, d_ff=3072, feed_forward_proj=variant, dropout_rate=0.0)
        torch.manual_seed(out["weight_seed"])
        m = T5DenseGatedActDense(cfg).eval().to("cuda", torch.bfloat16)
        x = torch.randn(
            1,
            seq,
            768,
            dtype=torch.bfloat16,
            device="cuda",
            generator=torch.Generator(device="cuda").manual_seed(out["input_seed"]),
        )
        ep = torch.export.export(m, (x,))
        gm, records = partition_export(ep, host_backend="native")
        ref = m(x)
        observed = {}
        for name, fn in [("eager", m), ("default_mixed", gm)]:
            check = compare(fn(x), ref)
            measurements = profile(lambda: fn(x))
            events = measurements.pop("gpu_events")
            observed[name] = dict(
                errors=check,
                **measurements,
                device_events=len(events),
                event_names=dict(Counter(events)),
            )
        out["cases"].append(dict(variant=variant, regions=records, observed=observed))
        print(
            variant,
            {
                n: (v["device_events"], v["allocation_peak_above_entry_bytes"])
                for n, v in observed.items()
            },
            flush=True,
        )
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seq", type=int, default=33)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(trace(a.seq), indent=2, allow_nan=False) + "\n")
