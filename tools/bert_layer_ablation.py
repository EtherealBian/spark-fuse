"""Locate whole-layer divergence without using a holdout to choose a production schedule."""

import argparse
import itertools
import json
from pathlib import Path
import torch
from torch import fx
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertLayer
from provenance import capture
from sparkfuse.partition import partition_export
from sparkfuse.measure import compare


@torch.inference_mode()
def run(seq):
    cfg = BertConfig(hidden_size=768, intermediate_size=3072, num_attention_heads=12)
    cfg._attn_implementation = "eager"
    torch.manual_seed(2643)
    eager = BertLayer(cfg).eval().to(device="cuda", dtype=torch.bfloat16)
    seeds = [794, 933, 1665, 3023, 5853, 9419, 4123, 5227]
    args = [
        (
            torch.randn(
                1,
                seq,
                768,
                generator=torch.Generator(device="cuda").manual_seed(s),
                device="cuda",
                dtype=torch.bfloat16,
            ),
        )
        for s in seeds
    ]
    ep = torch.export.export(eager, args[0])
    ref = [eager(*a) for a in args]
    clean = ep.module()
    mixed, regions = partition_export(ep, host_backend="native")
    names = [r["module"] for r in regions if "module" in r]
    fused = [mixed.get_submodule(name) for name in names]
    oracles = [fx.GraphModule({}, m.kernel.ep.graph) for m in fused]
    assert 1 <= len(fused) <= 5

    def audit(model):
        records = []
        for seed, a, reference in zip(seeds, args, ref):
            out = model(*a)
            try:
                compare(out, reference)
                passed = True
            except AssertionError:
                passed = False
            x, y = out[0].float(), reference[0].float()
            r = (x - y).abs()
            records.append(
                dict(
                    seed=seed,
                    passed=passed,
                    unequal=int((x != y).sum()),
                    max_abs=float(r.max()),
                    mean_abs=float(r.mean()),
                )
            )
        return records

    records = {"exported_aten": audit(clean)}
    for mask in itertools.product([0, 1], repeat=len(names)):
        for name, b, f, o in zip(names, mask, fused, oracles):
            setattr(mixed, name, f if b else o)
        records["".join(map(str, mask))] = audit(mixed)
    return dict(
        scope="Diagnostic only; one BF16 BertLayer. 1=fused default, 0=original ATen region, order follows region_names. No timing or promotion.",
        shape=[1, seq, 768],
        region_names=names,
        decisions=regions,
        records=records,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seq", type=int, default=33)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    provenance = capture(Path(__file__).resolve().parents[1] / ".runtime/bert-ablation/sources")
    r = run(a.seq)
    r["provenance"] = provenance
    a.output.write_text(json.dumps(r, indent=2, allow_nan=False) + "\n")
    print({k: sum(x["passed"] for x in v) for k, v in r["records"].items()})
