"""Compare two reduction policies at identical real BERT activations and at whole-layer outputs."""

import argparse
import json
import torch
from torch import fx, nn
from pathlib import Path
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertLayer
from sparkfuse.partition import partition_export, ordered_collectives
from sparkfuse.measure import compare
from sparkfuse.tuning import backend_id
from provenance import capture


def error(out, ref):
    try:
        compare(out, ref)
        passed = True
    except AssertionError:
        passed = False
    a = torch.utils._pytree.tree_leaves(out)
    b = torch.utils._pytree.tree_leaves(ref)
    return dict(
        passed=passed,
        unequal=sum(int((x != y).sum()) for x, y in zip(a, b)),
        n=sum(x.numel() for x in a),
        max_abs=max(float((x.float() - y.float()).abs().max()) for x, y in zip(a, b)),
    )


class Tap(nn.Module):
    def __init__(self, k, rec):
        super().__init__()
        self.k = k
        self.ref = fx.GraphModule({}, k.kernel.ep.graph)
        self.rec = rec

    def forward(self, *args):
        exact = self.ref(*args)
        self.rec.append(error(self.k(*args), exact))
        return exact


@torch.inference_mode()
def run(seq, weight_seed, affine, seeds):
    cfg = BertConfig(hidden_size=768, intermediate_size=3072, num_attention_heads=12)
    cfg._attn_implementation = "eager"
    torch.manual_seed(weight_seed)
    model = BertLayer(cfg).eval()
    if affine == "perturbed":
        for mod in model.modules():
            if isinstance(mod, nn.LayerNorm):
                mod.weight.normal_(1, 0.15)
                mod.bias.normal_(0, 0.05)
    model.to(device="cuda", dtype=torch.bfloat16)
    args = [
        (
            torch.randn(
                1,
                seq,
                768,
                device="cuda",
                dtype=torch.bfloat16,
                generator=torch.Generator(device="cuda").manual_seed(s),
            ),
        )
        for s in seeds
    ]
    ep = torch.export.export(model, args[0])
    refs = [model(*a) for a in args]
    rows = {}
    for k, policy in [("default", None), ("ordered", ordered_collectives)]:
        gm, rec = partition_export(ep, host_backend="native", schedule_policy=policy)
        errors = [error(gm(*a), ref) for a, ref in zip(args, refs)]
        isolated = {}
        for r in rec:
            if "module" not in r:
                continue
            name = r["module"]
            isolated[name] = []
            setattr(gm, name, Tap(gm.get_submodule(name), isolated[name]))
        # Tap returns unfused ATen, so every later region gets the *same* value for both policies.
        for a, ref in zip(args, refs):
            e = error(gm(*a), ref)
            if e["unequal"]:
                raise AssertionError("unfused exported reference differs from module")
        rows[k] = dict(regions=rec, isolated=isolated, whole=errors)
    return dict(seq=seq, weight_seed=weight_seed, affine=affine, seeds=seeds, backends=rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    seeds = [18144, 83116, 17320, 54211, 39943, 25041, 97182, 61021]
    result = dict(
        scope="One randomly initialized BF16 BertLayer, same ATen boundary inputs. Records in listed seed order. No tuning or timing.",
        compiler=backend_id(),
        provenance=capture(
            Path(__file__).resolve().parents[1] / ".runtime/ordered-boundaries/sources"
        ),
        cases=[],
    )
    for seq, weight, affine in [
        (33, 2643, "identity"),
        (97, 1332, "identity"),
        (33, 3308, "perturbed"),
        (97, 1348, "perturbed"),
    ]:
        row = run(seq, weight, affine, seeds)
        result["cases"].append(row)
        tmp = a.output.with_suffix(".pending")
        tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        tmp.replace(a.output)
        print(
            seq,
            weight,
            {
                k: (
                    sum(x["passed"] for x in v["whole"]),
                    sum(x["unequal"] for x in v["whole"]),
                    [sum(x["unequal"] for x in ss) for ss in v["isolated"].values()],
                )
                for k, v in row["backends"].items()
            },
            flush=True,
        )
