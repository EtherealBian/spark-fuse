"""Check the group-as-matrix-row GQA rewrite against actual exported Qwen2 BF16 matmuls."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch
import transformers
from provenance import capture
from qwen_cached_decode import bits, same, suite, trajectory
from qwen_decoder import stats

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("QK", "PV")
COHORTS = ((84470, (91001, 91121, 91301)), (94877, (97117, 97367, 97549)))
PREFIXES = (31, 32, 62, 63, 254, 255, 256, 1023, 1024)


class Tap(torch.fx.Interpreter):
    """Observe real operands and output without modifying the FX graph."""

    def __init__(self, gm):
        super().__init__(gm)
        self.mm = []

    def call_function(self, target, args, kwargs):
        out = super().call_function(target, args, kwargs)
        if target is torch.ops.aten.matmul.default:
            self.mm.append((*args, out))
        return out


def compare_bits(actual, ref):
    report = stats(actual, ref)
    different = bits(actual) != bits(ref)
    examples = []
    for idx in torch.nonzero(different)[:4].tolist():
        ij = tuple(idx)
        examples.append(
            dict(
                index=idx,
                eager_hex=f"{int(bits(ref)[ij]) & 65535:04x}",
                grouped_hex=f"{int(bits(actual)[ij]) & 65535:04x}",
            )
        )
    return {**report, "examples": examples}


def assess(mm, k, v):
    (q, kr, qk), (p, vr, pv) = mm
    b, hq, t, d = q.shape
    h, keys = k.shape[1:3]
    assert k.shape == v.shape == (b, h, keys, d) and hq % h == 0
    g = hq // h

    def repeat(x):
        return x[:, :, None].expand(b, h, g, keys, d).reshape(b, hq, keys, d)

    kref, vref = repeat(k).transpose(-1, -2), repeat(v)
    controls = {
        "key_operand": same(kref, kr),
        "value_operand": same(vref, vr),
        "qk": same(q @ kref, qk),
        "pv": same(p @ vref, pv),
    }
    if not all(controls.values()):
        raise RuntimeError("reconstructed original operands or matmuls do not match exported graph")
    qkg = (q.reshape(b, h, g * t, d) @ k.transpose(-1, -2)).reshape(b, hq, t, keys)
    pvg = (p.reshape(b, h, g * t, keys) @ v).reshape(b, hq, t, d)
    return dict(controls=controls, QK=compare_bits(qkg, qk), PV=compare_bits(pvg, pv))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    doc = dict(
        scope="Numerical admission only for folding query groups into matmul rows at single-token BF16 Qwen2 decode. No timing, compiler dispatch or model-level quality claim.",
        protocol="One real random-weight layer per cohort, first cached token after HF prefill. Capture actual QK/PV operands/results from torch.export. Verify reconstructed expanded operands and original matmuls exactly before comparing group-as-row. The second cohort uses independent weights and seeds.",
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        transformers=transformers.__version__,
        cuda=torch.version.cuda,
        provenance=capture(ROOT / ".runtime/grouped-kv-lab/sources"),
        cases=[],
    )
    for weight, seeds in COHORTS:
        model, rope = suite(weight - 60313)
        for past in PREFIXES:
            exp_seed = weight + 71
            args, _ = trajectory(model, rope, past, exp_seed)[0]
            ep = torch.export.export(model, args)
            tap = Tap(ep.module())
            row = dict(
                weight_seed=weight,
                export_seed=exp_seed,
                prefix=past,
                keys=past + 1,
                export_graph_sha256=sha256(str(ep.graph).encode()).hexdigest(),
                samples=[],
            )
            for seed in seeds:
                args, ref = trajectory(model, rope, past, seed)[0]
                tap.mm.clear()
                out = tap.run(*args)
                if not all(same(x, y) for x, y in zip(out, ref, strict=True)):
                    raise RuntimeError("export no longer matches stateful HF output")
                row["samples"].append(dict(seed=seed, **assess(tap.mm, *out[-2:])))
            doc["cases"].append(row)
            print(
                weight,
                past,
                [[s[k]["unequal_bits"] for k in LABELS] for s in row["samples"]],
                flush=True,
            )
    ops = [s[k] for row in doc["cases"] for s in row["samples"] for k in LABELS]
    doc["summary"] = dict(
        tensors=len(ops),
        tensors_with_bit_mismatch=sum(s["unequal_bits"] > 0 for s in ops),
        unequal_elements=sum(s["unequal_bits"] for s in ops),
        tolerance_failures=sum(not s["tolerance_pass"] for s in ops),
        max_abs=max(s["max_abs"] for s in ops),
        sampled_bit_exact=all(s["unequal_bits"] == 0 and s["finite"] for s in ops),
    )
    doc["decision"] = (
        "rejected_bit_exact"
        if not doc["summary"]["sampled_bit_exact"]
        else "no_counterexample_in_sample"
    )
    tmp = a.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    tmp.replace(a.output)
    print(doc["decision"], doc["summary"], flush=True)


if __name__ == "__main__":
    main()
