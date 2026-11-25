"""One real HF BertSelfOutput on GB10: full module parity/timing and distinct region tuning."""

import argparse, json, random
from pathlib import Path
from statistics import median
import torch, transformers
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertSelfOutput
from sparkfuse.calibration import calibrate_export
from sparkfuse.measure import compare, device_time, host_time
from sparkfuse.partition import partition_export
from sparkfuse.tuning import backend_id


@torch.inference_mode()
def run(db, rows, dt, budget, verify_only):
    torch.manual_seed(4313)
    cfg = BertConfig(hidden_size=768, hidden_dropout_prob=0.1, layer_norm_eps=1e-12)
    mod = BertSelfOutput(cfg).eval().to(device="cuda", dtype=dt)
    seeds = (341, 425, 842, 1435)

    def inputs(seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        return tuple(
            torch.randn(1, rows, 768, device="cuda", dtype=dt, generator=g) for _ in range(2)
        )

    a, b, c, d = [inputs(s) for s in seeds]
    ep = torch.export.export(mod, a)
    base, regions = partition_export(ep, host_backend="native")
    fused = [r for r in regions if "module" in r]
    if len(fused) != 1 or not any("layer_norm" in n for n in fused[0]["ops"]):
        raise AssertionError(regions)
    report = dict(
        scope="One actual BertSelfOutput: linear + evaluation dropout + residual + affine LayerNorm; random fixed model weights; not encoder or token throughput.",
        torch=str(torch.__version__),
        transformers=transformers.__version__,
        compiler=backend_id(),
        gpu=torch.cuda.get_device_name(),
        dtype=str(dt),
        rows=rows,
        hidden_size=768,
        layer_norm_eps=1e-12,
        seeds=seeds,
        weight_seed=4313,
        graph=str(ep.graph),
        mixed=str(base.graph),
        regions=regions,
    )
    if verify_only:
        report["verified"] = [compare(base(*args), mod(*args)) for args in (a, b, c, d)]
        return report
    tuned, cal = calibrate_export(ep, a, b, c, db, budget=budget, seed=5833)
    report["calibration"] = cal
    paths = {
        "eager": mod,
        "inductor": torch.compile(mod, backend="inductor", fullgraph=True),
        "mixed_default": base,
        "mixed_selected": tuned,
    }
    rec = {}
    rng = random.Random(316)
    orderings = []
    for name, fn in paths.items():
        rec[name] = {"inputs": [], "passed": True}
        for i, args in enumerate((a, b, c, d)):
            try:
                rec[name]["inputs"].append(
                    dict(seed=seeds[i], errors=compare(fn(*args), mod(*args)))
                )
            except AssertionError as e:
                rec[name]["inputs"].append(dict(seed=seeds[i], error=str(e)))
                rec[name]["passed"] = False
        if rec[name]["passed"]:
            rec[name]["windows"] = []
    for _ in range(7):
        order = [k for k, v in rec.items() if v["passed"]]
        rng.shuffle(order)
        orderings.append(order)
        for k in order:
            fn = lambda: paths[k](*d)
            rec[k]["windows"].append(
                dict(device=device_time(fn, groups=7), host=host_time(fn, samples=21))
            )
    for item in rec.values():
        if item["passed"]:
            for kind in ("device", "host"):
                item[kind + "_us"] = median(x[kind]["median_us"] for x in item["windows"])
    report.update(backends=rec, ordered_windows=orderings)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--db", type=Path)
    p.add_argument("--rows", type=int, default=97)
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    p.add_argument("--budget", type=int, default=12)
    p.add_argument("--verify-only", action="store_true")
    a = p.parse_args()
    if not a.verify_only and a.db is None:
        p.error("--db required for calibration")
    doc = run(a.db, a.rows, getattr(torch, a.dtype), a.budget, a.verify_only)
    a.output.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(
        "BertSelfOutput verified"
        if a.verify_only
        else {
            k: (v["passed"], v.get("device_us"), v.get("host_us"))
            for k, v in doc["backends"].items()
        }
    )
    if not a.verify_only and not all(
        doc["backends"][k]["passed"] for k in ("mixed_default", "mixed_selected")
    ):
        raise SystemExit(2)
