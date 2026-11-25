"""Actual Transformers T5 gated FFN; region exactness and whole-module timings are separate claims."""

import argparse
import json
import random
from pathlib import Path
from statistics import median

import torch
import transformers
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5DenseGatedActDense

from provenance import capture
from sparkfuse.calibration import calibrate_export
from sparkfuse.measure import compare, device_time, host_time
from sparkfuse.partition import partition_export
from sparkfuse.tuning import backend_id


@torch.inference_mode()
def run(args):
    source = capture(Path(__file__).resolve().parents[1] / ".runtime/t5-gate/sources")
    cfg = T5Config(d_model=768, d_ff=3072, feed_forward_proj=args.proj, dropout_rate=0.0)
    torch.manual_seed(args.weight_seed)
    m = T5DenseGatedActDense(cfg).eval().to(device="cuda", dtype=torch.bfloat16)
    inputs = [
        (
            torch.randn(
                1,
                args.seq,
                768,
                device="cuda",
                dtype=torch.bfloat16,
                generator=torch.Generator(device="cuda").manual_seed(seed),
            ),
        )
        for seed in args.seeds
    ]
    ep = torch.export.export(m, inputs[0])
    mixed, regions = partition_export(ep, host_backend="native")
    fused = [r for r in regions if "module" in r]
    assert len(fused) == 1 and len(fused[0]["ops"]) >= 2, f"expected one gate: {regions}"
    result = dict(
        provenance=source,
        compiler=backend_id(),
        torch=str(torch.__version__),
        transformers=transformers.__version__,
        device=torch.cuda.get_device_name(),
        dtype="bfloat16",
        seq=args.seq,
        weight_seed=args.weight_seed,
        input_seeds=args.seeds,
        proj=args.proj,
        activation=type(m.act).__name__,
        fx=str(ep.graph),
        regions=regions,
        scope="One random-weight T5DenseGatedActDense, batch 1, width 768, FFN 3072. Three linear projections stay ATen. No encoder/decoder or checkpoint throughput measurement.",
    )
    paths = {"eager": m, "exported_aten": ep.module(), "mixed": mixed}
    if not args.verify_only:
        try:
            paths["sampled_exact"], result["calibration"] = calibrate_export(
                ep, *inputs[:3], args.db, sampled_exact=True, seed=1021, budget=10
            )
        except AssertionError as exc:
            result["calibration"] = {"status": "rejected", "failure": str(exc)}
        paths["inductor"] = torch.compile(m, backend="inductor", fullgraph=True, dynamic=False)
    refs = [m(*a) for a in inputs]
    rows = {}
    for name, f in paths.items():
        checks = []
        for seed, a, reference in zip(args.seeds, inputs, refs):
            try:
                checks.append(
                    {
                        "seed": seed,
                        "errors": compare(
                            f(*a), reference, exact=(0,) if name == "sampled_exact" else ()
                        ),
                    }
                )
            except AssertionError as exc:
                checks.append({"seed": seed, "failure": str(exc)})
        rows[name] = {"samples": checks, "passed": all("errors" in c for c in checks)}
    if "sampled_exact" not in paths and not args.verify_only:
        rows["sampled_exact"] = {"passed": False, "failure": result["calibration"]["failure"]}
    if not args.verify_only:
        order = [name for name in paths if rows[name]["passed"]]
        rng = random.Random(1363)
        orders = []
        for _ in range(7):
            rng.shuffle(order)
            orders.append(order.copy())
            for name in order:

                def call():
                    return paths[name](*inputs[-1])

                rows[name].setdefault("windows", []).append(
                    {
                        "device": device_time(call, groups=7, unroll=8, replay=4),
                        "host": host_time(call, samples=15),
                    }
                )
        for name in order:
            for kind in ("device", "host"):
                rows[name][kind + "_us"] = median(
                    w[kind]["median_us"] for w in rows[name]["windows"]
                )
        result["orderings"] = orders
    result["backends"] = rows
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--proj", choices=["gated-gelu", "gated-gelu_pytorch_tanh"], required=True)
    p.add_argument("--seq", type=int, default=33)
    p.add_argument("--weight-seed", type=int, default=9923)
    p.add_argument("--seeds", nargs="+", type=int, default=[7841, 15319, 39133, 57829])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--db", type=Path)
    p.add_argument("--verify-only", action="store_true")
    a = p.parse_args()
    if (
        len(set(a.seeds)) != len(a.seeds)
        or len(a.seeds) < 4
        or a.seq < 1
        or not (a.db or a.verify_only)
    ):
        p.error(
            "need four distinct input seeds, a positive seq and a database unless verifying only"
        )
    r = run(a)
    a.output.write_text(json.dumps(r, indent=2, allow_nan=False) + "\n")
    print(
        a.proj,
        [(k, v["passed"], v.get("device_us"), v.get("host_us")) for k, v in r["backends"].items()],
    )
    raise SystemExit(
        0 if r["backends"]["mixed"]["passed"] and r["backends"]["exported_aten"]["passed"] else 2
    )
