"""Evaluate split-vs-unsplit legality on an actual Transformers BertLayer, not a hand-coded replica."""

import argparse, json, random, time
from pathlib import Path
from statistics import median
import torch, transformers
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertLayer
from sparkfuse.calibration import calibrate_export
from sparkfuse.partition import partition_export, ordered_collectives
from sparkfuse.measure import compare, device_time, host_time
from sparkfuse.tuning import backend_id
from provenance import capture


@torch.inference_mode()
def run(seq, db, verify_only, seeds):
    cfg = BertConfig(hidden_size=768, intermediate_size=3072, num_attention_heads=12)
    cfg._attn_implementation = "eager"
    torch.manual_seed(2643)
    model = BertLayer(cfg).eval().to(device="cuda", dtype=torch.bfloat16)

    def make(s):
        g = torch.Generator(device="cuda").manual_seed(s)
        return (torch.randn(1, seq, 768, generator=g, device="cuda", dtype=torch.bfloat16),)

    args = [make(s) for s in seeds]
    ep = torch.export.export(model, args[0])
    paths = {"eager": model}
    decisions = {}
    for label, enabled in [("unsplit", False), ("split", True)]:
        start = time.perf_counter()
        gm, rec = partition_export(ep, host_backend="native", split_collectives=enabled)
        paths[label] = gm
        decisions[label] = dict(
            regions=rec, fx=str(gm.graph), build_or_load_ms=(time.perf_counter() - start) * 1000
        )
    if not any("split" in r and "module" in r for r in decisions["split"]["regions"]):
        raise AssertionError("BertLayer did not exercise a legal split: " + str(decisions))
    result = dict(
        scope="One real randomly initialized BERT encoder layer, eval, BF16, batch 1. Eager self-attention, no mask. Not entire BERT.",
        torch=str(torch.__version__),
        transformers=transformers.__version__,
        compiler=backend_id(),
        device=torch.cuda.get_device_name(),
        weight_seed=2643,
        seeds=seeds,
        seq=seq,
        decisions=decisions,
        fx=str(ep.graph),
    )
    paths["ordered"], result["ordered_regions"] = partition_export(
        ep, host_backend="native", schedule_policy=ordered_collectives
    )
    paths["exported_aten"] = ep.module()
    if not verify_only:
        try:
            paths["selected"], result["calibration"] = calibrate_export(
                ep, *args[:3], db, budget=10, seed=2929
            )
        except AssertionError as exc:
            result["calibration"] = {"status": "rejected", "failure": str(exc)}
        paths["inductor"] = torch.compile(model, backend="inductor", fullgraph=True, dynamic=False)
    refs = [model(*a) for a in args]
    rows = {}
    for k, fn in paths.items():
        samples = []
        for s, a, ref in zip(seeds, args, refs):
            try:
                samples.append({"seed": s, "errors": compare(fn(*a), ref)})
            except AssertionError as e:
                samples.append({"seed": s, "failure": str(e)})
        rows[k] = dict(samples=samples, passed=all("errors" in v for v in samples))
    if "selected" not in paths and not verify_only:
        rows["selected"] = {"passed": False, "failure": result["calibration"]["failure"]}
    if not verify_only:
        orderings = []
        rng = random.Random(185)
        eligible = [k for k, v in rows.items() if v["passed"]]
        for k in eligible:
            rows[k]["windows"] = []
        for _ in range(7):
            order = eligible.copy()
            rng.shuffle(order)
            orderings.append(order)
            for k in order:
                f = lambda: paths[k](*args[-1])
                rows[k]["windows"].append(
                    dict(
                        device=device_time(f, groups=7, unroll=8, replay=4),
                        host=host_time(f, samples=15),
                    )
                )
        for k in eligible:
            for t in ("device", "host"):
                rows[k][t + "_us"] = median(w[t]["median_us"] for w in rows[k]["windows"])
        result["orderings"] = orderings
    result["backends"] = rows
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seq", type=int, default=33)
    p.add_argument("--db", type=Path)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[794, 933, 1665, 3023])
    a = p.parse_args()
    if len(set(a.seeds)) != len(a.seeds) or len(a.seeds) < 4:
        p.error("need at least four distinct seeds")
    if not a.verify_only and a.db is None:
        p.error("--db required unless --verify-only")
    source = capture(
        Path(__file__).resolve().parents[1] / ".runtime" / "bert-layer-source" / "sources"
    )
    r = run(a.seq, a.db, a.verify_only, a.seeds)
    r["provenance"] = source
    a.output.write_text(json.dumps(r, indent=2, allow_nan=False) + "\n")
    print(
        {k: (v["passed"], v.get("device_us"), v.get("host_us")) for k, v in r["backends"].items()}
    )
    raise SystemExit(
        0 if r["backends"]["ordered"]["passed"] and r["backends"]["exported_aten"]["passed"] else 2
    )
