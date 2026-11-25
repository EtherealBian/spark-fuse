"""Whole-layer selection of one Qwen2 SwiGLU site, with independent replay and baseline timing."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import torch
import transformers
from paired_timing import measure
from provenance import capture
from qwen_packed_mean import compile_modes, make_model, observed, whole_check
from sparkfuse import Schedule, compile_export
from sparkfuse.contextual import _exact, _key, _value, replay_site, select_site
from sparkfuse.packed_mean import match
from sparkfuse.partition import _RegionModule, ordered_collectives, partition_export


CANDIDATES = [
    Schedule(128, "flat", False, 4),
    Schedule(256, "flat", False, 4),
    Schedule(128, "flat", False, 8),
]
ROOT = Path(__file__).resolve().parents[1]


def case(seq, weight, seed, smoke, frozen):
    torch.compiler.reset()
    model, make = make_model(weight, seq)
    ep = torch.export.export(model, make(seed))
    rms = Schedule(**frozen["kernels"]["packed"]["schedule"])
    options = dict(
        host_backend="native",
        schedule_policy=lambda p: rms if match(p) else ordered_collectives(p),
        rotary_pairs=True,
    )
    base, records = partition_export(ep, **options)
    (hit,) = [r for r in records if "silu" in r.get("ops", []) and "module" in r]
    name = hit["module"]
    assert hit["live_ins"] == ["linear_4", "linear_5"] and hit["live_outs"] == ["mul_9"]
    requests = [make(seed + i) for i in (11, 41, 73, 107, 151)]
    t0 = time.monotonic()
    _, receipt = select_site(
        ep,
        name,
        CANDIDATES,
        *requests[:3],
        seed=seed + 577,
        partition_options=options,
        min_gain=1.02,
    )
    elapsed = time.monotonic() - t0
    nominee = receipt["nominee"]
    anchor = receipt["anchor"]
    chosen = receipt["decision"]["selected"]
    label = {anchor: "incumbent"}
    label.setdefault(nominee, "nominee")
    label.setdefault(chosen, "selected")
    bank = {anchor: base}
    k = base.get_submodule(name).kernel
    if nominee != anchor:
        g, second = partition_export(ep, **options)
        assert second == records and str(g.graph) == str(base.graph)
        s = next(s for s in CANDIDATES if _key(s) == nominee)
        setattr(g, name, _RegionModule(compile_export(k.ep, schedule=s), "native"))
        bank[nominee] = g
    layer = {label[tag]: g for tag, g in bank.items()}
    layer["eager"] = model
    if not smoke:
        layer.update(compile_modes(model))
    trials = []
    aten = ep.module()
    for req in requests[3:]:
        gm, checks = replay_site(
            ep, json.loads(json.dumps(receipt)), req, partition_options=options
        )
        wires, gate, ys = observed(aten, hit, req)
        eager = model(*req)
        assert whole_check(ys, eager, req[-1])["passed"]
        assert _exact(gm(*req), bank[chosen](*req), "replay") == 2
        results = {n: whole_check(m(*req), eager, req[-1]) for n, m in layer.items()}
        targets = {
            label[t]: _exact(g.get_submodule(name)(*wires), gate, "fresh isolated gate")
            for t, g in bank.items()
        }
        assert all(results[label[t]]["passed"] for t in bank)
        trials.append(
            dict(
                inputs=[_value(x) for x in req],
                wires=[_value(x) for x in wires],
                gate=_value(gate[0]),
                outputs=[_value(x) for x in eager],
                replay=checks,
                targets=targets,
                baselines=results,
            )
        )
    allowed = [n for n in layer if all(t["baselines"][n]["passed"] for t in trials)]
    whole = (
        measure({n: (lambda n=n: layer[n](*req)) for n in allowed}, seed + 1009)
        if not smoke
        else None
    )
    iso = (
        measure(
            {label[t]: (lambda g=g: g.get_submodule(name)(*wires)) for t, g in bank.items()},
            seed + 1013,
        )
        if not smoke
        else None
    )
    print(
        seq,
        weight,
        anchor,
        "->",
        nominee,
        "selected",
        chosen,
        receipt["decision"].get("median_ratio"),
        {n: round(v, 3) for n, v in whole["median_us"]["device"].items()} if whole else "smoke",
        flush=True,
    )
    return dict(
        sequence=seq,
        weight_seed=weight,
        request_seed=seed,
        region=hit,
        records=records,
        rms=vars(rms),
        receipt=receipt,
        selection_wall_s=elapsed,
        label=label,
        trials=trials,
        admitted=allowed,
        whole=whole,
        isolated=iso,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    src = ROOT / "reports/qwen-packed-mean-heldout-gb10.json"
    frozen = {c["sequence"]: c for c in json.loads(src.read_bytes())["cases"]}
    res = dict(
        protocol="BF16 random-weight Qwen2DecoderLayer, batch 2. Optimize only gate schedule; pin each previously frozen residual RMSNorm schedule. Five shuffled discovery windows, seven confirmation windows if there is a challenger. Require 6/7 paired wins AND median >=1.02x. Audit without reselection; two more independent requests for receipt replay and baseline checks. Profile and time only methods meeting the strict visible-output gate on both. CUDA graph device time and synchronized fresh host dispatch reported separately; fixed inputs, not serving throughput.",
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),
        transformers=transformers.__version__,
        smoke=a.smoke,
        rms_report_sha256=sha256(src.read_bytes()).hexdigest(),
        provenance=capture(a.output.parent / "sources"),
        cases=[],
    )
    for seq in [33] if a.smoke else [33, 65, 129]:
        for weight in [517129] if a.smoke else [533111, 581119]:
            res["cases"].append(case(seq, weight, weight + 13007 * seq, a.smoke, frozen[seq]))
    a.output.write_text(json.dumps(res, allow_nan=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
