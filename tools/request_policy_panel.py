"""Whole native request policy qualification on a BF16 HF BertLayer with frozen W8A8 Q/K/V."""

import argparse
import json
from pathlib import Path
from statistics import median
import tempfile
import time

import torch
import transformers
from bert_w8a8_panel import models, SPECS, oracle
from provenance import capture
from sparkfuse.contextual import _exact, _value
from sparkfuse.serving_policy import select_request_policy, replay_request_policy, _scores


SEEDS = (1840921, 1851289)


def requests(m, res, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [
        tuple(
            torch.randn(1, m, 768, generator=g, device="cuda", dtype=torch.bfloat16)
            for _ in range(2 if res else 1)
        )
        for _ in range(30)
    ]


def run(m, res, seed, objective):
    _, model = models(seed)
    del _
    data = requests(m, res, seed + 973 * m)
    ex, au = data[:2]
    d, c, h = data[2:10], data[10:18], data[18:26]
    new = data[26:]
    ep = torch.export.export(model, ex)
    refs = [model(*r) for r in new]
    cpu = oracle(model, new)
    with tempfile.TemporaryDirectory() as folder:
        file = Path(folder) / "choice.json"
        start = time.perf_counter_ns()
        exe, receipt = select_request_policy(
            ep, ex, au, d, c, h, objective=objective, seed=seed + m, receipt_path=file
        )
        with exe:
            setup_ms = (time.perf_counter_ns() - start) / 1e6
            for r, y in zip(new, refs, strict=True):
                _exact(exe(*r), y, "selected/unseen")
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        with replay_request_policy(ep, *new[:2], file, objective=objective) as replay:
            torch.cuda.synchronize()
            replay_ms = (time.perf_counter_ns() - start) / 1e6
            for r, y in zip(new[2:], refs[2:], strict=True):
                _exact(replay(*r), y, "replay/unseen")
        assert receipt == json.loads(file.read_bytes())
    body = receipt["body"]
    selected = body["decision"]["selected"]
    held = {k: median(v) for k, v in _scores(body["audit"], body["admitted"], 7).items()}
    return dict(
        seed=seed,
        m=m,
        residual=res,
        objective=objective,
        receipt=receipt,
        audit_us=held,
        selection_wall_ms=setup_ms,
        replay_wall_ms=replay_ms,
        new_cpu_qkv_oracle=cpu,
        new_ref_sha256=[[_value(t) for t in y] for y in refs],
        new_inputs_sha256=[[_value(t) for t in r] for r in new],
        saved_us_vs_independent=held["independent"] - held[selected],
        # Includes audit and writing the full receipt, excludes export and process startup.
        amortize_vs_independent_requests=(
            1000
            * (setup_ms - body["builds"]["independent"]["ms"])
            / (held["independent"] - held[selected])
            if held["independent"] > held[selected]
            else None
        ),
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    out = dict(
        protocol="Thirty independent input tuples per seed/shape. Two for capture, eight for discovery, eight for confirmation, eight for holdout timing; four new inputs validate the winner and two of these drive a winner-only recapture. Single native requests complete on the owner stream, or eight outstanding with independent fresh results, per-forward wall microseconds. Five randomly ordered discovery windows, seven paired confirmation windows with >=6 wins and >=1.03x median ratio. Seven untouched holdout windows cannot promote an arm. Setup includes hashing, validation, timings, cleanup and writing the receipt. No cold-start or model-quality claim.",
        smoke=a.smoke,
        torch=str(torch.__version__),
        transformers=transformers.__version__,
        device=torch.cuda.get_device_name(),
        provenance=capture(a.output.parent / "sources"),
        cases=[],
    )
    for seed in (1859963,) if a.smoke else SEEDS:
        for m, res in SPECS[:1] if a.smoke else SPECS:
            for objective in ("latency", "queue8"):
                torch.compiler.reset()
                r = run(m, res, seed, objective)
                out["cases"].append(r)
                print(
                    seed,
                    m,
                    objective,
                    r["receipt"]["body"]["decision"]["selected"],
                    r["audit_us"],
                    round(r["selection_wall_ms"]),
                    round(r["replay_wall_ms"]),
                    flush=True,
                )
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
