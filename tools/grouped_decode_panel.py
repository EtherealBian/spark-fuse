"""Shared-KV CTAs under hot, shared-prefix and distinct KV address patterns on one GPU."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
from decode_locality import experiment
from decode_sdpa_panel import inputs, cpu_reference, metrics
from examples.single_decode import SingleDecode
from ordered_score_panel import binary
from paired_timing import measure
from provenance import capture
from serving_clocks import clocks
from sparkfuse.decode_attention import compile_decode_attention
from token_quant_panel import digest

SEEDS = (2731711, 2742773)
SPECS = (
    (1, 16, 4, 257, 64, 0, "fp16", None),
    (2, 12, 3, 1025, 128, 2, "bf16", None),
    (1, 32, 8, 2049, 128, 0, "bf16", None),
    (1, 16, 4, 8193, 64, 0, "bf16", None),
    (1, 32, 4, 16385, 128, 0, "fp16", None),
)
# Frozen before the independent seeded trials: factorial partial-CTA schedules, plus ATen/Inductor.
ARMS = ((1, 32), (1, 128), (2, 32), (2, 128), (4, 32), (4, 128))


def case(spec, seed, smoke):
    m = SingleDecode().eval()
    rows = [inputs(spec, seed + 47 * i) for i in range(10)]
    ex, au, *held = rows
    ep = torch.export.export(m, ex)
    ref = [cpu_reference(xs, spec[-1]) for xs in held]
    expected = [m(*xs) for xs in held]
    hashes = [[digest(t) for t in xs] for xs in rows]
    paths = {"eager": m}
    meta = {}
    parity = {}
    low = {}
    errs = {}
    gates = {}
    for r, t in ARMS if not smoke else ((1, 32), (4, 32)):
        f = compile_decode_attention(ep, ex, au, tile=t, head_group=r)
        key = f"group{r}_tile{t}"
        paths[key] = f.run_bounded
        meta[key] = dict(plan=asdict(f.plan), binary=[binary(p) for p in f.programs])
        partial = [digest(f.programs[0](*xs)[0]) for xs in held]
        narrow = [digest(f.run_bounded(*xs)) for xs in held]
        if r == 1:
            low[t] = partial, narrow
        assert low[t] == (partial, narrow), "grouped CTA altered FP32 partials or final output"
        parity[key] = dict(partials=partial, narrow=narrow)
    if not smoke:
        f = torch.compile(m, fullgraph=True, dynamic=False)
        try:
            f(*ex)
        except (torch._dynamo.exc.BackendCompilerFailed, torch._dynamo.exc.Unsupported) as e:
            errs["inductor"] = type(e).__name__
        else:
            paths["inductor"] = f
    for name, fn in paths.items():
        outputs = [fn(*xs) for xs in held]
        g = [metrics(y, o, z, xs) for y, o, z, xs in zip(outputs, ref, expected, held, strict=True)]
        distinct = len({t.untyped_storage().data_ptr() for t in outputs}) == 8
        admitted = distinct and all(x["structure"] and x["outside_tolerance"] == 0 for x in g)
        gates[name] = dict(admitted=admitted, distinct=distinct, requests=g)
        assert admitted or name == "inductor"
    accepted = {n: f for n, f in paths.items() if gates[n]["admitted"]}
    warm = serving = locality = None
    if not smoke:
        warm = measure({n: (lambda f=f: f(*held[-1])) for n, f in accepted.items()}, seed + 181)
        serving = clocks(accepted, held, seed + 191)
        # Short S=257 cannot reach 3xL2 without an excessively long ring. Do not call it a cold-cache result.
        if spec[3] != 257:
            locality = experiment(spec, seed + 94499, m, accepted)
    assert hashes == [[digest(t) for t in xs] for xs in rows]
    return dict(
        spec=spec,
        seed=seed,
        input_sha256=hashes,
        cpu_fp64_sha256=[digest(x) for x in ref],
        export_graph=str(ep.graph),
        plan=meta,
        gates=gates,
        bitwise_parity=parity,
        build_failures=errs,
        hot=warm,
        requests=serving,
        locality=locality,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(4)
    report = dict(
        smoke=a.smoke,
        provenance=capture(a.output.parent / "sources"),
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        cases=[],
    )
    for spec in SPECS[1:2] if a.smoke else SPECS:
        for seed in (2789819,) if a.smoke else SEEDS:
            c = case(spec, seed, a.smoke)
            report["cases"].append(c)
            print(
                spec[3],
                seed,
                {k: round(v, 2) for k, v in c["hot"]["median_us"]["device"].items()}
                if c["hot"]
                else "qualified",
                flush=True,
            )
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
