"""Replay historical decomp RMSNorm requests and compare two BF16 rounding boundaries.

Counterfactuals are separate eager computations; no internal value is added to the
actual graph being compiled, correctness-checked or timed.
"""

import argparse
from hashlib import sha256
import gzip
import json
from pathlib import Path

import torch
import torch._inductor.config as config
from examples.graphs import case
from paired_timing import measure
from provenance import capture
from sparkfuse import Schedule, compile_export
from sparkfuse.measure import compare

RTOL, ATOL = 0.012, 0.0002


def digest(x):
    return sha256(x.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def mismatch(y, ref):
    if y.shape != ref.shape or y.dtype != torch.bfloat16 or ref.dtype != torch.bfloat16:
        raise ValueError("output metadata mismatch")
    if not torch.isfinite(y).all() or not torch.isfinite(ref).all():
        raise ValueError("expected finite random inputs and outputs")
    return int((y.view(torch.int16) != ref.view(torch.int16)).sum())


def counterfactuals(x, skip, w):
    unrounded = x.float() + skip.float()
    out = {}
    for h, f in [("rounded", unrounded.bfloat16().float()), ("wide", unrounded)]:
        mean = f.square().mean(-1, keepdim=True)
        inv = torch.rsqrt(mean + 1e-6)
        z = f * inv
        for n, v in [("rounded", z.bfloat16().float()), ("wide", z)]:
            out[h + "_" + n] = ((v * w.float()).bfloat16(), (f, mean, inv, z, v))
    return out


def assess(y, ref, oracles):
    bits = mismatch(y, ref)
    bad = ~torch.isclose(y, ref, rtol=RTOL, atol=ATOL, equal_nan=True)
    count = int(bad.sum())
    ok = True
    try:
        compare(y, ref)
    except AssertionError:
        ok = False
    assert ok == (count == 0)
    data = dict(
        passed=ok,
        elements=ref.numel(),
        bit_mismatches=bits,
        outside_tolerance=count,
        model_bit_mismatches={k: mismatch(y, v[0]) for k, v in oracles.items()},
    )
    if count:
        i, j = map(int, bad.nonzero()[0])
        pos = (i, j)

        def at(v):
            return float(v[i, 0 if v.shape[1] == 1 else j])

        data["first_outside"] = dict(
            index=pos,
            actual=float(y[pos]),
            expected=float(ref[pos]),
            model_values={
                k: dict(
                    residual=at(st[0]),
                    mean=at(st[1]),
                    rsqrt=at(st[2]),
                    normalized=at(st[3]),
                    before_weight=at(st[4]),
                    output=at(t),
                )
                for k, (t, st) in oracles.items()
            },
        )
    return data


def run(row, offset, verify):
    torch.compiler.reset()
    orig = row["comparison"]
    job = row["job"]
    n, w = orig["shape"]
    seed = orig["seed"]
    assert seed == job["seed"] + 2000
    m, example = case("decomp", n, w, torch.bfloat16, seed)
    ep = torch.export.export(m, example)
    assert str(ep.graph) == orig["torch_export_graph"]
    selected = compile_export(ep, schedule=Schedule(**orig["schedule"]))
    current = compile_export(ep)
    fns = dict(
        eager=m,
        selected=selected,
        current=current,
        inductor=torch.compile(m, backend="inductor", fullgraph=True, dynamic=False),
        inductor_casts=torch.compile(
            m,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
            options={"emulate_precision_casts": True},
        ),
    )
    req = []
    for s in (seed, seed + 104729 + offset, seed + 130363 + offset):
        _, args = case("decomp", n, w, torch.bfloat16, s)
        ref = m(*args)
        models = counterfactuals(*args)
        assert mismatch(models["rounded_rounded"][0], ref) == 0
        checks = {k: assess(fn(*args), ref, models) for k, fn in fns.items()}
        req.append(
            dict(
                seed=s,
                inputs_sha256=[digest(x) for x in args],
                output_sha256=digest(ref),
                checks=checks,
            )
        )
    allowed = [k for k in fns if all(z["checks"][k]["passed"] for z in req)]
    replay = {
        k: req[0]["checks"][k]["passed"] == (orig["backends"][old]["parity"] == "pass")
        for k, old in [("eager", "eager"), ("selected", "sparkfuse"), ("inductor", "inductor")]
    }
    rec = dict(
        id=row["id"],
        cohort=row["cohort"],
        original_worker_sha256=row["worker_sha256"],
        original_admission_reproduced=replay,
        export_sha256=sha256(str(ep.graph).encode()).hexdigest(),
        seeds=req,
        admitted=allowed,
        current_schedule=vars(current.schedule),
        selected_schedule=vars(selected.schedule),
        cuda_sha256={
            k: sha256(v.source.encode()).hexdigest()
            for k, v in [("current", current), ("selected", selected)]
        },
        timing_seed=None,
        timing=None,
    )
    if not verify:
        rec.update(
            timing_seed=s, timing=measure({k: (lambda k=k: fns[k](*args)) for k in allowed}, s)
        )
    print(
        row["id"],
        row["cohort"],
        allowed,
        [z["checks"]["inductor"]["outside_tolerance"] for z in req],
        flush=True,
    )
    return rec


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, default=Path("reports/decomp-cast-catalog.json.gz"))
    p.add_argument("--case", action="append")
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.seed_offset < 0:
        p.error("seed offset must be nonnegative")
    raw = a.catalog.read_bytes()
    catalog = json.loads(gzip.decompress(raw))
    rows = [x for x in catalog["cases"] if not a.case or x["id"] in a.case]
    if not rows or (a.case and len(rows) != len(set(a.case))):
        p.error("unknown case")
    out = dict(
        protocol="Fixed historical selected schedule, unchanged module for both Inductor options. The original comparison input and two independent seeds must all pass the original BF16 assert_close gate before timing the last input, in seven shuffled hot-device and synchronized-host windows. Four eager counterfactuals preserve or omit the BF16 round after the residual add and before the weight multiply; they are diagnostics, not generated code or timed paths.",
        rtol=RTOL,
        atol=ATOL,
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        default_emulate_precision_casts=config.emulate_precision_casts,
        comparator_options={"inductor": {}, "inductor_casts": {"emulate_precision_casts": True}},
        seed_offset=a.seed_offset,
        catalog_sha256=sha256(raw).hexdigest(),
        provenance=capture(a.output.parent / "sources"),
        cases=[run(row, a.seed_offset, a.verify_only) for row in rows],
    )
    out["passed"] = all({"eager", "current"} <= set(z["admitted"]) for z in out["cases"])
    a.output.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    if not out["passed"]:
        raise SystemExit("current SparkFuse missed the historical tolerance; report retained")


if __name__ == "__main__":
    main()
