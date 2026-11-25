"""Separate softmax reduction order from register-cache code generation on GB10."""

import argparse
from hashlib import sha256
import json
from math import prod
from pathlib import Path
import random
import re
from statistics import median
import subprocess

import torch
from inspect_reduction_sass import op_counts
from kernel_trace import kernel_census
from provenance import capture
from sparkfuse import Schedule, Unsupported, compile_export
from sparkfuse.partition import partition_export
from sparkfuse.measure import device_time, host_time
from sparkfuse.native import native_dispatch

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("decode", 24, 1025, torch.bfloat16, False),
    ("visible", 24, 1537, torch.bfloat16, True),
    ("edge", 24, 2048, torch.float16, True),
    ("explicit", 24, 2048, torch.float16, True),
    ("many", 384, 2047, torch.bfloat16, True),
    ("float", 47, 1089, torch.float32, True),
]
SEEDS = (70769, 70879, 70969, 71069, 71171)
EXPORT = 70571


class Score(torch.nn.Module):
    def __init__(self, visible, explicit=False):
        super().__init__()
        self.visible = visible
        self.explicit = explicit

    def forward(self, x, mask):
        z = x * 0.08838834764831845 + mask
        p = (z.float() if self.explicit else z).softmax(-1, dtype=torch.float32)
        if not self.visible:
            return (p.to(x.dtype),)
        return (z, p) if x.dtype == torch.float32 else (p.to(x.dtype), z, p)


def inputs(rows, w, dt, seed, offset):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    ret = []
    for shape, off in zip(((rows, w), (1, w)), (offset, 3 * offset), strict=True):
        base = torch.randn(shape, generator=gen, dtype=dt, device="cuda")
        buf = torch.empty(prod(shape) + off, device="cuda", dtype=dt)[off:].view(shape)
        buf.copy_(base)
        ret.append(buf)
    ret[0].mul_(9)
    ret[1][:, ::67] = float("-inf")
    ret[1][:, 1::131] = -8
    return tuple(ret)


def badbits(a, b):
    it = torch.int32 if a.dtype == torch.float32 else torch.int16
    return int((a.contiguous().view(it) != b.contiguous().view(it)).sum())


def binary(k):
    so = k.path.with_suffix(".so")
    usage = subprocess.check_output(["cuobjdump", "--dump-resource-usage", str(so)], text=True)
    sass = op_counts(subprocess.check_output(["cuobjdump", "--dump-sass", str(so)], text=True))
    (resources,) = re.findall(r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)", usage)
    (inst,) = sass.values()
    return dict(
        source_sha256=sha256(k.source.encode()).hexdigest(),
        binary_sha256=sha256(so.read_bytes()).hexdigest(),
        resources=dict(
            zip(
                ("registers", "stack_bytes", "static_shared_bytes", "local_bytes"),
                map(int, resources),
                strict=True,
            )
        ),
        static_instruction_sites=inst,
    )


def panel(case, emit_dir=None):
    label, rows, w, dt, visible = case
    m = Score(visible, explicit=label == "explicit")
    ep = torch.export.export(m, inputs(rows, w, dt, EXPORT, 0))
    fns = {
        "eager": m,
        "inductor": torch.compile(m, backend="inductor", fullgraph=True, dynamic=False),
    }
    build = {}
    for name, sched in [
        ("block", Schedule(128, "block", True)),
        ("loop", Schedule(128, "warp_xor", True)),
        ("slots", Schedule(128, "warp_slots", True)),
    ]:
        try:
            k = compile_export(ep, schedule=sched)
        except Unsupported as exc:
            if label != "edge" or name not in ("loop", "slots"):
                raise
            build[name] = dict(rejected=str(exc))
            continue
        if emit_dir is not None:
            emit_dir.mkdir(parents=True, exist_ok=True)
            (emit_dir / f"{label}-{name}.cu").write_text(k.source)
        fns[name] = native_dispatch(k)
        build[name] = dict(schedule=vars(sched), **binary(k))
    decisions = []
    if label == "edge":
        gm, decisions = partition_export(ep, ordered_scores="slots", host_backend="native")
        assert (
            sum(
                r.get("half_to_float") is True and r.get("ordered_width_limit") == 1024
                for r in decisions
                if r.get("kind") == "reference_boundary"
            )
            == 1
        )
        fns["partition_aten"] = gm
    required = {"eager", "partition_aten"} if label == "edge" else {"slots", "loop", "eager"}
    layouts = []
    for off in (0, 1):
        samples = {k: [] for k in fns}
        for seed in SEEDS:
            a = inputs(rows, w, dt, seed, off)
            ref = m(*a)
            for name, fn in fns.items():
                try:
                    out = fn(*a)
                    bad = [badbits(x, y) for x, y in zip(out, ref, strict=True)]
                    strides = [x.stride() == y.stride() for x, y in zip(out, ref, strict=True)]
                    sample = dict(
                        seed=seed,
                        unequal_bits=bad,
                        matching_strides=strides,
                        passed=not any(bad) and all(strides),
                    )
                except Exception as exc:
                    if name != "inductor":
                        raise
                    sample = dict(seed=seed, error=type(exc).__name__, passed=False)
                samples[name].append(sample)
        eligible = [k for k, s in samples.items() if all(r["passed"] for r in s)]
        rng = random.Random(7031 + w)
        wins = []
        for _ in range(7):
            order = eligible.copy()
            rng.shuffle(order)
            win = dict(order=order, device={}, host={})
            for name in order:

                def invoke():
                    return fns[name](*a)

                win["device"][name] = device_time(invoke, groups=7, unroll=8, replay=3)
                win["host"][name] = host_time(invoke, samples=21)
            wins.append(win)
        med = {
            metric: {k: median(w[metric][k]["median_us"] for w in wins) for k in eligible}
            for metric in ("device", "host")
        }
        layouts.append(
            dict(
                offset_elements=[off, 3 * off],
                samples=samples,
                windows=wins,
                median_us=med,
                profiles={k: kernel_census(lambda: fns[k](*a)) for k in eligible},
                passed=required <= set(eligible),
            )
        )
        print(label, off, med["device"], flush=True)
    return dict(
        label=label,
        rows=rows,
        width=w,
        dtype=str(dt),
        visible=visible,
        builds=build,
        decisions=decisions,
        required=sorted(required),
        layouts=layouts,
    )


def order_witness():
    """Emulate lane sums with ordinary PyTorch ops; profiler the unmodified eager Softmax separately."""
    lane = torch.arange(32, device="cuda")
    cases = []
    for w in (1024, 1025, 1537, 2048, 2049, 4096):
        samples = []
        for sd in SEEDS:
            x, mask = inputs(24, w, torch.bfloat16, sd, 0)
            z = (x * 0.08838834764831845 + mask).float().contiguous()
            p = z.softmax(-1)
            terms = (z - z.amax(-1, keepdim=True)).exp()
            tiles = torch.nn.functional.pad(terms, (0, (-w) % 32)).reshape(24, -1, 32)
            mass = torch.zeros_like(tiles[:, 0])
            for term in tiles.unbind(1):
                mass = mass + term
            for shift in (16, 8, 4, 2, 1):
                mass = mass + mass.index_select(1, lane ^ shift)
            out = (tiles / mass[:, None, :]).reshape(24, -1)[:, :w]
            samples.append(
                dict(
                    seed=sd,
                    fp32_unequal_bits=badbits(out, p),
                    bf16_unequal_bits=badbits(out.bfloat16(), p.bfloat16()),
                    max_abs=float((out - p).abs().max()),
                )
            )
        cases.append(
            dict(
                width=w,
                samples=samples,
                framework_softmax_profile=kernel_census(lambda: z.softmax(-1)),
            )
        )
    return cases


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--emit-dir", type=Path)
    a = p.parse_args()
    d = dict(
        scope="Isolated contiguous masked score rows; vector mask broadcasts across rows. The three-output low precision case exposes normalized FP32 probabilities, rounded probabilities, and scores. Framework/tolerant agreement after a cast does not establish agreement before the cast. No all-model performance claim.",
        protocol="Five holdout seeds per storage layout. All methods tested on identical values at both offsets; only paths matching all output bits and strides on all five samples are timed for that case/layout. Seven shuffled windows, last seed used for timing. Static cuobjdump resources and SASS instruction sites for the actual compiled objects are not dynamic counts.",
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        seeds=SEEDS,
        export_seed=EXPORT,
        timing_seed=SEEDS[-1],
        provenance=capture(ROOT / ".runtime/wide-xor-lab/sources"),
        order_witness=order_witness(),
        cases=[panel(c, a.emit_dir) for c in CASES],
    )
    d["passed"] = all(r["passed"] for c in d["cases"] for r in c["layouts"])
    tmp = a.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    tmp.replace(a.output)
    return 0 if d["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
