"""Measure paired GQA head materialization, with the same raw inputs and unchanged output contract."""

import argparse
import json
from math import prod
from pathlib import Path
import random
from statistics import median

import torch
from kernel_trace import kernel_census
from provenance import capture
from sparkfuse.measure import device_time, host_time
from sparkfuse.partition import partition_export

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("tiny-bf16", 2, 2, 6, 32, 128, torch.bfloat16),
    ("decode-bf16", 2, 2, 6, 1025, 128, torch.bfloat16),
    ("odd-fp16", 3, 4, 5, 67, 31, torch.float16),
    ("odd-fp32", 1, 3, 2, 17, 9, torch.float32),
]
SEEDS = (66119, 66221, 66457, 66593)


class Heads(torch.nn.Module):
    def __init__(self, g):
        super().__init__()
        self.g = g

    def forward(self, k, v):
        b, h, s, d = k.shape
        g = self.g

        def expand(x):
            return x.unsqueeze(2).expand(b, h, g, s, d).reshape(b, h * g, s, d)

        return expand(k), expand(v)


@torch.inference_mode()
def run(case):
    tag, b, h, g, s, d, dt = case
    shape = (b, h, s, d)
    it = torch.int32 if dt == torch.float32 else torch.int16

    def inputs(seed, offs):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        xs = []
        for off in offs:
            data = torch.randint(
                torch.iinfo(it).min,
                torch.iinfo(it).max,
                (prod(shape),),
                device="cuda",
                generator=gen,
                dtype=it,
            )
            storage = torch.empty(prod(shape) + off, device="cuda", dtype=it)[off:]
            storage.copy_(data)
            xs.append(storage.view(dt).reshape(shape))
        return tuple(xs)

    m = Heads(g)
    ep = torch.export.export(m, inputs(66047, (0, 0)))
    gm, rr = partition_export(ep, host_backend="native", paired_head_repeats=True)
    (r,) = [x for x in rr if x.get("kind") == "paired_head_repeat"]
    backends = {
        "eager": m,
        "sparkfuse": gm,
        "inductor": torch.compile(m, backend="inductor", fullgraph=True, dynamic=False),
    }
    layouts = []
    for offs in [(0, 0), (1, 3)]:
        samples = {k: [] for k in backends}
        for seed in SEEDS:
            args = inputs(seed, offs)
            reference = m(*args)
            for k, fn in backends.items():
                try:
                    actual = fn(*args)
                    bad = [
                        int((x.contiguous().view(it) != y.view(it)).sum())
                        for x, y in zip(actual, reference, strict=True)
                    ]
                    strides = [
                        x.stride() == y.stride() for x, y in zip(actual, reference, strict=True)
                    ]
                    stor = {x.untyped_storage().data_ptr() for x in actual}
                    fresh = len(stor) == 2 and stor.isdisjoint(
                        x.untyped_storage().data_ptr() for x in args
                    )
                    rec = dict(
                        seed=seed,
                        unequal_bits=bad,
                        matching_strides=strides,
                        new_storage=fresh,
                        passed=not any(bad) and all(strides) and fresh,
                    )
                except Exception as exc:
                    if k != "inductor":
                        raise
                    rec = dict(seed=seed, passed=False, error=type(exc).__name__)
                samples[k].append(rec)
        eligible = [k for k, v in samples.items() if all(x["passed"] for x in v)]
        gen = random.Random(5209 + s)
        windows = []
        for _ in range(7):
            order = eligible.copy()
            gen.shuffle(order)
            win = {"order": order, "host": {}, "device": {}}
            for k in order:

                def fn():
                    return backends[k](*args)

                win["device"][k] = device_time(fn, groups=9, unroll=8, replay=3)
                win["host"][k] = host_time(fn, samples=21)
            windows.append(win)
        med = {
            m: {k: median(w[m][k]["median_us"] for w in windows) for k in eligible}
            for m in ("device", "host")
        }
        traces = {k: kernel_census(lambda: backends[k](*args)) for k in eligible}
        layouts.append(
            dict(
                offsets=offs,
                samples=samples,
                windows=windows,
                median_us=med,
                profiles=traces,
                passed={"sparkfuse", "eager"} <= set(eligible),
            )
        )
        print(tag, offs, med["device"], flush=True)
    nbytes = 2 * prod(shape) * torch.empty((), dtype=dt).element_size()
    return dict(
        label=tag,
        record=r,
        dtype=str(dt),
        input_bytes=nbytes,
        materialized_output_bytes=nbytes * g,
        layouts=layouts,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    doc = dict(
        scope="Only paired materialization, not attention, DRAM bandwidth or token throughput. Same unsigned bit patterns across alignments; seven shuffled timing windows on the last admitted input. Device time uses hot-address graphs; synchronized host includes fresh allocations. Sizes count logical input and output tensor bytes, not physical traffic.",
        torch=str(torch.__version__),
        device=torch.cuda.get_device_name(),
        export_seed=66047,
        seeds=SEEDS,
        timing_seed=SEEDS[-1],
        provenance=capture(ROOT / ".runtime/grouped-kv-lab/sources"),
        cases=[run(c) for c in CASES],
    )
    doc["passed"] = all(r["passed"] for c in doc["cases"] for r in c["layouts"])
    tmp = a.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    tmp.replace(a.output)
    raise SystemExit(0 if doc["passed"] else 2)
