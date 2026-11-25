"""Isolate paired rotate-half from projection/attention. Compare reordered layouts and shifted pointers."""

import argparse
import json
from math import prod
from pathlib import Path
import random
from statistics import median

import torch
from provenance import capture
from sparkfuse.measure import compare, device_time, host_time
from sparkfuse.partition import partition_export


class Rotation(torch.nn.Module):
    def forward(self, q, k, c, s):
        def one(x):
            half = x.shape[-1] // 2
            return x * c + torch.cat((-x[..., half:], x[..., :half]), -1) * s

        return one(q), one(k)


CASES = [
    ("bshd-prefill", 2, 12, 2, 97, 128, torch.bfloat16, (True, True), False),
    ("bhsd-prefill", 2, 12, 2, 97, 128, torch.bfloat16, (False, False), False),
    ("shared-phase-mixed", 3, 5, 1, 17, 30, torch.float16, (True, False), True),
    ("single-token", 2, 12, 2, 1, 128, torch.bfloat16, (False, False), False),
    ("small-fp32", 2, 7, 3, 31, 14, torch.float32, (True, True), True),
]


@torch.inference_mode()
def bench(case):
    label, b, hq, hk, t, d, dt, orders, shared = case

    def inp(seed, shift):
        g = torch.Generator(device="cuda").manual_seed(seed)

        def tensor(shape, offset):
            raw = torch.randn(shape, device="cuda", dtype=dt, generator=g)
            v = torch.empty(prod(shape) + offset, device="cuda", dtype=dt)[offset:].view(shape)
            v.copy_(raw)
            return v

        qk = []
        for h, order, off in zip((hq, hk), orders, (1, 2), strict=True):
            x = tensor((b, t, h, d) if order else (b, h, t, d), off if shift else 0)
            qk.append(x.transpose(1, 2) if order else x)
        return (
            *qk,
            *(tensor((1 if shared else b, 1, t, d), off if shift else 0) for off in (3, 5)),
        )

    m = Rotation()
    ep = torch.export.export(m, inp(48767, False))
    gm, records = partition_export(ep, rotary_pairs=True, host_backend="native")
    (hit,) = [r for r in records if r.get("kind") == "rotary_pair"]
    paths = {
        "eager": m,
        "inductor": torch.compile(m, fullgraph=True, dynamic=False),
        "sparkfuse": gm,
    }
    data = []
    for shifted in (False, True):
        checks = {name: [] for name in paths}
        for seed in (49033, 49253, 49367, 49429):
            args = inp(seed, shifted)
            ref = m(*args)
            for k, f in paths.items():
                out = f(*args)
                inttype = torch.int32 if dt == torch.float32 else torch.int16
                diffs = [
                    int((x.contiguous().view(inttype) != y.contiguous().view(inttype)).sum())
                    for x, y in zip(out, ref, strict=True)
                ]
                strides = all(x.stride() == y.stride() for x, y in zip(out, ref, strict=True))
                row = dict(seed=seed, bit_mismatches=diffs, strides_match=strides)
                try:
                    row["errors"] = compare(out, ref)
                    row["passed"] = strides and (not any(diffs) or k == "inductor")
                except AssertionError as exc:
                    row.update(passed=False, failure=str(exc))
                checks[k].append(row)
        eligible = [k for k, rs in checks.items() if all(row["passed"] for row in rs)]
        rng = random.Random(6701)
        windows = []
        for i in range(7):
            order = eligible.copy()
            rng.shuffle(order)
            w = {"order": order, "device": {}, "host": {}}
            for k in order:

                def f():
                    return paths[k](*args)

                w["device"][k] = device_time(f, groups=7, unroll=8, replay=4)
                w["host"][k] = host_time(f, samples=15)
            windows.append(w)
        data.append(
            dict(
                offsets=[v.storage_offset() for v in args],
                strides=[list(v.stride()) for v in args],
                checks=checks,
                windows=windows,
                median_us={
                    metric: {
                        k: median(w[metric][k]["median_us"] for w in windows) for k in eligible
                    }
                    for metric in ("device", "host")
                },
                passed="sparkfuse" in eligible,
            )
        )
    return dict(
        label=label,
        logical_shapes=[list(v.shape) for v in args],
        dtype=str(dt),
        record=hit,
        exported_fx=str(ep.graph),
        mixed_fx=str(gm.graph),
        layouts=data,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    doc = dict(
        scope="A single isolated Q/K pair, no projections or attention. The two layouts per case use identical values, distinct storage offsets. All table values, including the second half, are sampled independently.",
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        export_seed=48767,
        seeds=[49033, 49253, 49367, 49429],
        timing_seed=49429,
        provenance=capture(root / ".runtime/rotary-panel/sources"),
        cases=[],
    )
    for c in CASES:
        result = bench(c)
        doc["cases"].append(result)
        print(
            result["label"],
            [(v["passed"], v["median_us"]["device"]) for v in result["layouts"]],
            flush=True,
        )
    doc["passed"] = all(v["passed"] for case in doc["cases"] for v in case["layouts"])
    a.output.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    raise SystemExit(0 if doc["passed"] else 2)
