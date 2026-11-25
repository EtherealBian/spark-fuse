"""Paired cache append: compare pure data movement across identical lengths/layouts/alignments."""

import argparse
import json
from math import prod
from pathlib import Path
import random
from statistics import median
import torch
from sparkfuse.cache_append import fuse_appends
from sparkfuse.partition import partition_export
from sparkfuse.measure import device_time, host_time
from kernel_trace import kernel_census
from provenance import capture


class Append(torch.nn.Module):
    def forward(self, k, kt, v, vt):
        return torch.cat((k, kt), -2), torch.cat((v, vt), -2)


CASES = [
    ("prefill-short", 2, 2, 31, 1, 128, torch.bfloat16, (True, False, True, False)),
    ("prefill-long", 2, 2, 1023, 1, 128, torch.bfloat16, (True, False, True, False)),
    ("carry-long", 2, 2, 1024, 1, 128, torch.bfloat16, (False,) * 4),
    ("chunk-mixed", 3, 4, 257, 7, 30, torch.float16, (True, False, False, True)),
    ("odd-fp32", 2, 3, 17, 3, 9, torch.float32, (False, True, True, False)),
]
SEEDS = (63031, 63149, 63277, 63391)


@torch.inference_mode()
def run(case):
    label, b, h, past, t, d, dt, layouts = case
    it = torch.int32 if dt == torch.float32 else torch.int16

    def make(seed, offs):
        g = torch.Generator(device="cuda").manual_seed(seed)
        args = []
        for length, sw, o in zip((past, t, past, t), layouts, offs, strict=True):
            sh = (b, length, h, d) if sw else (b, h, length, d)
            src = torch.randint(
                torch.iinfo(it).min,
                torch.iinfo(it).max,
                (prod(sh),),
                dtype=it,
                device="cuda",
                generator=g,
            )
            raw = torch.empty(src.numel() + o, dtype=it, device="cuda")
            raw[o:].copy_(src)
            x = raw[o:].view(dt).reshape(sh)
            args.append(x.transpose(1, 2) if sw else x)
        return tuple(args)

    m = Append()
    args = make(62981, (0,) * 4)
    ep = torch.export.export(m, args)
    gm, records = partition_export(ep, host_backend="native", paired_appends=True)
    (record,) = [r for r in records if r.get("kind") == "paired_append"]
    fixed = ep.module()
    (control,) = fuse_appends(fixed, "cuda", "native", adaptive=False)
    p = {
        "fixed_width": fixed,
        "eager": m,
        "sparkfuse": gm,
        "inductor": torch.compile(m, fullgraph=True, dynamic=False, backend="inductor"),
    }
    align = []
    for offs in [(0, 0, 0, 0), (1, 2, 3, 5)]:
        rows = {k: [] for k in p}
        for seed in SEEDS:
            a = make(seed, offs)
            ref = m(*a)
            for k, fn in p.items():
                try:
                    out = fn(*a)
                    mismatches = [
                        int((x.contiguous().view(it) != y.view(it)).sum())
                        for x, y in zip(out, ref, strict=True)
                    ]
                    strides = [x.stride() == y.stride() for x, y in zip(out, ref, strict=True)]
                    addr = [x.untyped_storage().data_ptr() for x in out]
                    fresh = len(set(addr)) == 2 and set(addr).isdisjoint(
                        x.untyped_storage().data_ptr() for x in a
                    )
                    sample = dict(
                        seed=seed,
                        unequal_bits=mismatches,
                        matching_strides=strides,
                        new_storage=fresh,
                        passed=not any(mismatches) and all(strides) and fresh,
                    )
                except Exception as exc:
                    if k != "inductor":
                        raise
                    sample = dict(seed=seed, passed=False, error=type(exc).__name__)
                rows[k].append(sample)
        eligible = [k for k, v in rows.items() if all(r["passed"] for r in v)]
        rng = random.Random(8191 + past)
        windows = []
        for _ in range(7):
            order = eligible.copy()
            rng.shuffle(order)
            win = dict(order=order, device={}, host={})
            for k in order:

                def fn():
                    return p[k](*a)

                win["device"][k] = device_time(fn, groups=9, unroll=16, replay=4)
                win["host"][k] = host_time(fn, samples=21)
            windows.append(win)
        meds = {
            m: {k: median(w[m][k]["median_us"] for w in windows) for k in eligible}
            for m in ("host", "device")
        }
        print(label, offs, meds["device"], flush=True)
        align.append(
            dict(
                offsets=offs,
                samples=rows,
                windows=windows,
                median_us=meds,
                passed={"fixed_width", "sparkfuse", "eager"}.issubset(eligible),
            )
        )
    witness = []
    if label == "carry-long":
        for off in (0, 1, 2, 4):
            a = make(SEEDS[-1], (off,) * 4)
            prof = kernel_census(lambda: gm(*a))
            witness.append(
                dict(
                    offset_elements=off,
                    input_address_mod_16=[x.data_ptr() % 16 for x in a],
                    profile=prof,
                )
            )
    return dict(
        label=label,
        packet_witness=witness,
        dtype=str(dt),
        logical_shape=[b, h, past, t, d],
        requested_token_major=layouts,
        record=record,
        control=control,
        layouts=align,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    doc = dict(
        scope="Two fresh out-of-place BHSD length-axis concatenations; no attention math. Identical arbitrary input bit patterns, separate storage offsets, all dtypes checked bitwise including NaN payloads. Seven shuffled same-input windows; device replay is hot-address and synchronized host includes allocation.",
        torch=str(torch.__version__),
        device=torch.cuda.get_device_name(),
        export_seed=62981,
        seeds=SEEDS,
        timing_seed=SEEDS[-1],
        provenance=capture(root / ".runtime/decode-audit/sources"),
        cases=[run(c) for c in CASES],
    )
    doc["passed"] = all(x["passed"] for c in doc["cases"] for x in c["layouts"])
    tmp = a.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    tmp.replace(a.output)
    raise SystemExit(0 if doc["passed"] else 2)
