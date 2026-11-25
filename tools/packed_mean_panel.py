"""Select old/new schedules independently, then time fresh ordinary calls under a bitwise gate."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
from statistics import median

import torch
from torch.utils._pytree import tree_flatten
from examples.graphs import DecomposedNorm, case
from paired_timing import measure
from provenance import capture
from sparkfuse import compile_export
from sparkfuse.measure import device_time
from sparkfuse.tuning import candidates


class Visible(torch.nn.Module):
    """Permutes placeholders and returns the narrow residual before the weighted result."""

    def forward(self, w, skip, x):
        r = x + skip
        f = r.float()
        y = (f * (f.square().mean(-1, keepdim=True) + 2**-12).rsqrt()).to(w.dtype) * w
        return r, y


CASES = [
    (256, 6144, "bf16", False),
    (31, 1024, "bf16", False),
    (192, 2560, "bf16", False),
    (256, 5120, "bf16", False),
    (32, 3584, "bf16", False),
    (17, 4100, "f16", True),
    (63, 8160, "f16", False),
    (33, 2048, "bf16", True),
    (96, 7680, "f16", True),
]


def digest(x):
    return sha256(x.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def checks(y, ref):
    a, s = tree_flatten(y)
    b, t = tree_flatten(ref)
    assert s == t and all(
        x.shape == z.shape and x.dtype == z.dtype and x.device == z.device
        for x, z in zip(a, b, strict=True)
    )
    out = []
    for x, z in zip(a, b, strict=True):
        rtol, atol = (0.012, 0.0002) if z.dtype == torch.bfloat16 else (0.0018, 3e-5)
        assert torch.isfinite(z).all()
        out.append(
            {
                "bit_mismatches": int((x.view(torch.int16) != z.view(torch.int16)).sum()),
                "outside_tolerance": int((~torch.isclose(x, z, rtol=rtol, atol=atol)).sum()),
                "elements": z.numel(),
            }
        )
    return out


def good(check):
    return all(v["bit_mismatches"] == 0 for v in check)


def args_for(n, h, dt, seed, offsets, visible):
    _, a = case("decomp", n, h, dt, seed)
    a = tuple(
        torch.cat((x.new_zeros(off), x.reshape(-1)))[off:].reshape_as(x) if off else x
        for x, off in zip(a, offsets, strict=True)
    )
    return a[::-1] if visible else a


def key(s):
    return f"{s.layout}-t{s.threads}-v{s.vec}-c{int(s.cache)}"


def timed_windows(fns, rng, count):
    windows = []
    for _ in range(count):
        order = rng.sample(list(fns), len(fns))
        windows.append(
            dict(
                order=order,
                device={k: device_time(fns[k], unroll=8, replay=4, groups=7) for k in order},
            )
        )
    return windows


def metrics(windows):
    return {k: median(w["device"][k]["median_us"] for w in windows) for k in windows[0]["order"]}


def one(i, data, offset, fixed, verify):
    torch.compiler.reset()
    n, h, dt, visible = data
    name = f"{n}x{h}-{dt}-" + ("visible" if visible else "final")
    dtype = {"bf16": torch.bfloat16, "f16": torch.float16}[dt]
    seed = 162109 + i * 997
    m = Visible() if visible else DecomposedNorm()
    ex = args_for(n, h, dtype, seed, (0, 0, 0), visible)
    ep = torch.export.export(m, ex)
    base = compile_export(ep)
    grid = candidates(base.plan)
    assert len(grid) == 10 + 3 * sum(h % v == 0 for v in (2, 4, 8))
    prior = {key(s) for s in grid if s.layout != "mean_packed_tile"}
    tile = {key(s) for s in grid} - prior
    assert len(prior) == 10 and tile
    by_key = {key(s): s for s in grid}
    needed = (
        set(by_key)
        if fixed is None and not verify
        else (
            {key(base.schedule), *{k for z in fixed["alignments"] for k in z["selected"].values()}}
            if fixed
            else {key(base.schedule), min(tile)}
        )
    )
    kernels = {k: compile_export(ep, schedule=by_key[k]) for k in sorted(needed)}
    modes = {
        "eager": m,
        "inductor": torch.compile(m, fullgraph=True, dynamic=False),
        "inductor_casts": torch.compile(
            m, fullgraph=True, dynamic=False, options={"emulate_precision_casts": True}
        ),
    }
    rec = dict(
        id=name,
        shape=[n, h],
        dtype=dt,
        visible=visible,
        export_sha256=sha256(str(ep.graph).encode()).hexdigest(),
        export_seed=seed,
        default=key(base.schedule),
        kernels={
            k: dict(
                schedule=vars(z.schedule),
                cuda_sha256=sha256(z.source.encode()).hexdigest(),
                binary_sha256=sha256(z.path.with_name("kernel.so").read_bytes()).hexdigest(),
                cache_id=z.path.parent.name,
            )
            for k, z in kernels.items()
        },
        alignments=[],
    )
    for idx, offs in enumerate([(0, 0, 0), (1, 2, 3)]):
        cal = seed + 1237 + idx
        timeseed = seed + offset + 91309 + 149 * idx
        rng = random.Random(timeseed)
        samples = []
        quality = {}
        fast = {}
        select = None
        if fixed is not None:
            assert (
                fixed["id"] == name
                and fixed["export_sha256"] == rec["export_sha256"]
                and fixed["alignments"][idx]["offsets"] == list(offs)
            )
            select = dict(fixed["alignments"][idx]["selected"])
        elif verify:
            select = {"previous_best": key(base.schedule), "packed_best": min(tile)}
        else:
            a = args_for(n, h, dtype, cal, offs, visible)
            ref = m(*a)
            fns = {k: (lambda z=z: z(*a)) for k, z in kernels.items()}
            quality = {k: checks(fn(), ref) for k, fn in fns.items()}
            assert all(map(good, quality.values()))
            samples = timed_windows(fns, rng, 3)
            fast = metrics(samples)
            select = {
                "previous_best": min(sorted(prior), key=fast.__getitem__),
                "packed_best": min(sorted(tile), key=fast.__getitem__),
            }
        selected = {"default": base, **{k: kernels[v] for k, v in select.items()}}
        funcs = {**modes, **selected}
        trials = []
        for seed2 in [
            seed + offset + 32003 + idx,
            seed + offset + 56009 + idx,
            seed + offset + 85009 + idx,
        ]:
            a = args_for(n, h, dtype, seed2, offs, visible)
            ref = m(*a)
            values = {k: checks(fn(*a), ref) for k, fn in funcs.items()}
            trials.append(
                dict(
                    seed=seed2,
                    input_sha256=[digest(x) for x in a],
                    output_sha256=[digest(x) for x in tree_flatten(ref)[0]],
                    checks=values,
                )
            )
        allowed = [k for k in funcs if all(good(row["checks"][k]) for row in trials)]
        assert set(selected) | {"eager"} <= set(allowed), (name, offs, trials)
        fresh = {k: (lambda k=k: funcs[k](*a)) for k in allowed}
        prepared = {k: f.prepare(*a) for k, f in selected.items()}
        fr = measure(fresh, timeseed) if not verify else None
        pr = timed_windows(prepared, rng, 7) if not verify else None
        print(name, offs, select, fr["median_us"]["device"] if fr else allowed, flush=True)
        rec["alignments"].append(
            dict(
                offsets=offs,
                calibration_seed=cal if samples else None,
                calibration_checks=quality,
                calibration_windows=samples,
                calibration_medians=fast,
                selected=select,
                holdout=trials,
                admitted=allowed,
                timing_seed=timeseed,
                fresh=fr,
                prepared_windows=pr,
                prepared_medians=metrics(pr) if pr else {},
            )
        )
    return rec


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection", type=Path)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--index", type=int, action="append")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.seed_offset < 0 or (a.selection and not a.seed_offset):
        ap.error("frozen repeats require a positive seed offset")
    ids = list(range(len(CASES))) if a.index is None else a.index
    if any(i not in range(len(CASES)) for i in ids) or len(ids) != len(set(ids)):
        ap.error("invalid indices")
    raw = a.selection.read_bytes() if a.selection else None
    source = json.loads(raw) if raw else None
    if source and (
        source["case_indices"] != ids or source["verify_only"] or source["selection_sha256"]
    ):
        ap.error("selection must be the original timed discovery panel for exactly these indices")
    result = dict(
        protocol="Fresh ordinary callable for every calibration and fresh comparator. Three shuffled CUDA Graph windows to select the fastest of six tree plus four cohort schedules vs all divisible tile vectors x three CTAs, independently for each alignment. Three unseen requests must all be bitwise exact on all visible outputs; only passing paths timed on the third. Median of seven shuffled CUDA Graph and synchronized-host windows; separately report persistent-output SparkFuse-only replay. Repetitions freeze both schedule IDs from the source report. Bitwise admission is intentionally stricter than ordinary tolerance.",
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        verify_only=a.verify_only,
        seed_offset=a.seed_offset,
        selection_sha256=sha256(raw).hexdigest() if raw else None,
        case_indices=ids,
        provenance=capture(a.output.parent / "sources"),
        cases=[],
    )
    for j, i in enumerate(ids):
        result["cases"].append(
            one(i, CASES[i], a.seed_offset, source["cases"][j] if source else None, a.verify_only)
        )
    a.output.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
