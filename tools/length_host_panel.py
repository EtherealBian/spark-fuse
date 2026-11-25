"""Switch host dispatch only; replay and measure the same generated CUDA pair."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from decode_sdpa_panel import inputs, cpu_reference, metrics
from examples.length_decode import CanonicalDecode, export_length_decode
from length_decode_panel import counted_inductor
from ordered_score_panel import binary
from paired_timing import measure
from serving_clocks import clocks
from provenance import capture
from token_quant_panel import digest
from sparkfuse.length_attention import compile_length_decode
from sparkfuse.measure import host_time
from statistics import median
import random


SEEDS = (3197351, 3208319)
SIZES = (2, 31, 32, 33, 129, 513, 1025, 2049)
# Compare each dispatch on exactly the same request, both visible state and policy unchanged.
CONFIGS = ((2, 8, 2, 64, 2, "bf16"), (1, 16, 4, 128, 0, "fp16"))


def case(cfg, seed, smoke):
    b, h, g, d, mask, dt = cfg

    def make(s, i):
        xs = inputs((b, h, g, s, d, mask, dt, None), seed + i)
        return xs[:3] + ((xs[3].to(xs[0].dtype),) if mask else ())

    ex = make(17, 0)
    au = make(65, 73)
    model = CanonicalDecode()
    ep = export_length_decode(model, ex)
    objs = {
        name: compile_length_decode(ep, ex, au, host_backend=name) for name in ("ctypes", "native")
    }
    assert objs["native"].paths == objs["ctypes"].paths
    bins = [binary(SimpleNamespace(path=p, source=p.read_text())) for p in objs["ctypes"].paths]
    paths = {"eager": model, **{name: obj.run_bounded for name, obj in objs.items()}}
    graphs = []
    fail = {}
    torch._dynamo.reset()
    if not smoke:
        paths["inductor_dynamic"], graphs = counted_inductor(model, True)
    sizes = (2, 32, 33, 129) if smoke else SIZES
    req = [make(s, 571 + 173 * j + 2333 * i) for i, s in enumerate(sizes) for j in range(3)]
    hashes = [[digest(x) for x in xs] for xs in req]
    refs = [cpu_reference(xs, None) for xs in req]
    expected = [model(*xs) for xs in req]
    gates = {}
    for name, fn in paths.items():
        try:
            ys = [fn(*xs) for xs in req]
        except (torch._dynamo.exc.BackendCompilerFailed, torch._dynamo.exc.Unsupported) as exc:
            assert name == "inductor_dynamic"
            fail[name] = type(exc).__name__
            continue
        zs = [metrics(y, r, e, xs) for y, r, e, xs in zip(ys, refs, expected, req, strict=True)]
        distinct = len({y.untyped_storage().data_ptr() for y in ys}) == len(req)
        ok = distinct and all(z["structure"] and not z["outside_tolerance"] for z in zs)
        gates[name] = dict(admitted=ok, all_live=distinct, requests=zs)
        assert ok or name == "inductor_dynamic"
    for a, b2 in zip(gates["native"]["requests"], gates["ctypes"]["requests"], strict=True):
        assert a == b2
    allowed = {k: paths[k] for k, v in gates.items() if v["admitted"]}
    mixed = hot = checked = None
    calls_when_warm = len(graphs)
    if not smoke:
        mixed = clocks(allowed, [req[3 * i + 2] for i in range(8)], seed + 17)
        hot = {}
        for i in (0, 3, 6, 7):
            xs = req[3 * i + 1]
            hot[str(sizes[i])] = measure(
                {k: (lambda f=f: f(*xs)) for k, f in allowed.items()}, seed + sizes[i]
            )
        # Safety scans stay enabled in this separate host wall-clock panel. No capture.
        checked = {}
        rng = random.Random(seed + 47)
        for i in (0, 7):
            xs = req[3 * i + 1]
            windows = []
            arms = {
                nm + suffix: fn
                for nm, obj in objs.items()
                for suffix, fn in (("_checked", obj), ("_bounded", obj.run_bounded))
            }
            for _ in range(7):
                order = list(arms)
                rng.shuffle(order)
                w = {n: host_time(lambda: arms[n](*xs), samples=13)["samples_us"] for n in order}
                windows.append(dict(order=order, samples=w))
            checked[str(sizes[i])] = dict(
                windows=windows,
                median_us={k: median(median(w["samples"][k]) for w in windows) for k in arms},
            )
    assert calls_when_warm == len(graphs) and hashes == [[digest(x) for x in xs] for xs in req]
    return dict(
        config=cfg,
        seed=seed,
        lengths=sizes,
        same_binary=bins,
        inductor_graphs_when_warm=calls_when_warm,
        inductor_graphs_after_timing=len(graphs),
        input_sha256=hashes,
        cpu_fp64_sha256=[digest(r) for r in refs],
        gates=gates,
        build_failures=fail,
        hot=hot,
        mixed=mixed,
        checked=checked,
    )


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    data = dict(
        smoke=a.smoke,
        provenance=capture(a.output.parent / "sources"),
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        cases=[],
    )
    for cfg in CONFIGS[:1] if a.smoke else CONFIGS:
        for seed in (3220903,) if a.smoke else SEEDS:
            c = case(cfg, seed, a.smoke)
            data["cases"].append(c)
            print(cfg, seed, c["mixed"]["median_us"] if c["mixed"] else "verified", flush=True)
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
