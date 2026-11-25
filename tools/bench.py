"""One shape, three independent backends, same data and capture policy; JSON output."""

import argparse, json, platform, time
from pathlib import Path
import torch
from examples.graphs import case
from sparkfuse import compile_export, Schedule
from sparkfuse.measure import compare, device_time, host_time


def profile(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    del out
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        out = fn()
        torch.cuda.synchronize()
    kernels = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return {"gpu_events": kernels, "allocation_peak_above_entry_bytes": peak}


@torch.inference_mode()
def bench(name, rows, width, dt, seed=100, do_profile=False, sched=None):
    m, args = case(name, rows, width, dt, seed)
    eager = lambda: m(*args)
    ref = eager()
    torch.cuda.synchronize()
    start = time.perf_counter()
    ep = torch.export.export(m, args)
    export_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    k = compile_export(ep, schedule=sched)
    sf_ms = (time.perf_counter() - start) * 1000
    fast = lambda: k(*args)
    start = time.perf_counter()
    ind = torch.compile(m, backend="inductor", fullgraph=True, dynamic=False)
    indfn = lambda: ind(*args)
    indy = indfn()
    torch.cuda.synchronize()
    ind_ms = (time.perf_counter() - start) * 1000
    record = {
        "case": name,
        "shape": [rows, width],
        "dtype": str(dt),
        "seed": seed,
        "device": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "schedule": vars(k.schedule),
        "export_ms": export_ms,
        "sparkfuse_build_or_load_ms": sf_ms,
        "inductor_wrapper_plus_first_call_ms": ind_ms,
        "plan": k.plan.explain(),
        "torch_export_graph": str(ep.graph),
        "backends": {},
    }
    for label, fn in [("eager", eager), ("inductor", indfn), ("sparkfuse", fast)]:
        try:
            err = compare(fn(), ref, exact=(1,) if name in ("resnorm2", "scores2") else ())
            rec = dict(parity="pass", errors=err, device=device_time(fn), host_sync=host_time(fn))
        except AssertionError as exc:
            # Preserve the other backends. Never attribute speed to a result that missed parity.
            record["backends"][label] = dict(parity="fail", diagnostic=str(exc))
            continue
        if do_profile:
            rec.update(profile(fn))
        record["backends"][label] = rec
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="resnorm2")
    p.add_argument("--rows", type=int, default=32)
    p.add_argument("--width", type=int, default=1536)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--output", type=Path)
    opt = p.parse_args()
    rec = bench(opt.case, opt.rows, opt.width, getattr(torch, opt.dtype), do_profile=opt.profile)
    s = json.dumps(rec, indent=2)
    print(s) if opt.output is None else opt.output.write_text(s + "\n")

    if any(v["parity"] != "pass" for v in rec["backends"].values()):
        raise SystemExit(2)
