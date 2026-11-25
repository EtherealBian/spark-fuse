"""Row-polymorphic, fresh-output host dispatch benchmark (serialize GPU use externally)."""

import argparse
import json
import random
import time
from pathlib import Path
from statistics import median
import torch
from torch.utils._pytree import tree_leaves
from examples.graphs import case
from provenance import capture
from sparkfuse import Schedule
from sparkfuse.dynamic import DynamicRows
from sparkfuse.measure import compare, device_time
from sparkfuse.tuning import backend_id


@torch.inference_mode()
def panel(name, h, dtype, verify_only):
    def request(n, i):
        _, a = case(name, n, h, dtype, seed=4109 + i)
        return (
            tuple(torch.cat([x.new_zeros(1), x.flatten()])[1:].view_as(x) for x in a)
            if i % 2
            else a
        )

    m, example = case(name, 7, h, dtype, seed=191)
    sym = torch.export.Dim("tokens", min=1, max=129)
    ep = torch.export.export(
        m, example, dynamic_shapes=tuple({0: sym} if x.ndim == 2 else None for x in example)
    )
    sched = Schedule(256, "flat", False, 4) if name == "gate" else Schedule(128, "block", True, 4)
    start = time.perf_counter()
    dyn = DynamicRows(ep, schedule=sched)
    plan_setup = time.perf_counter() - start
    start = time.perf_counter()
    native = dyn.native_dispatch()
    host_setup = time.perf_counter() - start
    runner = {"eager": m, "python": dyn, "native": native}
    if not verify_only:
        runner["inductor"] = torch.compile(m, fullgraph=True, dynamic=True)
    train = [1, 7, 32]
    held = [2, 3, 15, 31, 33, 47, 48, 64, 65, 96, 97, 128, 129]
    # Inputs and references are ready before timing; each call still allocates normal outputs.
    lengths = train + held
    requests = [request(n, i) for i, n in enumerate(lengths)]
    checks = {k: [] for k in runner}
    errors = {}
    exact = (1,) if name == "resnorm2" else ()
    for i, a in enumerate(requests):
        expected = m(*a)
        reference = dyn(*a)
        for k, fn in runner.items():
            try:
                got = fn(*a)
                quality = compare(got, expected, exact=exact)
                if k == "native":
                    # Byte comparison includes the sign of zero. Includes [R,1] for mean.
                    assert all(
                        torch.equal(x.view(torch.uint8), y.view(torch.uint8))
                        for x, y in zip(tree_leaves(got), tree_leaves(reference), strict=True)
                    )
                checks[k].append(dict(rows=lengths[i], offset=i % 2, seed=4109 + i, errors=quality))
            except AssertionError as e:
                if k != "inductor":
                    raise
                errors.setdefault(k, []).append(dict(rows=lengths[i], diagnostic=str(e)))
    doc = dict(
        case=name,
        width=h,
        dtype=str(dtype),
        bounds=dyn.bounds,
        example_rows=7,
        warmup_rows=train,
        held_out_rows=held,
        kernel_cache_id=dyn.kernel.path.parent.name,
        schedule=vars(sched),
        construction_s={"dynamic_plan": plan_setup, "native_host_plan": host_setup},
        checks=checks,
        rejected=errors,
        host_windows=[],
        device={},
        verify_only=verify_only,
    )
    if verify_only:
        return doc
    ids = list(range(len(train), len(lengths)))
    rng = random.Random(20261126)
    admitted = [k for k in runner if k not in errors]
    for k in admitted:
        for i in ids:
            runner[k](*requests[i])
    torch.cuda.synchronize()
    for _ in range(9):
        seq = ids * 3
        rng.shuffle(seq)
        order = admitted.copy()
        rng.shuffle(order)
        win = {}
        for k in order:
            fn = runner[k]
            samples = []
            for i in seq:
                a = requests[i]
                t = time.perf_counter_ns()
                fn(*a)
                torch.cuda.synchronize()
                samples.append((time.perf_counter_ns() - t) / 1000)
            win[k] = samples
        doc["host_windows"].append(
            dict(order=order, rows=[lengths[i] for i in seq], samples_us=win)
        )
    doc["host_us"] = {
        k: median(median(w["samples_us"][k]) for w in doc["host_windows"]) for k in admitted
    }
    doc["native_faster_windows"] = {
        k: sum(
            median(w["samples_us"]["native"]) < median(w["samples_us"][k])
            for w in doc["host_windows"]
        )
        for k in admitted
        if k != "native"
    }
    for n in (33, 129):
        a = requests[lengths.index(n)]
        order = admitted.copy()
        rng.shuffle(order)
        doc["device"][str(n)] = {k: device_time(lambda: runner[k](*a)) for k in order}
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--verify-only", action="store_true")
    a = p.parse_args()
    d = dict(
        provenance=capture(Path(__file__).resolve().parents[1] / ".runtime/dynamic-native/sources"),
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        compiler=backend_id(),
        method="Rank-2 dynamic exports. Identical seeded inputs across all backends. Single SparkFuse binary and native descriptor per workload. Raw synchronized host microseconds for nine randomized rounds, each with three shuffled traversals of 13 held-out lengths. CUDA Graph timings are per fixed row count and exclude host dispatch. Setup is reported separately (cache state not controlled). Inductor excluded from timing if any length fails eager-based parity.",
        cases=[],
    )
    for name, h, dt in [
        ("gate", 8960, torch.bfloat16),
        ("resnorm2", 1536, torch.bfloat16),
        ("mean", 768, torch.float32),
    ]:
        result = panel(name, h, dt, a.verify_only)
        d["cases"].append(result)
        print(
            name,
            result.get("host_us", "verify only"),
            "excluded:",
            list(result["rejected"]),
            flush=True,
        )
    t = a.output.with_suffix(".tmp")
    t.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    t.replace(a.output)
