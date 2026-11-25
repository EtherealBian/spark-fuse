"""Calibrate legal partitions on their *actual* intermediate activations, then audit the composed graph."""

from contextlib import closing
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from .codegen import Schedule
from .ir import Unsupported
from .measure import compare
from .partition import partition_export
from .runtime import compile_export
from .tuning import connect, tune


def _save(path, records):
    with closing(connect(path)) as db, db:
        db.executemany(
            "INSERT OR REPLACE INTO schedules VALUES (?,?)",
            [(r["key"], json.dumps(r)) for r in records],
        )


@torch.inference_mode()
def calibrate_export(
    ep,
    train,
    holdout,
    audit,
    path,
    *,
    device="cuda",
    host_backend="native",
    budget=12,
    seed=107,
    capture_bytes_limit=256 * 1024**2,
    exact_region=None,
    exact_outputs=(),
):
    """Three positional-argument tuples from independent requests; weights may be shared.

    Publishes selected schedules transactionally only after comparing the entire mixed graph
    against the original on `audit`. The memory limit bounds storage retained by the temporary
    input hooks, not allocator peak. No captured tensors are stored on disk.
    """
    if budget < 1 or capture_bytes_limit < 1:
        raise ValueError("positive budget and capture limit required")
    if not all(isinstance(a, tuple) for a in (train, holdout, audit)):
        raise TypeError("calibration inputs must be three positional-argument tuples")
    mixed, regions = partition_export(ep, device=device, host_backend=host_backend)
    names = [r["module"] for r in regions if "module" in r]
    if not names:
        raise Unsupported("no eligible regions to calibrate")
    samples = {n: [] for n in names}
    retained = {}
    retained_bytes = 0

    def watch(n):
        def on_input(mod, args):
            nonlocal retained_bytes
            for x in args:
                store = x.untyped_storage()
                key = (str(x.device), store.data_ptr())
                if key not in retained:
                    retained[key] = store.nbytes()
                    retained_bytes += retained[key]
            if retained_bytes > capture_bytes_limit:
                raise ValueError("captured storage exceeds calibration limit")
            samples[n].append(tuple(x.detach() for x in args))

        return on_input

    handles = []
    try:
        for n in names:
            handles.append(mixed.get_submodule(n).register_forward_pre_hook(watch(n)))
        for a in (train, holdout, audit):
            mixed(*a)
    finally:
        for h in handles:
            h.remove()
    records = {}
    for ix, n in enumerate(names):
        if len(samples[n]) != 3:
            raise Unsupported("expected one call per region per request")
        a, b, c = samples[n]
        sub = mixed.get_submodule(n).kernel.ep
        oracle = torch.fx.GraphModule({}, sub.graph)
        exact = (exact_region or {}).get(n, ())
        rec = tune(
            sub,
            a,
            oracle(*a),
            b,
            oracle(*b),
            path,
            budget=budget,
            seed=seed + ix,
            exact=exact,
            store=False,
        )
        k = compile_export(sub, device=device, schedule=Schedule(**rec["selected"]))
        rec["third_input_errors"] = compare(k(*c), oracle(*c), exact=exact)
        records[n] = rec
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".sparkfuse-calibration-", dir=target.parent) as tmp:
        trial = Path(tmp) / "trial.sqlite"
        _save(trial, records.values())
        tuned, actual = partition_export(
            ep, device=device, database=trial, host_backend=host_backend
        )
        applied = {r["module"]: r["schedule"] for r in actual if "module" in r}
        if applied != {n: r["selected"] for n, r in records.items()}:
            raise RuntimeError("repartition did not apply proposed schedules")
        errors = compare(tuned(*audit), ep.module()(*audit), exact=exact_outputs)
    _save(target, records.values())
    return tuned, dict(
        records=records,
        mixed_audit_errors=errors,
        captured_distinct_storage_bytes=retained_bytes,
        actual_regions=actual,
    )
