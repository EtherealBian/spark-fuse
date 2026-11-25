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
from .tuning import connect, signature, tune
from .sampled_exact import tune_shared_exact


def _save(path, records):
    by_key = {}
    for rec in records:
        other = by_key.setdefault(rec["key"], rec)
        if rec["selected"] != other["selected"]:
            raise AssertionError(
                "incompatible schedules for a shared dispatch key; dispatch not published"
            )
    with closing(connect(path)) as db, db:
        db.executemany(
            "INSERT OR REPLACE INTO schedules VALUES (?,?)",
            [(r["key"], json.dumps(r)) for r in by_key.values()],
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
    sampled_exact=False,
):
    """Three positional-argument tuples from independent requests; weights may be shared.

    Publishes selected schedules transactionally only after comparing the entire mixed graph
    against the original on train, holdout, and audit requests. The memory limit bounds storage retained by the temporary
    input hooks, not allocator peak. No captured tensors are stored on disk.
    sampled_exact jointly chooses one schedule per dispatch key on raw ATen activations
    and additionally requires equality at every region and composed output on these requests.
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

    # Execute the unfused region oracles during capture: candidates at later sites must
    # never be calibrated on the numerical drift of an earlier default kernel.
    kernels = {n: mixed.get_submodule(n).kernel for n in names}
    oracles = {n: torch.fx.GraphModule({}, k.ep.graph) for n, k in kernels.items()}
    handles = []
    try:
        for n in names:
            setattr(mixed, n, oracles[n])
            handles.append(oracles[n].register_forward_pre_hook(watch(n)))
        for a in (train, holdout, audit):
            mixed(*a)
    finally:
        for h in handles:
            h.remove()
    records = {}
    if sampled_exact:
        groups = {}
        for n in names:
            groups.setdefault(signature(kernels[n].plan, device), []).append(n)
        for ix, group in enumerate(groups.values()):
            sites = {n: [(a, oracles[n](*a)) for a in samples[n]] for n in group}
            rec = tune_shared_exact(kernels[group[0]].ep, sites, budget=budget, seed=seed + ix)
            records.update((n, rec) for n in group)
    else:
        for ix, n in enumerate(names):
            if len(samples[n]) != 3:
                raise Unsupported("expected one call per region per request")
            a, b, c = samples[n]
            sub = kernels[n].ep
            oracle = oracles[n]
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
        eager = ep.module()
        whole_errors = {}
        for phase, request in (("train", train), ("holdout", holdout), ("audit", audit)):
            try:
                ref = eager(*request)
                required = (
                    tuple(range(len(torch.utils._pytree.tree_leaves(ref))))
                    if sampled_exact
                    else exact_outputs
                )
                whole_errors[phase] = compare(tuned(*request), ref, exact=required)
            except AssertionError as e:
                raise AssertionError(
                    f"whole-graph {phase} rejected; dispatch not published: {e}"
                ) from e
    _save(target, records.values())
    return tuned, dict(
        records=records,
        mixed_audit_errors=whole_errors["audit"],
        mixed_request_errors=whole_errors,
        captured_distinct_storage_bytes=retained_bytes,
        actual_regions=actual,
        captured_from="unfused_aten",
        sampled_exact=sampled_exact,
    )
