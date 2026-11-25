"""Select a frozen W8A8 lowering using complete, rotating-address native requests."""

from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
from statistics import median
import time

import torch
from torch.utils import _pytree as tree

from .contextual import SiteMismatch, _describe, _exact, _hash, _value, _write_once
from .ir import Unsupported
from .resident import _request, _store, _validate_export, capture_export
from .runtime import toolchain

SCHEMA = "sparkfuse.request-policy.v1"
ANCHOR = "independent"
ARMS = (ANCHOR, "shared", "packed")
OBJECTIVES = ("latency", "queue8")


def _context(ep, example, objective):
    if objective not in OBJECTIVES:
        raise ValueError("objective must be latency or queue8")
    layout = _describe(example)
    dev = example[0].device
    if (
        torch.cuda.current_device() != dev.index
        or torch.cuda.is_current_stream_capturing()
        or torch.is_autocast_enabled("cuda")
    ):
        raise Unsupported("requires owning device without autocast or nested capture")
    _request(example, dev)
    _validate_export(ep, quantized=True, activation=True)
    state = {}
    aliases = {}
    for group, entries in [("state", ep.state_dict), ("constant", ep.constants)]:
        for name, t in entries.items():
            if not isinstance(t, torch.Tensor) or t.layout != torch.strided or t.device != dev:
                raise Unsupported("requires dense tensor state on the target device")
            state[group + ":" + name] = [
                str(t.dtype),
                list(t.shape),
                list(t.stride()),
                t.storage_offset(),
                t.data_ptr() % 32,
                aliases.setdefault(_store(t), len(aliases)),
                _value(t),
            ]
    root = Path(__file__).parent
    files = sorted([*root.glob("*.py"), *(root / "csrc").glob("*.cpp")])
    prop = torch.cuda.get_device_properties(dev)
    driver = Path("/proc/driver/nvidia/version")
    if not driver.is_file() or not getattr(prop, "uuid", None):
        raise Unsupported("requires identifiable CUDA device and driver")
    return dict(
        schema=SCHEMA,
        objective=objective,
        inputs=layout,
        graph=_hash(str(ep.graph)),
        signature=_hash(str(ep.graph_signature)),
        trees=_hash(
            [tree.treespec_dumps(ep.call_spec.in_spec), tree.treespec_dumps(ep.call_spec.out_spec)]
        ),
        state=_hash(state),
        device=_hash(str(prop.uuid)),
        chip=prop.name,
        cc=[prop.major, prop.minor],
        sm=prop.multi_processor_count,
        driver=sha256(driver.read_bytes()).hexdigest(),
        torch=str(torch.__version__),
        torch_git=torch.version.git_version,
        cuda=torch.version.cuda,
        nvcc=_hash(toolchain()[1]),
        project=_hash(
            {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest() for p in files}
        ),
        precision=[
            torch.get_float32_matmul_precision(),
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            torch.backends.cuda.matmul.allow_tf32,
            torch.are_deterministic_algorithms_enabled(),
            os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            os.environ.get("NVIDIA_TF32_OVERRIDE"),
        ],
    ), set(aliases)


def _requests(context, groups, sizes, state, previous=()):
    seen = set(previous)
    storages = set(state)
    out = {}
    for (name, rows), size in zip(groups.items(), sizes, strict=True):
        if not isinstance(rows, (list, tuple)) or len(rows) != size:
            raise ValueError(f"{name}: expected {size} requests")
        hashes = []
        for row in rows:
            if _describe(row) != context["inputs"]:
                raise ValueError(f"{name}: incompatible input descriptor")
            _request(row, row[0].device)
            ptrs = {_store(x) for x in row}
            if ptrs & storages:
                raise ValueError(f"{name}: repeated or state-aliased request storage")
            storages.update(ptrs)
            h = _hash([_value(x) for x in row])
            hashes.append(h)
            if h in seen:
                raise ValueError(f"{name}: repeated request content")
            seen.add(h)
        out[name] = hashes
    return out


def _capture(ep, ex, au, name):
    return capture_export(
        ep,
        ex,
        au,
        partition_options=dict(
            token_quant=True,
            activation_quant=True,
            share_activation_quant=name != "independent",
            affine_epilogues=True,
            group_epilogues=True,
            projection_epilogue=True,
            host_backend="native",
            min_ops=100000,
        ),
        request_backend="native",
        int8_rhs_layout="packed_fanout" if name == "packed" else "column_major",
    )


def _kinds(handle, name):
    kinds = Counter(r["kind"] for r in handle.metadata["regions"] if "kind" in r)
    if (
        not (kinds["activation_token_quant"] or kinds["residual_token_quant"])
        or (name != "independent" and not kinds["shared_hidden_token_quantizers"])
        or (name == "packed" and not kinds["packed_int32_projection"])
    ):
        raise Unsupported(f"{name}: required W8A8 rewrite did not fire")
    return dict(kinds)


def _gate(handle, rows, refs, state):
    # Keep results alive across all calls: a borrowed output/internal staging buffer must fail.
    outputs = [handle(*r) for r in rows]
    forbidden = state | {_store(x) for row in rows for x in row}
    counts = []
    for got, ref in zip(outputs, refs, strict=True):
        counts.append(_exact(got, ref, "whole request"))
        for x, y in zip(tree.tree_leaves(got), tree.tree_leaves(ref), strict=True):
            if x.storage_offset() != y.storage_offset() or _store(x) in forbidden:
                raise SiteMismatch("whole request: output offset or ownership")
            forbidden.add(_store(x))
    return counts


def _windows(handles, rows, objective, seed, n):
    stream = torch.cuda.current_stream()
    rng = random.Random(seed)
    out = []
    depth = 8 if objective == "queue8" else 1
    for w in range(n):
        order = sorted(handles)
        rng.shuffle(order)
        samples = {}
        for name in order:
            fn = handles[name]
            values = []
            for j in range(8):
                args = [rows[(w + j + i) % len(rows)] for i in range(depth)]
                stream.synchronize()
                start = time.perf_counter_ns()
                result = fn(*args[0]) if depth == 1 else [fn(*r) for r in args]
                stream.synchronize()
                stop = time.perf_counter_ns()
                del result
                values.append((stop - start) / (1000 * depth))
            samples[name] = values
        out.append(dict(order=order, samples_us=samples))
    return out


def _scores(rows, arms, n):
    if len(rows) != n:
        raise ValueError("invalid number of windows")
    scores = {k: [] for k in arms}
    for w in rows:
        if sorted(w["order"]) != sorted(arms) or set(w["samples_us"]) != set(arms):
            raise ValueError("invalid arms in timing windows")
        for k, v in w["samples_us"].items():
            if len(v) != 8 or any(
                type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in v
            ):
                raise ValueError("invalid request timing")
            scores[k].append(median(v))
    return scores


def _nominate(rows, arms):
    s = _scores(rows, arms, 5)
    return min(s, key=lambda k: (median(s[k]), k != ANCHOR, k))


def _decision(rows, nominee):
    s = _scores(rows, {ANCHOR, nominee}, 7)
    r = [x / y for x, y in zip(s[ANCHOR], s[nominee], strict=True)]
    wins = sum(v > 1 for v in r)
    m = median(r)
    return dict(
        selected=nominee if wins >= 6 and m >= 1.03 else ANCHOR, wins=wins, median_ratio=m, ratios=r
    )


def _validate(receipt):
    if set(receipt) != {"body", "sha256"} or _hash(receipt["body"]) != receipt["sha256"]:
        raise ValueError("invalid receipt checksum")
    b = receipt["body"]
    arms = b["admitted"]
    nom = b["nominee"]
    if (
        b["context"]["schema"] != SCHEMA
        or b["context"]["objective"] not in OBJECTIVES
        or ANCHOR not in arms
        or not set(arms) <= set(ARMS)
        or len(set(arms)) != len(arms)
        or set(arms) & set(b["rejections"])
        or set(arms) | set(b["rejections"]) != set(ARMS)
        or nom != _nominate(b["discovery"], arms)
        or b["decision"] != _decision(b["confirmation"], nom)
    ):
        raise ValueError("inconsistent selection receipt")
    ids = b["requests"]
    flat = [x for v in ids.values() for x in v]
    if (
        {k: len(v) for k, v in ids.items()}
        != {"capture": 2, "discovery": 8, "confirmation": 8, "audit": 8}
        or len(set(flat)) != 26
        or any(
            not isinstance(x, str) or len(x) != 64 or any(c not in "0123456789abcdef" for c in x)
            for x in flat
        )
    ):
        raise ValueError("invalid selection cohorts")
    _scores(b["audit"], arms, 7)
    if set(b["gates"]) != set(arms) or any(
        set(row) != {"discovery", "confirmation", "audit"}
        or any(len(v) != 8 or any(type(t) is not int or t < 1 for t in v) for v in row.values())
        for row in b["gates"].values()
    ):
        raise ValueError("invalid request gates")
    return b


@torch.inference_mode()
def select_request_policy(
    ep, example, check, discovery, confirmation, audit, *, objective, seed=173, receipt_path=None
):
    """Return (owned executable, receipt). Exclusive setup; fixed graph/state, 26 distinct requests.

    Clock includes native input copy, graph launch, fresh outputs and stream completion. The audit
    set can fail the entire qualification, but never nominates, promotes, or retries an arm.
    Caller must not mutate state/requests during setup. Close the returned handle on its owner stream.
    """
    start = time.perf_counter_ns()
    ctx, state = _context(ep, example, objective)
    ids = _requests(
        ctx,
        dict(capture=[example, check], discovery=discovery, confirmation=confirmation, audit=audit),
        (2, 8, 8, 8),
        state,
    )
    raw = ep.module()
    groups = {"discovery": discovery, "confirmation": confirmation}
    refs = {k: [raw(*r) for r in rows] for k, rows in groups.items()}
    stream = torch.cuda.current_stream()
    base = torch.cuda.memory_allocated()
    handles = {}
    gates = {}
    builds = {}
    reject = {}
    owners = {}
    with ExitStack() as cleanup:
        for name in ARMS:
            own = cleanup.enter_context(ExitStack())
            t = time.perf_counter_ns()
            before = torch.cuda.memory_allocated()
            try:
                h = own.enter_context(_capture(ep, example, check, name))
                kinds = _kinds(h, name)
                stream.synchronize()
                builds[name] = dict(
                    ms=(time.perf_counter_ns() - t) / 1e6,
                    allocator_delta_bytes=torch.cuda.memory_allocated() - before,
                    kinds=kinds,
                )
                gates[name] = {k: _gate(h, rows, refs[k], state) for k, rows in groups.items()}
            except (Unsupported, SiteMismatch) as e:
                if name == ANCHOR:
                    raise
                own.close()
                reject[name] = type(e).__name__ + ": " + str(e)
                continue
            handles[name] = h
            owners[name] = own
        d = _windows(handles, discovery, objective, seed, 5)
        nom = _nominate(d, handles)
        c = _windows({k: handles[k] for k in (ANCHOR, nom)}, confirmation, objective, seed + 1, 7)
        decision = _decision(c, nom)
        selected = decision["selected"]
        refs_a = [raw(*r) for r in audit]
        for k, h in handles.items():
            gates[k]["audit"] = _gate(h, audit, refs_a, state)
        ha = _windows(handles, audit, objective, seed + 2, 7)
        del refs_a
        for k in handles:
            if k != selected:
                owners[k].close()
        stream.synchronize()
        body = dict(
            context=ctx,
            requests=ids,
            seed=seed,
            admitted=list(handles),
            rejections=reject,
            builds=builds,
            gates=gates,
            discovery=d,
            nominee=nom,
            confirmation=c,
            decision=decision,
            audit=ha,
            setup_ms=(time.perf_counter_ns() - start) / 1e6,
            allocator_delta_after_loser_close_bytes=torch.cuda.memory_allocated() - base,
        )
        rec = dict(body=body, sha256=_hash(body))
        _validate(rec)
        if receipt_path is not None:
            _write_once(receipt_path, rec)
        owners[selected].pop_all()
        return handles[selected], rec


@torch.inference_mode()
def replay_request_policy(ep, example, check, receipt, *, objective):
    """Validate a trusted local receipt and two unseen requests, capture only its selected arm."""
    if isinstance(receipt, (str, Path)):
        receipt = json.loads(Path(receipt).read_bytes())
    b = _validate(receipt)
    ctx, state = _context(ep, example, objective)
    if ctx != b["context"]:
        raise ValueError("request policy context mismatch")
    _requests(
        ctx,
        {"capture": [example, check]},
        (2,),
        state,
        [h for v in b["requests"].values() for h in v],
    )
    name = b["decision"]["selected"]
    with ExitStack() as own:
        h = own.enter_context(_capture(ep, example, check, name))
        _kinds(h, name)
        raw = ep.module()
        rows = [example, check]
        _gate(h, rows, [raw(*r) for r in rows], state)
        own.pop_all()
        return h
