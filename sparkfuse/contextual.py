"""Opt-in whole-graph timing of one static site; no installation in the shared IR cache."""

from dataclasses import asdict
from hashlib import sha256
import json
import math
import operator
import os
from pathlib import Path
import random
from statistics import median
from tempfile import NamedTemporaryFile

import torch
from torch.utils import _pytree as tree

from .codegen import Schedule
from .ir import Unsupported
from .measure import device_time
from .partition import _RegionModule, partition_export
from .runtime import Kernel, compile_export
from .tuning import signature


class SiteMismatch(ValueError):
    """A visible value or structural guard differs; never suppress operational errors."""


def _hash(obj):
    return sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _bytes(t):
    return t.detach().contiguous().reshape(-1).view(torch.uint8)


def _value(t):
    return sha256(memoryview(_bytes(t).cpu().numpy())).hexdigest()


def _layout(t):
    return [str(t.device), str(t.dtype), list(t.shape), list(t.stride())]


def _exact(got, ref, what):
    xs, a = tree.tree_flatten(got)
    ys, b = tree.tree_flatten(ref)
    if a != b or not ys:
        raise SiteMismatch(what + ": output tree")
    for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise Unsupported("site validation requires tensor leaves")
        if _layout(x) != _layout(y):
            raise SiteMismatch(f"{what}[{i}]: device, dtype, shape or stride")
        if not torch.equal(_bytes(x), _bytes(y)):
            raise SiteMismatch(f"{what}[{i}]: unequal bits")
    return len(ys)


def _describe(args):
    if (
        not isinstance(args, tuple)
        or not args
        or not all(isinstance(x, torch.Tensor) for x in args)
    ):
        raise TypeError("a nonempty positional tuple of tensors is required")
    dev = args[0].device
    if dev.type != "cuda" or any(
        x.device != dev or x.layout != torch.strided or x.numel() == 0 for x in args
    ):
        raise Unsupported("expected nonempty strided inputs on one CUDA device")
    aliases = {}
    return [
        {
            "layout": _layout(x),
            "offset": x.storage_offset(),
            "align32": x.data_ptr() % 32,
            "storage": aliases.setdefault(x.untyped_storage().data_ptr(), len(aliases)),
        }
        for x in args
    ]


def _key(s):
    return f"{s.layout}:{s.threads}:{int(s.cache)}:{s.vec}"


def _open(ep, name, dev, options):
    if ep.range_constraints:
        raise Unsupported("context selection currently requires a static export")

    def uses_rng(n):
        if torch.Tag.nondeterministic_seeded not in getattr(n.target, "tags", ()):
            return False
        # Export retains aten.dropout even in eval, where it is an identity.
        if n.target == torch.ops.aten.dropout.default and len(n.args) == 3 and n.args[2] is False:
            return False
        return True

    if torch.is_autocast_enabled("cuda") or any(uses_rng(n) for n in ep.graph.nodes):
        raise Unsupported("autocast and random operators are outside deterministic replay")
    gm, recs = partition_export(ep, device=dev, **options)
    matches = [r for r in recs if r.get("module") == name]
    if len(matches) != 1 or not isinstance(gm.get_submodule(name), _RegionModule):
        raise Unsupported("expected a named ordinary generated region")
    rec = matches[0]
    site = gm.get_submodule(name)
    if not isinstance(site.kernel, Kernel):
        raise Unsupported("site is not an ordinary generated kernel")
    raw = ep.module()
    original = {n.name: n for n in raw.graph.nodes}
    calls = [n for n in gm.graph.nodes if n.op == "call_module" and n.target == name]
    if len(calls) != 1 or calls[0].kwargs:
        raise Unsupported("expected one positional site invocation")
    producers = {r["module"]: r for r in recs if "module" in r}

    def resolve(n):
        if (
            n.op == "call_function"
            and n.target is operator.getitem
            and n.args[0].op == "call_module"
        ):
            outs = producers.get(n.args[0].target, {}).get("live_outs", [])
            idx = n.args[1]
            if type(idx) is int and 0 <= idx < len(outs) and outs[idx] in original:
                return outs[idx]
            raise Unsupported("cannot trace producer output to original ATen graph")
        if n.name in original and (n.op, n.target) == (
            original[n.name].op,
            original[n.name].target,
        ):
            return n.name
        raise Unsupported("cannot trace site input to original ATen graph")

    edges = [[resolve(n) for n in calls[0].args], rec["live_outs"]]
    if not set(edges[1]) <= original.keys():
        raise Unsupported("cannot trace site output to original ATen graph")
    return gm, raw, recs, rec, site, edges


def _stamp(ep, gm, recs, rec, site, args):
    root = Path(__file__).parent
    files = sorted([*root.glob("*.py"), *(root / "csrc").glob("*.cpp")])
    state = {}
    aliases = {}
    request_storages = {x.untyped_storage().data_ptr() for x in args}
    driver = Path("/proc/driver/nvidia/version")
    for prefix, vals in [("state", ep.state_dict), ("constants", ep.constants)]:
        for n, t in vals.items():
            if not isinstance(t, torch.Tensor):
                raise Unsupported("context state must be tensor-only")
            storage = t.untyped_storage().data_ptr()
            if storage in request_storages:
                raise Unsupported("input aliases model state")
            state[prefix + ":" + n] = [
                _layout(t),
                t.storage_offset(),
                t.data_ptr() % 32,
                aliases.setdefault(storage, len(aliases)),
                _value(t),
            ]
    return dict(
        site=rec["module"],
        driver_fingerprint=sha256(driver.read_bytes()).hexdigest() if driver.is_file() else None,
        device_fingerprint=_hash(
            str(getattr(torch.cuda.get_device_properties(args[0].device), "uuid", None))
        ),
        torch_git=torch.version.git_version,
        region_signature=signature(site.kernel.plan, args[0].device),
        project=_hash(
            {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest() for p in files}
        ),
        export=_hash(str(ep.graph)),
        mixed=_hash(str(gm.graph)),
        records=_hash(recs),
        call_spec=_hash(
            [tree.treespec_dumps(ep.call_spec.in_spec), tree.treespec_dumps(ep.call_spec.out_spec)]
        ),
        state=_hash(state),
        inputs=_describe(args),
        flags=dict(
            matmul=torch.get_float32_matmul_precision(),
            fp16_reduce=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            bf16_reduce=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            deterministic=torch.are_deterministic_algorithms_enabled(),
            workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        ),
    )


def _oracle(raw, edges, args):
    found = {}
    want = set(edges[0] + edges[1])

    class Tap(torch.fx.Interpreter):
        def run_node(self, node):
            val = super().run_node(node)
            if node.name in want:
                found[node.name] = val
            return val

    whole = Tap(raw).run(*args)
    return tuple(found[n] for n in edges[0]), tuple(found[n] for n in edges[1]), whole


def _alias_pattern(*values):
    # Only relationships are observable here. Never bind a fresh output allocation's address.
    tensors = tree.tree_leaves(values)
    if not all(isinstance(x, torch.Tensor) and x.numel() for x in tensors):
        raise Unsupported("alias checks require nonempty tensor leaves")
    seen = {}
    pattern = []
    for x in tensors:
        ptr = x.untyped_storage().data_ptr()
        offset = x.storage_offset() * x.element_size()
        group, base = seen.setdefault((str(x.device), ptr), (len(seen), offset))
        pattern.append((group, offset - base))
    return pattern


def _check(gm, name, mod, args, oracle):
    setattr(gm, name, mod)
    ins, outs, whole = oracle
    wires = []

    def see(_module, a, b):
        wires.append((_exact(a, ins, "live-ins"), _exact(b, outs, "live-outs")))
        if _alias_pattern(a, b) != _alias_pattern(ins, outs):
            raise SiteMismatch("region boundary alias mismatch")

    handle = mod.register_forward_hook(see)
    try:
        y = gm(*args)
    finally:
        handle.remove()
    if len(wires) != 1:
        raise Unsupported("expected one observed site invocation")
    if _alias_pattern(args, y) != _alias_pattern(args, whole):
        raise SiteMismatch("composed output alias mismatch")
    return dict(
        live_ins=wires[0][0], live_outs=wires[0][1], whole=_exact(y, whole, "composed output")
    )


def _windows(gm, name, mods, args, seed, n):
    rng = random.Random(seed)
    windows = []
    for _ in range(n):
        order = sorted(mods)
        rng.shuffle(order)
        samples = {}
        for k in order:
            setattr(gm, name, mods[k])
            samples[k] = device_time(lambda: gm(*args), groups=7, unroll=3, replay=4)["samples_us"]
        windows.append(dict(order=order, samples_us=samples))
    return windows


def _med(w, k):
    xs = w["samples_us"][k]
    if len(xs) != 7 or any(not math.isfinite(x) or x <= 0 for x in xs):
        raise ValueError("invalid timing samples")
    return median(xs)


def _nominate(windows, anchor):
    # Break exact ties in favor of the incumbent, then by stable schedule ID.
    return min(
        windows[0]["order"], key=lambda k: (median(_med(w, k) for w in windows), k != anchor, k)
    )


def _decide(windows, anchor, challenger, min_gain):
    if len(windows) != 7 or not math.isfinite(min_gain) or min_gain <= 1:
        raise ValueError("seven confirmation windows and gain > 1 required")
    ratios = [_med(w, anchor) / _med(w, challenger) for w in windows]
    wins = sum(x > 1 for x in ratios)
    return dict(
        ratios=ratios,
        wins=wins,
        median_ratio=median(ratios),
        min_gain=min_gain,
        selected=challenger if wins >= 6 and median(ratios) >= min_gain else anchor,
    )


def _write_once(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        try:
            f.write((json.dumps(obj, sort_keys=True, indent=1, allow_nan=False) + "\n").encode())
            f.flush()
            os.fsync(f.fileno())
            os.link(tmp, path)  # No overwrite, including competing writers.
        finally:
            tmp.unlink(missing_ok=True)


@torch.inference_mode()
def select_site(
    ep,
    name,
    schedules,
    train,
    confirm,
    audit,
    *,
    partition_options=None,
    seed=2917,
    min_gain=1.02,
    path=None,
):
    """Measure one ordinary site inside its composed graph. Five discovery + seven confirmation windows.

    Three distinct, same-layout tensor requests; deterministic, effect-free, hook-free inference only.
    Returns a new mixed graph and JSON-serializable receipt. Never modifies a dispatch database.
    """
    layouts = [_describe(a) for a in (train, confirm, audit)]
    if not (layouts[0] == layouts[1] == layouts[2]) or not math.isfinite(min_gain) or min_gain <= 1:
        raise ValueError("same request layouts and finite minimum gain > 1 required")
    if not (1 <= len(schedules) <= 16) or not all(isinstance(s, Schedule) for s in schedules):
        raise ValueError("supply 1..16 explicit schedules")
    hashes = [_hash([_value(x) for x in a]) for a in (train, confirm, audit)]
    if len(set(hashes)) != 3:
        raise ValueError("three content-distinct requests required")
    options = dict(partition_options or {})
    dev = train[0].device
    with torch.cuda.device(dev):
        gm, raw, recs, rec, base, edges = _open(ep, name, dev, options)
        stamp = _stamp(ep, gm, recs, rec, base, train)
        anchor = _key(base.kernel.schedule)
        mods = {anchor: base}
        gates = {}
        rejected = {}
        oracle = _oracle(raw, edges, train)
        gates["train"] = {anchor: _check(gm, name, base, train, oracle)}
        for s in dict.fromkeys(schedules):
            k = _key(s)
            if k == anchor:
                continue
            try:
                s.validate(base.kernel.plan)
            except Unsupported as e:
                rejected[k] = dict(phase="legality", reason=str(e))
                continue
            mod = _RegionModule(compile_export(base.kernel.ep, dev, s), rec["host_backend"])
            try:
                gates["train"][k] = _check(gm, name, mod, train, oracle)
            except SiteMismatch as e:
                rejected[k] = dict(phase="train", reason=str(e))
                continue
            mods[k] = mod
        del oracle
        discovery = _windows(gm, name, mods, train, seed, 5)
        nominee = _nominate(discovery, anchor)
        oracle = _oracle(raw, edges, confirm)
        gates["confirm"] = {anchor: _check(gm, name, base, confirm, oracle)}
        confirmation = []
        decision = dict(selected=anchor, reason="incumbent nominated")
        if nominee != anchor:
            try:
                gates["confirm"][nominee] = _check(gm, name, mods[nominee], confirm, oracle)
            except SiteMismatch as e:
                rejected[nominee] = dict(phase="confirm", reason=str(e))
                decision["reason"] = "nominee numerically rejected"
            else:
                del oracle
                confirmation = _windows(
                    gm, name, {k: mods[k] for k in (anchor, nominee)}, confirm, seed + 1, 7
                )
                decision = _decide(confirmation, anchor, nominee, min_gain)
        oracle = _oracle(raw, edges, audit)
        chosen = decision["selected"]
        gates["audit"] = {}
        for k in dict.fromkeys([anchor, chosen]):
            gates["audit"][k] = _check(gm, name, mods[k], audit, oracle)
        receipt = dict(
            schema=1,
            stamp=stamp,
            edges=edges,
            seed=seed,
            min_gain=min_gain,
            request_hashes=hashes,
            anchor=anchor,
            candidates=[asdict(s) for s in schedules],
            rejected=rejected,
            discovery=discovery,
            nominee=nominee,
            confirmation=confirmation,
            decision=decision,
            gates=gates,
            selected=asdict(mods[chosen].kernel.schedule),
            cuda_sha256=sha256(mods[chosen].kernel.source.encode()).hexdigest(),
        )
        receipt["checksum"] = _hash(receipt)
        if path is not None:
            _write_once(path, receipt)
        return gm, receipt


@torch.inference_mode()
def replay_site(ep, receipt, request, *, partition_options=None):
    """Repartition, verify context + source, and recheck the actual wires and full output on fresh data."""
    if receipt.get("schema") != 1 or receipt.get("checksum") != _hash(
        {k: v for k, v in receipt.items() if k != "checksum"}
    ):
        raise ValueError("invalid context receipt")
    _describe(request)
    if _hash([_value(x) for x in request]) in receipt["request_hashes"]:
        raise ValueError("fresh replay request required")
    dev = request[0].device
    name = receipt["stamp"]["site"]
    with torch.cuda.device(dev):
        gm, raw, recs, rec, base, edges = _open(ep, name, dev, dict(partition_options or {}))
        if _stamp(ep, gm, recs, rec, base, request) != receipt["stamp"]:
            raise SiteMismatch("context fingerprint changed")
        s = Schedule(**receipt["selected"])
        mod = (
            base
            if s == base.kernel.schedule
            else _RegionModule(compile_export(base.kernel.ep, dev, s), rec["host_backend"])
        )
        if (
            _key(s) != receipt["decision"]["selected"]
            or sha256(mod.kernel.source.encode()).hexdigest() != receipt["cuda_sha256"]
        ):
            raise SiteMismatch("selected schedule or CUDA source changed")
        checks = _check(gm, name, mod, request, _oracle(raw, edges, request))
        return gm, checks
