"""Specialize only symbol-independent regions in an already qualified dynamic decode graph."""

from dataclasses import dataclass, field
import operator
from collections import deque
import torch
from torch import fx
from torch.utils import _pytree as tree
from .decode_attention import _require
from .decode_sites import _signature
from .ir import DT, Unsupported
from .partition import eligible, _region, _RegionModule, _InsertionBarrier, _costed_slices
from .runtime import compile_export
from .symbolic_sites import partition_symbolic_decode


@dataclass
class HybridDecode:
    module: fx.GraphModule
    attention: list
    regions: list
    host_calls: list
    rotary: list = field(default_factory=list)
    appends: list = field(default_factory=list)
    protection: dict = field(default_factory=dict)


def _static(n):
    t = n.meta.get("val")
    return (
        isinstance(t, torch.Tensor)
        and t.layout == torch.strided
        and t.device.type == "cuda"
        and t.dtype in DT
        and not t.is_neg()
        and not t.is_conj()
        and all(type(v) is int and v > 0 for v in (*t.shape, *t.stride()))
    )


def _components(gm, excluded=()):
    pos = {n: i for i, n in enumerate(gm.graph.nodes)}
    pot = set()
    for n in pos:
        if n in excluded or not eligible(n) or not all(_static(v) for v in n.all_input_nodes):
            continue
        # Assertions have no tensor output; lower() proves the literal dtype/device/layout
        # against the static boundary. Never fold dynamic export GuardsFn or shape arithmetic.
        if _static(n) or n.target is torch.ops.aten._assert_tensor_metadata.default:
            pot.add(n)
    while pot:
        first = min(pot, key=pos.get)
        stack = [first]
        pot.remove(first)
        component = []
        while stack:
            n = stack.pop()
            component.append(n)
            for u in (*n.all_input_nodes, *n.users):
                if u in pot:
                    pot.remove(u)
                    stack.append(u)
        g = sorted(component, key=pos.get)
        if sum(v.target is not torch.ops.aten._assert_tensor_metadata.default for v in g) >= 2:
            yield g


def _bits(x, y):
    return (
        x.shape == y.shape
        and x.dtype == y.dtype
        and torch.equal(
            x.contiguous().reshape(-1).view(torch.uint8),
            y.contiguous().reshape(-1).view(torch.uint8),
        )
    )


def _check_fragment(rep, ins, outs, traces, device, host):
    kernel = compile_export(rep, device)
    region = _RegionModule(kernel, host)
    for env in traces:
        args = tuple(env[v] for v in ins)
        ref = tuple(env[v] for v in outs)
        got = region(*args)
        if _signature(got, args) != _signature(ref, args) or any(
            (a._base is None) != (b._base is None) or not _bits(a, b)
            for a, b in zip(got, ref, strict=True)
        ):
            raise Unsupported("not byte/ownership equivalent at qualification boundary")
    return region


def _trace(gm, batches):
    traces = []
    answers = []
    for xs in batches:
        interp = fx.Interpreter(gm, garbage_collect_values=False)
        answers.append(interp.run(*xs))
        traces.append(interp.env)
    return traces, answers


def partition_hybrid_decode(
    ep,
    example,
    audit,
    *,
    domain="checked",
    host_backend="native",
    host_calls=False,
    split_barriers=True,
    paired_rotary=False,
    paired_append=False,
    exact_outputs=(),
    tile=32,
    warp_heads=0,
    max_workspace_bytes=32 << 20,
):
    """Retain a single dynamic attention program; test each static CUDA island independently.

    Qualification is sampled, not a proof for unseen values. Storage geometry and bytes of
    every escaping island result must agree on *both* independent qualification requests.
    A rejected island remains ATen. No weight freezing or shape specialization on a trace.
    """
    _require(
        type(host_calls) is bool
        and type(split_barriers) is bool
        and type(paired_rotary) is bool
        and type(paired_append) is bool
        and (not paired_append or host_backend == "native"),
        "explicit Boolean options required",
    )
    p = partition_symbolic_decode(
        ep,
        example,
        audit,
        domain=domain,
        host_backend=host_backend,
        exact_outputs=exact_outputs,
        tile=tile,
        warp_heads=warp_heads,
        max_workspace_bytes=max_workspace_bytes,
    )
    gm = p.module
    from .observable import OutputSlice

    locked = OutputSlice.capture(gm, exact_outputs)
    audit_slice = locked.describe(p.skipped)
    if not p.sites:
        return HybridDecode(gm, [], [], [], protection=audit_slice)
    batches = tuple(map(tuple, (example, audit)))
    traces, refs = _trace(gm, batches)
    rotary = []
    if paired_rotary:
        from .rotary import fuse_pairs

        def qualify(region, q, k):
            if locked.nodes & (q.members | k.members):
                return False
            for env in traces:
                if any(n not in env for n in (q.x, k.x, q.c, q.s, q.out, k.out)):
                    return False
                qs, ks = env[q.x], env[k.x]
                args = (
                    qs.transpose(1, 2) if q.bshd else qs,
                    ks.transpose(1, 2) if k.bshd else ks,
                    env[q.c],
                    env[q.s],
                )
                out = region(*args)
                got = tuple(
                    y.transpose(1, 2) if b.bshd else y for y, b in zip(out, (q, k), strict=True)
                )
                ref = (env[q.out], env[k.out])
                if _signature(got, args) != _signature(ref, args) or any(
                    (a._base is None) != (b._base is None) or not _bits(a, b)
                    for a, b in zip(got, ref, strict=True)
                ):
                    return False
            return True

        rotary = fuse_pairs(
            gm,
            example[0].device,
            "native" if host_backend == "native" else "python",
            qualify=qualify,
        )
        if any("module" in r for r in rotary):
            traces, _ = _trace(gm, batches)
    appends = []
    if paired_append:
        from .dynamic_append import fuse_dynamic_appends

        appends = fuse_dynamic_appends(gm, p.sites, traces, excluded=locked.nodes)
        if any("module" in r for r in appends):
            traces, _ = _trace(gm, batches)
    records = []
    proposals = []
    pos = {n: i for i, n in enumerate(gm.graph.nodes)}
    queue = deque((g, None) for g in _components(gm, locked.nodes))
    while queue:
        group, parent = queue.popleft()
        rec = dict(ops=[n.name for n in group])
        records.append(rec)
        if parent is not None:
            rec["split_from"] = parent
        try:
            rep, ins, outs, anchor = _region(group, pos)
            rec.update(live_ins=[v.name for v in ins], live_outs=[v.name for v in outs])
            region = _check_fragment(
                rep,
                ins,
                outs,
                traces,
                example[0].device,
                "native" if host_backend == "native" else "python",
            )
        except Unsupported as e:
            rec["fallback"] = str(e)
            if split_barriers and parent is None and isinstance(e, _InsertionBarrier):
                split = _costed_slices(group, pos, 2, allow_single=True)
                if split:
                    blocks, _ = split
                    tag = rec["ops"]
                    rec["split_parts"] = [[x.name for x in group[i:j]] for i, j, f in blocks if f]
                    queue.extend((group[i:j], tag) for i, j, f in blocks if f)
            continue
        proposals.append((group, ins, outs, anchor, region, rec))
    # Proposals were built from the same unmodified graph. Insert all calls before erasing
    # any nodes: a later island's original insertion anchor can belong to an earlier island.
    patches = []
    for i, (group, ins, outs, anchor, region, rec) in enumerate(proposals):
        label = f"_sparkfuse_static_island_{i}"
        _require(not hasattr(gm, label), "static island name collision")
        gm.add_module(label, region)
        with gm.graph.inserting_before(anchor):
            call = gm.graph.call_module(label, tuple(ins))
            items = []
            for j, old in enumerate(outs):
                new = gm.graph.call_function(operator.getitem, (call, j))
                new.meta = dict(old.meta)
                items.append(new)
        patches.extend((old, new, group) for old, new in zip(outs, items, strict=True))
        k = region.kernel
        rec.update(
            module=label,
            output_shapes=[list(k.plan.vals[o].shape) for o in k.plan.outputs],
            schedule=vars(k.schedule),
            qualification="two independent byte-exact traces",
        )
    for old, new, owners in patches:
        for user in list(old.users):
            if user not in owners:
                user.replace_input_with(old, new)
    for group, *_ in proposals:
        for n in reversed(group):
            gm.graph.erase_node(n)
    special = []
    if host_calls:
        from .host_calls import specialize_host_calls

        special = specialize_host_calls(gm, excluded=locked.nodes)
    gm.graph.lint()
    gm.recompile()
    for xs, ref in zip(batches, refs, strict=True):
        got = gm(*xs)
        _require(
            _signature(got, xs) == _signature(ref, xs)
            and all(
                (a._base is None) == (b._base is None) and _bits(a, b)
                for a, b in zip(tree.tree_leaves(got), tree.tree_leaves(ref), strict=True)
            ),
            "composition changed a public attention-only result",
        )
    locked.verify(gm)
    return HybridDecode(gm, p.sites, records, special, rotary, appends, audit_slice)
