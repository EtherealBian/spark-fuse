"""Partition a functional exported graph; unsupported ops remain in PyTorch."""

from dataclasses import dataclass
from collections import deque
from math import prod
import operator
from types import SimpleNamespace
from torch import fx, nn
from torch.utils import _pytree as tree
from .ir import DT, Unsupported, lower, ordered_score_limit
from .runtime import compile_export
from .codegen import Schedule
from .tuning import compile_tuned

CALLS = frozenset(
    (
        "add.Tensor",
        "mul.Tensor",
        "sub.Tensor",
        "div.Tensor",
        "add.Scalar",
        "mul.Scalar",
        "sub.Scalar",
        "silu.default",
        "gelu.default",
        "rsqrt.default",
        "relu.default",
        "neg.default",
        "sigmoid.default",
        "tanh.default",
        "square.default",
        "pow.Tensor_Scalar",
        "to.dtype",
        "_to_copy.default",
        "to.dtype_layout",
        "rms_norm.default",
        "layer_norm.default",
        "mean.dim",
        "softmax.int",
        "_softmax.default",
        "_assert_tensor_metadata.default",
    )
)


def eligible(n):
    return n.op == "call_function" and (
        str(n.target).removeprefix("aten.") in CALLS
        or str(n.target) == "prims.convert_element_type.default"
    )


@dataclass
class RegionSpec:
    graph: fx.Graph
    graph_signature: object
    call_spec: object
    state_dict: dict
    constants: dict


class _InsertionBarrier(Unsupported):
    """A fallback consumer needs a result before the entire component can execute."""


def _region(members, pos):
    active = set(members)
    incoming = sorted(
        {n for u in members for n in u.all_input_nodes if n not in active}, key=pos.get
    )
    outgoing = [n for n in members if any(u not in active for u in n.users)]
    consumers = {u for n in outgoing for u in n.users if u not in active}
    if not incoming or not outgoing:
        raise Unsupported("empty boundary")
    anchor = min(consumers, key=pos.get)
    if max(pos[n] for n in incoming) >= pos[anchor]:
        raise _InsertionBarrier("no legal insertion point between live-ins and live-outs")
    if any(
        str(n.target)
        in (
            "aten.to.dtype",
            "aten._to_copy.default",
            "aten.to.dtype_layout",
            "prims.convert_element_type.default",
        )
        and n.meta["val"].dtype == n.args[0].meta["val"].dtype
        for n in outgoing
    ):
        raise Unsupported("possible alias escaping region")
    g = fx.Graph()
    remap = {}
    for j, n in enumerate(incoming):
        v = g.placeholder(f"arg{j}")
        v.meta = dict(n.meta)
        remap[n] = v
    for n in members:
        remap[n] = g.node_copy(n, lambda old: remap[old])
    g.output(tuple(remap[n] for n in outgoing))

    def kind(tag):
        return SimpleNamespace(kind=SimpleNamespace(name=tag))

    sig = SimpleNamespace(
        input_specs=[kind("USER_INPUT") for _ in incoming],
        output_specs=[kind("USER_OUTPUT") for _ in outgoing],
    )
    _, inspec = tree.tree_flatten((tuple(range(len(incoming))), {}))
    _, outspec = tree.tree_flatten(tuple(range(len(outgoing))))
    ep = RegionSpec(g, sig, SimpleNamespace(in_spec=inspec, out_spec=outspec), {}, {})
    return ep, incoming, outgoing, anchor


_COLLECTIVES = frozenset(
    (
        "aten.mean.dim",
        "aten.rms_norm.default",
        "aten.layer_norm.default",
        "aten.softmax.int",
        "aten._softmax.default",
    )
)


def _costed_slices(group, pos, min_ops, allow_single=False):
    """Bounded contiguous topological intervals. This is a structural objective, not a latency model."""
    n = len(group)
    collectives = sum(str(v.target) in _COLLECTIVES for v in group)
    if n > 48 or collectives < (1 if allow_single else 2):
        return None

    def connected(nodes):
        active = set(nodes)
        todo = [nodes[0]]
        seen = {nodes[0]}
        while todo:
            v = todo.pop()
            for u in (*v.all_input_nodes, *v.users):
                if u in active and u not in seen:
                    seen.add(u)
                    todo.append(u)
        return len(seen) == len(nodes)

    # Lexicographic objective: cover most operators, fewer GPU launches, then fewer
    # logical boundary bytes. An unsupported singleton can stay in PyTorch.
    best = [None] * (n + 1)
    best[n] = ((0, 0, 0), [])
    for i in reversed(range(n)):
        score, tail = best[i + 1]
        best[i] = (score, [(i, i + 1, False)] + tail)
        reds = count = 0
        for j in range(i + 1, n + 1):
            op = str(group[j - 1].target)
            reds += op in _COLLECTIVES
            count += op != "aten._assert_tensor_metadata.default"
            if reds > 1:
                break
            if count < min_ops or not connected(group[i:j]):
                continue
            try:
                rep, *_ = _region(group[i:j], pos)
                plan = lower(rep)
            except Unsupported:
                continue
            written = sum(
                prod(plan.vals[o].shape) * (4 if plan.vals[o].dt == "f32" else 2)
                for o in plan.outputs
            )
            suffix, blocks = best[j]
            key = (count + suffix[0], suffix[1] - 1, suffix[2] - written)
            if key > best[i][0]:
                best[i] = (key, [(i, j, True)] + blocks)
    score, blocks = best[0]
    return (blocks, score) if score[0] else None


class _RegionModule(nn.Module):
    def __init__(self, k, host_backend):
        super().__init__()
        self.kernel = k
        if host_backend == "native":
            from .native import native_dispatch

            self.dispatch = native_dispatch(k)
        else:
            self.dispatch = k

    def forward(self, *args):
        return self.dispatch(*args)


def ordered_collectives(plan):
    """Optional fixed reduction orders for supported dense rows; other regions use the existing path."""
    from .mean_tree import eligible, ordered_default

    if eligible(plan):
        return ordered_default(plan)
    if plan.red is not None:
        kind = plan.vals[plan.red].op
        if kind == "layer_norm" and plan.width % 4 == 0:
            return Schedule(128, "block_welford", plan.width <= 4096, 4)
        if kind == "softmax" and 32 <= plan.width <= 1024:
            return Schedule(128, "warp_xor", True)
    return None


def partition_export(
    ep,
    device="cuda",
    database=None,
    min_ops=2,
    host_backend="python",
    split_collectives=True,
    schedule_policy=None,
    split_barriers=True,
    rotary_pairs=False,
    paired_appends=False,
    paired_head_repeats=False,
    framework_softmax=False,
    ordered_scores=None,
):
    """Functional inference only. Return (mixed FX module, auditable region decisions)."""
    if ordered_scores not in (None, "loop", "slots", "guarded") or (
        framework_softmax and ordered_scores is not None
    ):
        raise ValueError(
            "choose one softmax policy: framework_softmax or ordered_scores='loop'/'slots'/'guarded'"
        )
    if host_backend not in ("python", "native"):
        raise ValueError("unknown host backend")
    if any(s.kind.name != "USER_OUTPUT" for s in ep.graph_signature.output_specs):
        raise Unsupported("graph with externally observable effects")
    gm = ep.module()
    for n in gm.graph.nodes:
        schema = getattr(n.target, "_schema", None)
        if schema is not None and schema.is_mutable:
            raise Unsupported("requires functional graph without mutations")
    records = []
    if rotary_pairs:
        from .rotary import fuse_pairs

        records.extend(fuse_pairs(gm, device, host_backend))
    if paired_appends:
        from .cache_append import fuse_appends

        records.extend(fuse_appends(gm, device, host_backend))
    if paired_head_repeats:
        from .head_repeat import fuse_head_repeats

        records.extend(fuse_head_repeats(gm, device, host_backend))
    nodes = list(gm.graph.nodes)
    pos = {n: i for i, n in enumerate(nodes)}
    pot = {n for n in nodes if eligible(n)}
    if framework_softmax or ordered_scores is not None:
        for n in nodes:
            if n not in pot or str(n.target) not in ("aten.softmax.int", "aten._softmax.default"):
                continue
            shape = getattr(n.meta.get("val"), "shape", ())
            last_axis = (
                len(n.args) >= 2 and type(n.args[1]) is int and n.args[1] in (-1, len(shape) - 1)
            )
            arg0 = n.args[0] if n.args else n.kwargs.get("self")
            src_dt = DT.get(getattr(getattr(arg0, "meta", {}).get("val"), "dtype", None))
            dst_dt = DT.get(getattr(n.meta.get("val"), "dtype", None))
            half_to_float = src_dt == "f16" and dst_dt == "f32"
            limit = ordered_score_limit(dst_dt, half_to_float)
            if (
                not framework_softmax
                and shape
                and last_axis
                and type(shape[-1]) is int
                and src_dt is not None
                and dst_dt is not None
                and 32 <= shape[-1] <= limit
            ):
                continue
            pot.remove(n)
            records.append(
                dict(
                    kind="reference_boundary",
                    ops=[n.name],
                    target=str(n.target),
                    reason="framework_softmax"
                    if framework_softmax
                    else "outside_ordered_score_domain",
                    input_dtype=src_dt,
                    output_dtype=dst_dt,
                    ordered_width_limit=limit,
                    half_to_float=half_to_float,
                    fallback="preserve framework softmax",
                )
            )
    components = []
    while pot:
        first = min(pot, key=pos.get)
        todo = [first]
        comp = []
        pot.remove(first)
        while todo:
            v = todo.pop()
            comp.append(v)
            for u in (*v.all_input_nodes, *v.users):
                if u in pot:
                    todo.append(u)
                    pot.remove(u)
        components.append(sorted(comp, key=pos.get))
    pending = deque((str(ci), g, None, False) for ci, g in enumerate(components))
    while pending:
        ci, group, parent, skip = pending.popleft()
        rec = {"ops": [n.name for n in group]}
        if parent is not None:
            rec["split"] = parent
        records.append(rec)
        if skip:
            rec["fallback"] = "not included in a legal collective split"
            continue
        if sum(str(n.target) != "aten._assert_tensor_metadata.default" for n in group) < min_ops:
            rec["fallback"] = "below minimum region size"
            continue
        try:
            # Re-evaluate topological positions because earlier contractions can replace live-ins.
            positions = {n: i for i, n in enumerate(gm.graph.nodes)}
            rep, inputs, outputs, anchor = _region(group, positions)
            plan = lower(rep)
            if (
                ordered_scores is not None
                and plan.red is not None
                and plan.vals[plan.red].op == "softmax"
            ):
                from .score_rounding import only_bf16_probability

                layout = "warp_xor"
                if plan.width > 1024 and ordered_scores != "loop":
                    layout = (
                        "warp_bf16_guard"
                        if ordered_scores == "guarded" and only_bf16_probability(plan)
                        else "warp_slots"
                    )
                override = Schedule(128, layout, True)
            else:
                override = schedule_policy(plan) if schedule_policy is not None else None
            if override is not None:
                k = compile_export(rep, device, schedule=override)
            else:
                k = (
                    compile_tuned(rep, database, device)
                    if database
                    else compile_export(rep, device)
                )
        except Unsupported as e:
            selection = (
                _costed_slices(
                    group,
                    positions,
                    min_ops,
                    allow_single=split_barriers and isinstance(e, _InsertionBarrier),
                )
                if split_collectives and parent is None
                else None
            )
            if selection:
                blocks, score = selection
                records.pop()
                info = {
                    "component": ci,
                    "reason": "fallback_barrier"
                    if isinstance(e, _InsertionBarrier)
                    else "rejected_multi_collective",
                    "original_rejection": str(e),
                    "covered_ops": score[0],
                    "planned_regions": -score[1],
                    "logical_output_bytes": -score[2],
                }
                pending.extendleft(
                    reversed(
                        [
                            (f"{ci}_{k}", group[i:j], info, not fused)
                            for k, (i, j, fused) in enumerate(blocks)
                        ]
                    )
                )
                continue
            rec["fallback"] = str(e)
            continue
        label = f"_sparkfuse_region_{ci}"
        gm.add_module(label, _RegionModule(k, host_backend))
        with gm.graph.inserting_before(anchor):
            call = gm.graph.call_module(label, tuple(inputs))
            items = []
            for idx, old in enumerate(outputs):
                new = gm.graph.call_function(operator.getitem, (call, idx))
                new.meta = dict(old.meta)
                items.append(new)
        for old, new in zip(outputs, items):
            for user in list(old.users):
                if user not in group:
                    user.replace_input_with(old, new)
        for old in reversed(group):
            gm.graph.erase_node(old)
        rec.update(
            module=label,
            live_ins=[n.name for n in inputs],
            live_outs=[n.name for n in outputs],
            schedule=vars(k.schedule),
            host_backend=host_backend,
        )
    gm.graph.lint()
    gm.recompile()
    return gm, records
