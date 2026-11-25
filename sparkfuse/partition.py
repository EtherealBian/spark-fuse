"""Partition a functional exported graph; unsupported ops remain in PyTorch."""

from dataclasses import dataclass
import operator
from types import SimpleNamespace
import torch
from torch import fx, nn
from torch.utils import _pytree as tree
from .ir import Unsupported, lower
from .runtime import compile_export
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
        raise Unsupported("no legal insertion point between live-ins and live-outs")
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
    kind = lambda tag: SimpleNamespace(kind=SimpleNamespace(name=tag))
    sig = SimpleNamespace(
        input_specs=[kind("USER_INPUT") for _ in incoming],
        output_specs=[kind("USER_OUTPUT") for _ in outgoing],
    )
    _, inspec = tree.tree_flatten((tuple(range(len(incoming))), {}))
    _, outspec = tree.tree_flatten(tuple(range(len(outgoing))))
    ep = RegionSpec(g, sig, SimpleNamespace(in_spec=inspec, out_spec=outspec), {}, {})
    return ep, incoming, outgoing, anchor


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


def partition_export(ep, device="cuda", database=None, min_ops=2, host_backend="python"):
    """Functional inference only. Return (mixed FX module, auditable region decisions)."""
    if host_backend not in ("python", "native"):
        raise ValueError("unknown host backend")
    if any(s.kind.name != "USER_OUTPUT" for s in ep.graph_signature.output_specs):
        raise Unsupported("graph with externally observable effects")
    gm = ep.module()
    nodes = list(gm.graph.nodes)
    pos = {n: i for i, n in enumerate(nodes)}
    pot = {n for n in nodes if eligible(n)}
    for n in nodes:
        schema = getattr(n.target, "_schema", None)
        if schema is not None and schema.is_mutable:
            raise Unsupported("requires functional graph without mutations")
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
    records = []
    for ci, group in enumerate(components):
        rec = {"ops": [n.name for n in group]}
        records.append(rec)
        if sum(str(n.target) != "aten._assert_tensor_metadata.default" for n in group) < min_ops:
            rec["fallback"] = "below minimum region size"
            continue
        try:
            # Re-evaluate topological positions because earlier contractions can replace live-ins.
            positions = {n: i for i, n in enumerate(gm.graph.nodes)}
            rep, inputs, outputs, anchor = _region(group, positions)
            lower(rep)
            k = compile_tuned(rep, database, device) if database else compile_export(rep, device)
        except Unsupported as e:
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
