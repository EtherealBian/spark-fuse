"""Opt-in inference partitioning of observed static SDPA sites in pure ATen graphs."""

from dataclasses import asdict, dataclass
from math import prod
import torch
from torch.export._unlift import GuardsFn
from torch import fx
from torch.utils import _pytree as tree
from .decode_attention import _build_site, _require
from .head_repeat import _branch
from .ir import Unsupported

SDPA = torch.ops.aten.scaled_dot_product_attention.default
# Explicitly allow only side-effect-free surrounding ops. Validate before executing
# qualification, so rejecting a graph cannot mutate state, inputs or RNG.
_PURE = set(
    "_assert_tensor_metadata.default view_as.default to.dtype pow.Tensor_Scalar mean.dim add.Tensor rsqrt.default mul.Tensor linear.default view.default transpose.int unsqueeze.default slice.Tensor neg.default cat.default expand.default reshape.default alias.default silu.default clone.default detach.default relu.default".split()
)


def _options(n):
    schema = n.target._schema.arguments
    opt = {s.name: s.default_value for s in schema}
    opt.update({s.name: a for s, a in zip(schema, n.args)})
    opt.update(n.kwargs)
    _require(
        opt["dropout_p"] == 0.0 and opt["is_causal"] is False and type(opt["enable_gqa"]) is bool,
        "only noncausal inference SDPA with fixed flags",
    )
    ins = [opt[k] for k in ("query", "key", "value")]
    if opt["attn_mask"] is not None:
        ins.append(opt["attn_mask"])
    _require(all(isinstance(x, fx.Node) for x in ins), "expected live tensor operands")
    return ins, opt


class _Region(torch.nn.Module):
    def __init__(self, op, checked):
        super().__init__()
        self.op = op
        self.checked = checked

    def forward(self, *xs):
        return self.op(*xs) if self.checked else self.op.run_bounded(*xs)


@dataclass
class DecodePartition:
    module: fx.GraphModule
    sites: list
    skipped: list


def _signature(ys, xs):
    values, spec = tree.tree_flatten(ys)
    _require(all(isinstance(y, torch.Tensor) for y in values), "only tensor output leaves")

    def ptr(x):
        return x.untyped_storage().data_ptr()

    desc = [
        (y.shape, y.stride(), y.dtype, y.device, y.storage_offset(), y.untyped_storage().nbytes())
        for y in values
    ]
    aliases = [[ptr(y) == ptr(z) for z in (*xs, *values)] for y in values]
    return spec, desc, aliases


def partition_decode_sites(
    ep,
    example,
    audit,
    *,
    domain="checked",
    elide_kv_repeats=True,
    tile=32,
    head_group=1,
    max_workspace_bytes=32 << 20,
):
    """Keep unsupported sites in ATen. `bounded` requires valid values at every lowered site."""
    _require(
        torch.is_inference_mode_enabled() and not torch.is_autocast_enabled("cuda"),
        "inference mode without autocast required",
    )
    _require(
        domain in ("checked", "bounded") and type(elide_kv_repeats) is bool,
        "invalid partition mode",
    )
    batches = tuple(map(tuple, (example, audit)))
    n = len(batches[0])
    _require(
        n == len(batches[1])
        and all(type(x) is torch.Tensor and x.device.type == "cuda" for xs in batches for x in xs),
        "expected positional CUDA tensors",
    )
    _require(
        {x.untyped_storage().data_ptr() for x in batches[0]}.isdisjoint(
            x.untyped_storage().data_ptr() for x in batches[1]
        ),
        "independent graph inputs required",
    )
    _require(
        not ep.range_constraints
        and ep.call_spec.in_spec == tree.tree_flatten((tuple(range(n)), {}))[1]
        and all(x.kind.name == "USER_OUTPUT" for x in ep.graph_signature.output_specs),
        "expected static, nonmutating tensor export",
    )
    gm = ep.module()
    nodes = list(gm.graph.nodes)
    candidates = {}
    for node in nodes:
        if node.op in ("placeholder", "get_attr", "output"):
            continue
        if (
            node.op == "call_module"
            and node.target == "_guards_fn"
            and type(gm.get_submodule(node.target)) is GuardsFn
        ):
            continue  # PyTorch's unlifted input guard is retained in the executable GraphModule.
        _require(
            node.op == "call_function"
            and isinstance(node.target, torch._ops.OpOverload)
            and not node.target._schema.is_mutable,
            "unsupported effect or mutation",
        )
        if node.target is SDPA:
            # A nonzero or dynamic dropout is stateful; reject before the first interpretation.
            arg = node.args[4] if len(node.args) > 4 else node.kwargs.get("dropout_p", 0.0)
            _require(type(arg) in (int, float) and arg == 0, "dropout would consume RNG")
            candidates[node] = None
        else:
            _require(
                str(node.target).removeprefix("aten.") in _PURE and node.target.namespace == "aten",
                "unrecognized surrounding op: " + str(node.target),
            )
    if not candidates:
        return DecodePartition(gm, [], [])
    traces = []
    outputs = []
    for batch in batches:
        i = fx.Interpreter(gm, garbage_collect_values=False)
        outputs.append(i.run(*batch))
        traces.append(i.env)
    plans = []
    skipped = []
    for node in candidates:
        try:
            ins, opt = _options(node)
            branches = tuple(_branch(i) for i in ins[1:3]) if elide_kv_repeats else (None, None)
            elide = (
                all(branches)
                and branches[0].shape == branches[1].shape
                and branches[0].out is not branches[1].out
            )
            if elide:
                ins = [ins[0], branches[0].src, branches[1].src, *ins[3:]]
                opt["enable_gqa"] = True
            else:
                branches = ()
            ex, au = [tuple(env[x] for x in ins) for env in traces]
            op = _build_site(
                ins,
                opt,
                ex,
                au,
                reference=lambda j, b: traces[j][node],
                tile=tile,
                head_group=head_group,
                max_workspace_bytes=max_workspace_bytes,
            )
            plans.append((node, ins, op, branches))
        except (Unsupported, ValueError) as e:
            skipped.append(dict(name=node.name, reason=str(e)))
    records = []
    replaced = {}
    dead = []
    for node, ins, op, branches in plans:
        name = f"_sparkfuse_decode_site_{len(records)}"
        _require(not hasattr(gm, name), "module name collision")
        gm.add_module(name, _Region(op, domain == "checked"))
        with gm.graph.inserting_before(node):
            new = gm.graph.call_module(name, tuple(replaced.get(x, x) for x in ins))
            new.meta = dict(node.meta)
        replaced[node] = new
        node.replace_all_uses_with(new)
        gm.graph.erase_node(node)
        record = dict(
            name=node.name,
            module=name,
            domain=domain,
            operands=[x.name for x in ins],
            plan=asdict(op.plan),
            bypassed_repeats=[br.out.name for br in branches],
            erased_repeats=[],
            removed_materialized_bytes=0,
        )
        records.append(record)
        for br in branches:
            dead.append((br, record))
    # Shared repeats may be referenced by multiple sites. Attribute each erased materialization once.
    visited = set()
    for br, r in dead:
        if br.out not in visited and not br.out.users:
            visited.add(br.out)
            r["erased_repeats"].append(br.out.name)
            r["removed_materialized_bytes"] += (
                prod(br.shape) * torch.empty((), dtype=br.dtype).element_size()
            )
            gm.graph.erase_node(br.out)
    for attr in ("expand", "unsqueeze"):
        for node in dict.fromkeys(getattr(br, attr) for br, _ in dead):
            if not node.users:
                gm.graph.erase_node(node)
    if plans:
        gm.graph.lint()
        gm.recompile()
        for batch, ref in zip(batches, outputs):
            out = gm(*batch)
            _require(
                _signature(out, batch) == _signature(ref, batch),
                "whole-graph output structure/storage differs",
            )
            for y, r in zip(tree.tree_leaves(out), tree.tree_leaves(ref)):
                at, rt = (
                    (0.012, 0.02)
                    if r.dtype == torch.bfloat16
                    else (0.002, 0.004)
                    if r.dtype == torch.float16
                    else (0, 0)
                )
                try:
                    torch.testing.assert_close(y, r, atol=at, rtol=rt, check_stride=True)
                except AssertionError as e:
                    raise Unsupported("whole-graph result differs from original export") from e
    return DecodePartition(gm, records, skipped)
