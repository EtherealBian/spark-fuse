"""Optional FX host-call specialization for ordinary inference tensors (mutates a mixed module).

Only literal, audited ATen overloads are eligible. Does not drop export guards, freeze
parameters, generate GEMMs or bypass module hooks. Operator-level tracing can differ.
"""

import torch
from torch import fx
from torch.nn import functional as F


def specialize_host_calls(module: fx.GraphModule, *, excluded=()):
    if not isinstance(module, fx.GraphModule):
        raise TypeError("expected a partitioned FX GraphModule")
    changed = []
    aten = torch.ops.aten
    for node in tuple(module.graph.nodes):
        if node in excluded or node.op != "call_function":
            continue
        args, kw, op = node.args, node.kwargs, node.target
        target = None
        method = False
        if op is aten.dropout.default and not kw and len(args) == 3:
            # ATen validates p even in eval mode. A missing/unknown training argument is not False.
            if type(args[1]) in (int, float) and 0 <= args[1] <= 1 and args[2] is False:
                changed.append(
                    {"node": node.name, "from": str(op), "to": "identity (eval dropout)"}
                )
                node.replace_all_uses_with(args[0])
                module.graph.erase_node(node)
            continue
        if op is aten.linear.default and not kw and len(args) in (2, 3):
            target = F.linear
        elif op is aten.matmul.default and not kw and len(args) == 2:
            target = torch.matmul
        elif (
            op is aten.gelu.default
            and len(args) == 1
            and (not kw or (set(kw) == {"approximate"} and kw["approximate"] in ("none", "tanh")))
        ):
            target = F.gelu
        elif (
            op in (aten.view.default, aten.permute.default)
            and not kw
            and len(args) == 2
            and isinstance(args[1], (list, tuple))
            and all(type(x) is int for x in args[1])
        ):
            target = "view" if op is aten.view.default else "permute"
            method = True
        elif (
            op is aten.transpose.int
            and not kw
            and len(args) == 3
            and all(type(x) is int for x in args[1:])
        ):
            target = "transpose"
            method = True
        elif op is aten.contiguous.default and not kw and len(args) == 1:
            target = "contiguous"
            method = True
        if target is None:
            continue
        changed.append(
            {"node": node.name, "from": str(op), "to": target if method else target.__name__}
        )
        if method:
            node.op = "call_method"
        node.target = target
    if changed:
        module.graph.lint()
        module.recompile()
    return changed
