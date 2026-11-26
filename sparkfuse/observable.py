"""Freeze the complete dataflow slice of named-by-position observable output leaves."""

from dataclasses import dataclass
from torch import fx
from torch.utils import _pytree as tree
from .decode_attention import _require


@dataclass
class OutputSlice:
    indices: tuple
    leaves: tuple
    nodes: frozenset
    calls: dict

    @classmethod
    def capture(cls, gm, indices):
        (out,) = [n for n in gm.graph.nodes if n.op == "output"]
        leaves = tree.tree_leaves(out.args[0])
        _require(
            type(indices) is tuple
            and all(
                type(i) is int and 0 <= i < len(leaves) and isinstance(leaves[i], fx.Node)
                for i in indices
            )
            and len(set(indices)) == len(indices),
            "exact_outputs must be distinct valid flat output indices",
        )
        selected = tuple(leaves[i] for i in indices)
        seen = set()
        todo = list(selected)
        while todo:
            n = todo.pop()
            if n not in seen:
                seen.add(n)
                todo.extend(n.all_input_nodes)
        # FX args/kwargs are immutable containers. Hold actual node identities and literal
        # arguments, not a hash or just a count; do not accept a replacement with the same name.
        return cls(
            indices,
            selected,
            frozenset(seen),
            {n: (n.op, n.target, n.args, n.kwargs) for n in seen},
        )

    def verify(self, gm):
        now = OutputSlice.capture(gm, self.indices)
        _require(
            now.leaves == self.leaves and now.nodes == self.nodes and now.calls == self.calls,
            "protected output dependency changed during rewrite",
        )

    def describe(self, skipped):
        return dict(
            flat_indices=list(self.indices),
            outputs=[n.name for n in self.leaves],
            protected_nodes=sorted(n.name for n in self.nodes),
            unchanged_dependencies=True,
            skipped_attention=skipped,
        )
