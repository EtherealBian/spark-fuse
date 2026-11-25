"""One symbolic leading row count; one executable, guarded launch shapes."""

from copy import copy
from dataclasses import replace
from functools import lru_cache
from types import SimpleNamespace
import torch
from torch import fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils import _pytree as tree
from .ir import Plan, Unsupported
from .runtime import compile_export


class DynamicRows:
    def __init__(self, ep, device="cuda", schedule=None):
        if len(ep.range_constraints) != 1:
            raise Unsupported("requires exactly one bounded export symbol")
        ((sym, span),) = ep.range_constraints.items()
        try:
            lo, hi = max(1, int(span.lower)), int(span.upper)
        except (TypeError, ValueError, OverflowError, AttributeError) as e:
            raise Unsupported("requires finite integer row bounds") from e
        if hi < max(lo, 2):
            raise Unsupported("empty or singleton dynamic domain")
        representative = max(2, lo)
        g = fx.Graph()
        nodes = {}
        inp_index = 0
        first = None
        mode = FakeTensorMode()
        for n in ep.graph.nodes:
            v = g.node_copy(n, lambda old: nodes[old])
            nodes[n] = v
            v.meta = dict(n.meta)
            t = n.meta.get("val")
            dynamic = False
            if isinstance(t, torch.Tensor):
                dims = []
                for axis, x in enumerate(t.shape):
                    if type(x) is int:
                        dims.append(x)
                    elif (
                        isinstance(x, torch.SymInt)
                        and axis == 0
                        and t.ndim == 2
                        and x.node.expr == sym
                    ):
                        dims.append(representative)
                        dynamic = True
                    else:
                        raise Unsupported(
                            "symbol must occur only as the leading dimension of a rank-2 tensor"
                        )
                if t.ndim >= 2 and not dynamic and not (t.ndim == 2 and dims[0] == 1):
                    raise Unsupported(
                        "rank-2 tensors require the leading symbol or a singleton broadcast row"
                    )
                if not t.is_contiguous():
                    raise Unsupported("requires contiguous dynamic tensors")
                with mode:
                    v.meta["val"] = torch.empty(dims, device=t.device, dtype=t.dtype)
            if n.op == "placeholder":
                spec = ep.graph_signature.input_specs[inp_index]
                if dynamic and spec.kind.name != "USER_INPUT":
                    raise Unsupported("dynamic parameter or buffer")
                if dynamic and first is None:
                    first = sum(
                        s.kind.name == "USER_INPUT"
                        for s in ep.graph_signature.input_specs[:inp_index]
                    )
                inp_index += 1
        if first is None:
            raise Unsupported("no user input with dynamic rows")
        proxy = SimpleNamespace(
            graph=g,
            graph_signature=ep.graph_signature,
            call_spec=ep.call_spec,
            state_dict=ep.state_dict,
            constants=ep.constants,
        )
        self.kernel = compile_export(proxy, device, schedule)
        if (
            self.kernel.plan.domain[0] != representative
            or len(self.kernel.plan.domain) != 2
            or hi * self.kernel.plan.width >= 2**31
        ):
            raise Unsupported("invalid dynamic launch domain")
        self.bounds = (lo, hi)
        self._first = first
        self._spec = ep.call_spec.in_spec

        # Per-instance bounded metadata cache; never recompiles or copies GPU tensors.
        @lru_cache(maxsize=32)
        def adapt(rows):
            k = copy(self.kernel)
            p = k.plan
            # representative >= 2: a static singleton row never aliases the symbolic extent.
            vs = [
                replace(v, shape=(rows, v.shape[1]))
                if len(v.shape) == 2 and v.shape[0] == representative
                else v
                for v in p.vals
            ]
            k.plan = Plan(vs, p.inputs, p.outputs, (rows, p.width), p.red)
            return k

        self._adapt = adapt

    def native_dispatch(self):
        """Build one native guarded dispatcher for every row count in this export."""
        from .native import _native_dispatch

        return _native_dispatch(self.kernel, self.bounds)

    def prepare(self, *args, **kwargs):
        flat, spec = tree.tree_flatten((args, kwargs))
        if spec != self._spec:
            raise ValueError("input pytree mismatch")
        x = flat[self._first]
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError("expected a rank-2 row input")
        rows = x.shape[0]
        if not self.bounds[0] <= rows <= self.bounds[1]:
            raise ValueError(f"row count {rows} outside export bounds {self.bounds}")
        return self._adapt(rows).prepare(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.prepare(*args, **kwargs)()
