"""Lower one entire bounded [tokens, 2*width] packed SwiGLU export to a native tensor ABI."""

from hashlib import sha256
import torch
from torch.utils import _pytree as tree
from .ir import DT, Unsupported
from .index_program import IndexProgram
from .native import _native_dispatch
from .packed_swiglu import candidate, emit


class PackedRows:
    """One compiled object across the export's positive row range, static width and dtype.

    Input: one ordinary dense CUDA tensor, positional only. Outputs: product, or a flat
    tuple (product,) / (product, rounded_activation). Does not expose split views, bind
    weights, execute FX/module hooks, or implement full-model dynamic partitioning.
    """

    def __init__(self, ep, device="cuda", schedule="adaptive"):
        if schedule not in ("adaptive", "scalar"):
            raise ValueError("packed rows schedule must be adaptive or scalar")
        sig = ep.graph_signature
        if (
            len(sig.input_specs) != 1
            or sig.input_specs[0].kind.name != "USER_INPUT"
            or any(s.kind.name != "USER_OUTPUT" for s in sig.output_specs)
        ):
            raise Unsupported("requires a pure single-input gate with no captured state")
        if ep.call_spec.in_spec != tree.tree_flatten(((0,), {}))[1]:
            raise Unsupported("one positional tensor input required")
        if len(ep.range_constraints) != 1:
            raise Unsupported("requires one bounded leading row symbol")
        ((symbol, span),) = ep.range_constraints.items()
        try:
            lo, hi = max(1, int(span.lower)), int(span.upper)
        except (TypeError, ValueError, OverflowError, AttributeError) as e:
            raise Unsupported("requires finite row bounds") from e
        rows = max(2, lo)
        if hi < rows:
            raise Unsupported("empty or singleton row domain")
        nodes = list(ep.graph.nodes)
        x = nodes[0]
        if x.op != "placeholder" or nodes[-1].op != "output":
            raise Unsupported("invalid export input or output")
        xm = x.meta.get("val")
        if (
            not isinstance(xm, torch.Tensor)
            or xm.ndim != 2
            or type(xm.shape[1]) is not int
            or xm.shape[1] <= 0
            or hi * xm.shape[1] > 2**30
        ):
            raise Unsupported("requires static positive packed width and bounded 32-bit launch")
        device = torch.device(device)
        if device.type != "cuda" or xm.device.type != "cuda":
            raise Unsupported("CUDA export required")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if xm.device != device:
            raise Unsupported("export and launch devices must agree")

        def meta(n):
            v = getattr(n, "meta", {}).get("val")
            if not (
                isinstance(v, torch.Tensor)
                and v.ndim == 2
                and v.device == device
                and v.layout == torch.strided
                and not v.is_neg()
                and not v.is_conj()
                and isinstance(v.shape[0], torch.SymInt)
                and v.shape[0].node.expr == symbol
                and type(v.shape[1]) is int
                and all(type(k) is int and k > 0 for k in v.stride())
            ):
                return None
            return torch.empty_strided((rows, v.shape[1]), v.stride(), dtype=v.dtype, device="meta")

        outs = nodes[-1].args[0]
        ospec = ep.call_spec.out_spec
        if (
            not isinstance(outs, (tuple, list))
            or len(outs) not in (1, 2)
            or (
                ospec != tree.tree_flatten(tuple(range(len(outs))))[1]
                and not (len(outs) == 1 and ospec.is_leaf())
            )
        ):
            raise Unsupported("requires a tensor or flat tuple of fresh gate outputs")
        match = candidate(outs[0], tensor_meta=meta)
        if match is None:
            raise Unsupported("noncanonical packed SwiGLU or unsupported symbolic layout")
        src, split, gate, up, act, shape, half = match
        visible = len(outs) == 2
        if (
            src is not x
            or (visible and outs[1] is not act)
            or set(nodes) != {src, split, gate, up, act, outs[0], nodes[-1]}
        ):
            raise Unsupported("only the gated product and optional rounded activation may escape")
        source, packet = emit(
            shape, DT[xm.dtype], half, visible, schedule != "scalar", dynamic_rows=True
        )
        program = IndexProgram(
            [(rows, xm.shape[1])], [shape] * len(outs), DT[xm.dtype], device, source
        )
        program.ep.call_spec = ep.call_spec
        program.plan.rows = rows
        self.bounds = (lo, hi)
        self.source = source
        self.record = dict(
            kind="packed_rows",
            bounds=[lo, hi],
            width=shape[1],
            dtype=DT[xm.dtype],
            gate_half=half,
            visible_activation=visible,
            schedule=schedule,
            max_packet_elements=packet,
            cuda_sha256=sha256(source.encode()).hexdigest(),
        )
        self.run = _native_dispatch(program, self.bounds)

    def __call__(self, *args):
        return self.run(*args)
