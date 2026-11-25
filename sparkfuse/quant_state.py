"""Eligibility for a private packed RHS in a frozen W8A8 CUDA graph (no GEMM rewriting)."""

from math import prod
import torch
from .ir import Unsupported
from .rotary import _is
from .token_quant import candidate

A = torch.ops.aten


def _static(t, dtype):
    return (
        isinstance(t, torch.Tensor)
        and t.layout == torch.strided
        and t.dtype is dtype
        and t.device.type == "cuda"
        and not t.is_neg()
        and not t.is_conj()
        and t.ndim == 2
        and all(type(j) is int and j > 0 for j in (*t.shape, *t.stride()))
        and prod(t.shape) <= 2**28
    )


def certify(ep):
    """Admit the extra operators only inside the exact quantizer and bounded signed GEMM."""
    added = set()
    codes = set()
    mms = 0
    for node in ep.graph.nodes:
        if not _is(node, A.to.dtype, 2) or node.args[1] is not torch.int8:
            continue
        v = node.meta.get("val")
        site = candidate(node, v.device) if isinstance(v, torch.Tensor) else None
        if site is not None:
            added.update(site.members)
            codes.add(node)
    for node in ep.graph.nodes:
        if not _is(node, A._int_mm.default, 2):
            continue
        lhs, rhs = node.args
        origin = (
            lhs.args[0] if _is(lhs, A.reshape.default, 2) or _is(lhs, A.view.default, 2) else lhs
        )
        if origin not in codes:
            raise Unsupported("int_mm must consume this exported signed token quantizer")
        x = getattr(lhs, "meta", {}).get("val")
        w = getattr(rhs, "meta", {}).get("val")
        out = node.meta.get("val")
        if not (_static(x, torch.int8) and _static(w, torch.int8) and _static(out, torch.int32)):
            raise Unsupported("requires static int8 matrices and an int32 accumulator")
        m, k = x.shape
        kw, n = w.shape
        if (
            k != kw
            # Conservative common domain for row/column GB10 signed GEMM.
            # Some narrower K work only in one layout, not both.
            or m <= 16
            or k % 64 != 0
            or n % 8 != 0
            or k * 128**2 >= 2**31
            or out.device != x.device
            or w.device != x.device
            or tuple(out.shape) != (m, n)
            or x.stride() != (k, 1)
            or w.stride() not in ((n, 1), (1, k))
            or out.stride() != (n, 1)
        ):
            raise Unsupported("outside frozen int_mm shape/stride/accumulator domain")
        added.add(node)
        mms += 1
    if not mms:
        raise Unsupported("frozen W8A8 capture requires a qualified signed integer projection")
    return added


def frozen_weights(graph, gm, dev, layout, get, store, geometry):
    """Every live INT8 state alias must be an unmodified, direct GEMM RHS with identical layout."""
    seen = {}
    binding = {}
    inventory = []
    integer_gemms = set()
    nodes = [n for n in graph.nodes if n.op == "get_attr" and n.users]
    for node in nodes:
        t = get(gm, node.target)
        if not isinstance(t, torch.Tensor) or t.dtype is not torch.int8:
            continue
        if (
            type(t) not in (torch.Tensor, torch.nn.Parameter)
            or not _static(t, torch.int8)
            or t.device != dev
            or min(t.shape) < 2
            or t.storage_offset() != 0
            or t.untyped_storage().nbytes() != t.nbytes
            or t.stride() not in ((t.shape[1], 1), (1, t.shape[0]))
        ):
            raise Unsupported("frozen int8 state must own a full row- or column-major matrix")
        if not node.users or not all(
            _is(u, A._int_mm.default, 2) and u.args[1] is node and u.args[0] is not node
            for u in node.users
        ):
            raise Unsupported("frozen int8 state may be used only as a direct int_mm RHS")
        integer_gemms.update(node.users)
        key = store(t)
        g = geometry(t)
        if key in seen and seen[key][0] != g:
            raise Unsupported("incompatible tied int8 state layouts")
        group = seen.setdefault(key, (g, len(seen)))[1]
        target = tuple(t.stride()) if layout == "preserve" else (1, t.shape[0])
        binding[node.target] = target
        inventory.append(
            dict(
                name=node.target,
                shape=list(t.shape),
                source_stride=list(t.stride()),
                snapshot_stride=list(target),
                storage_group=group,
                bytes=t.nbytes,
            )
        )
    if not binding or integer_gemms != {n for n in graph.nodes if _is(n, A._int_mm.default, 2)}:
        raise Unsupported("each int_mm requires a direct, private frozen RHS")
    for n in nodes:
        t = get(gm, n.target)
        if isinstance(t, torch.Tensor) and store(t) in seen and n.target not in binding:
            raise Unsupported("non-int8 or other state aliases a frozen RHS")
    return binding, inventory
