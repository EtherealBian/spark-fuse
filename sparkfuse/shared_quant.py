"""Share hidden signed token quantizers across independently exported linear modules."""

from dataclasses import dataclass
from math import prod
import torch
from torch import fx
from .rotary import _is
from .token_quant import candidate, _dense

A = torch.ops.aten


@dataclass
class Branch:
    site: object
    codes: fx.Node
    row: fx.Node
    mm: fx.Node


def only(n):
    return next(iter(n.users)) if len(n.users) == 1 else None


def branch(q, dev):
    site = candidate(q, dev, activation=True)
    if site is None or site.roles != ("codes", "scale"):
        return None
    q, s = site.outs
    readers = [u for u in s.users if u not in (*site.members, *site.guards)]
    codes, row = only(q), readers[0] if len(readers) == 1 else None
    k = site.shape[-1]
    m = prod(site.shape[:-1])
    for n, width in [(codes, k), (row, 1)]:
        if (
            not (_is(n, A.reshape.default, 2) or _is(n, A.view.default, 2))
            or not isinstance(n.args[1], (list, tuple))
            or tuple(n.args[1]) not in ((-1, width), (m, width))
        ):
            return None
    mm = only(codes)
    scale = only(row)
    if (
        not _is(mm, A._int_mm.default, 2)
        or mm.args[0] is not codes
        or not isinstance(mm.args[1], fx.Node)
        or mm.args[1].op != "get_attr"
    ):
        return None
    if not _is(scale, A.mul.Tensor, 2) or scale.args[1] is not row:
        return None
    cast = scale.args[0]
    if not _is(cast, A.to.dtype, 2) or cast.args != (mm, torch.float32):
        return None
    v = getattr(mm.args[1], "meta", {}).get("val")
    y = mm.meta.get("val")
    if (
        not isinstance(v, torch.Tensor)
        or v.device != dev
        or v.dtype != torch.int8
        or v.ndim != 2
        or not all(type(j) is int and j > 0 for j in (*v.shape, *v.stride()))
    ):
        return None
    kw, n = v.shape
    if (
        k != kw
        or m <= 16
        or k % 64
        or n % 8
        or k * 128**2 >= 2**31
        or tuple(v.stride()) not in ((1, k), (n, 1))
    ):
        return None
    if (
        not isinstance(y, torch.Tensor)
        or not _dense(mm, (m, n), torch.int32, dev)
        or not _dense(codes, (m, k), torch.int8, dev)
        or not _dense(row, (m, 1), torch.float32, dev)
    ):
        return None
    return Branch(site, codes, row, mm)


# Only move independent functional GEMMs across this closed, static projection vocabulary.
# In particular do not cross unknown assertions, RNG, custom calls, or data-dependent control.
PURE = frozenset(
    (
        A.to.dtype,
        A.abs.default,
        A.amax.default,
        A.clamp_min.default,
        A.div.Tensor,
        A.round.default,
        A.clamp.default,
        A.reshape.default,
        A.view.default,
        A.permute.default,
        A.transpose.int,
        A._int_mm.default,
        A.mul.Tensor,
        A.add.Tensor,
    )
)


def transparent(n):
    if n.op != "call_function":
        return False
    if n.target in PURE:
        return True
    if n.target is not A._assert_tensor_metadata.default or len(n.args) != 1 or n.users:
        return False
    v = getattr(n.args[0], "meta", {}).get("val")
    return isinstance(v, torch.Tensor) and n.kwargs == {
        "dtype": v.dtype,
        "device": v.device,
        "layout": v.layout,
    }


def share(gm, device):
    dev = torch.device(device)
    if dev.type != "cuda":
        return []
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    graph = gm.graph
    groups = {}
    records = []
    for q in graph.nodes:
        b = branch(q, dev)
        if b is not None:
            groups.setdefault(b.site.inputs[0], []).append(b)
    for source, branches in groups.items():
        if not 2 <= len(branches) <= 4:
            continue
        nodes = list(graph.nodes)
        pos = {n: i for i, n in enumerate(nodes)}
        first = branches[0]
        ms = [b.mm for b in branches]
        # Overlapping subgraphs can occur after another simplifier. Only touch whole independent copies.
        own = [n for b in branches for n in (*b.site.members, *b.site.guards, b.codes, b.row, b.mm)]
        if len(set(own)) != len(own) or any(n not in pos for n in own):
            continue
        lo = min(pos[n] for n in own)
        hi = max(pos[n] for n in own)
        if any(not transparent(n) for n in nodes[lo : hi + 1]):
            continue
        if not all(
            pos[b.mm.args[1]] < pos[first.mm] and pos[first.site.outs[1]] < pos[first.mm]
            for b in branches
        ):
            continue
        if first.row is not first.mm.prev:
            first.mm.prepend(first.row)
        prev = first.mm
        for b in branches[1:]:
            b.codes.replace_all_uses_with(first.codes)
            b.row.replace_all_uses_with(first.row)
            graph.erase_node(b.codes)
            graph.erase_node(b.row)
            for n in (*b.site.guards, *reversed(b.site.members)):
                graph.erase_node(n)
            if b.mm is not prev.next:
                prev.append(b.mm)
            prev = b.mm
        records.append(
            dict(
                kind="shared_hidden_token_quantizers",
                activation=source.name,
                branches=len(branches),
                shape=list(first.site.shape),
                gemms=[x.name for x in ms],
                eliminated=len(branches) - 1,
            )
        )
    if records:
        graph.lint()
        gm.recompile()
    return records
