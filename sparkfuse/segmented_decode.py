"""Fuse two proven functional KV appends with a one-token decode site. No cache mutation."""

import operator
import torch
from torch import fx
from torch.utils import _pytree as tree
from .cache_append import _candidate
from .decode_attention import _bounded, _require
from .decode_sites import partition_decode_sites, _Region, _signature
from .index_program import IndexProgram
from .native import native_dispatch
from .segmented_codegen import emit_segmented


def _geometry(x):
    return x.shape, x.stride(), x.dtype, x.device


def _bits(x, y):
    return (
        x.shape == y.shape
        and x.dtype == y.dtype
        and x.stride() == y.stride()
        and torch.equal(
            x.detach().contiguous().reshape(-1).view(torch.uint8),
            y.detach().contiguous().reshape(-1).view(torch.uint8),
        )
    )


class SegmentedRegion(torch.nn.Module):
    def __init__(self, base, xs, publish):
        super().__init__()
        self.base = base
        self.publish = tuple(publish)
        self.desc = tuple(_geometry(x) for x in xs)
        base.op._guard(xs)
        p = base.op.plan
        _require(
            p.head_group == 1 and len(xs) in (5, 6),
            "segmented implementation supports one query head per CTA",
        )
        q, k0, k1, v0, v1, *mask = xs
        _require(
            q.dtype in (torch.float16, torch.bfloat16)
            and all(x.dtype == q.dtype and x.device == q.device for x in (k0, k1, v0, v1)),
            "invalid segmented operands",
        )
        for x, s in ((k0, p.sequence - 1), (k1, 1), (v0, p.sequence - 1), (v1, 1)):
            _require(
                x.shape == (p.batch, p.kv_heads, s, p.width)
                and (x.is_contiguous() or x.transpose(1, 2).is_contiguous())
                and x.stride(-1) == 1,
                "unsupported real append operand layout",
            )
        dtype = "bf16" if q.dtype == torch.bfloat16 else "f16"
        md = (dtype if mask[0].dtype == q.dtype else "f32") if mask else None
        src = emit_segmented(
            p,
            dtype,
            [x.stride() for x in (k0, k1, v0, v1)],
            md,
            mask[0].shape[0] if mask else 0,
            self.publish,
        )
        scratch = (p.batch, p.heads, p.chunks, p.width + 2)
        cache = (p.batch, p.kv_heads, p.sequence, p.width)
        outputs = [scratch if p.chunks > 1 else tuple(q.shape)] + [cache for v in publish if v]
        self.program = IndexProgram(
            [tuple(x.shape) for x in xs],
            outputs,
            dtype,
            q.device,
            src,
            input_strides=[x.stride() for x in xs],
            input_dtypes=[dtype] * 5 + ([md] if mask else []),
            output_dtypes=["f32" if p.chunks > 1 else dtype] + [dtype] * sum(publish),
        )
        self.dispatch = native_dispatch(self.program)

    def forward(self, *xs):
        base = self.base.op
        base._guard(xs)
        if self.base.checked:
            if self.desc != tuple(_geometry(x) for x in xs):
                raise ValueError("segmented operand descriptors changed")
            _bounded((xs[0], xs[1], xs[3], *xs[5:]))
            _bounded((xs[2], xs[4]))
        partial, *live = self.dispatch(*xs)
        y = base._second(partial)[0] if base._second else partial
        if base._fix:
            y.as_strided_(y.shape, base.plan.output_stride)
        return y, *live


def partition_segmented_decode(
    ep, example, audit, *, domain="checked", tile=32, max_workspace_bytes=32 << 20
):
    """On a single qualified site, compute attention from the old cache and one new token.

    A concatenate observable elsewhere is emitted as a fresh CUDA output; private
    concatenates need no allocations. Leave the qualified compact site intact when
    recognizing/scheduling the two appends is not possible.
    """
    base = partition_decode_sites(
        ep,
        example,
        audit,
        domain=domain,
        elide_kv_repeats=True,
        tile=tile,
        head_group=1,
        max_workspace_bytes=max_workspace_bytes,
    )
    gm = base.module

    def skip(why):
        base.skipped.append(dict(kind="segmented_cache", reason=why))
        return base

    if len(base.sites) != 1:
        return skip("requires exactly one qualified SDPA site")
    rec = base.sites[0]
    site = next(n for n in gm.graph.nodes if n.op == "call_module" and n.target == rec["module"])
    region = gm.get_submodule(site.target)
    assert isinstance(region, _Region)
    q, ko, vo, *mask = site.args
    ks, vs = _candidate(ko), _candidate(vo)
    if ko is vo or not ks or not vs or ks[:3] != vs[:3] or ks[0][3] != 1:
        return skip("requires two independent length-axis single-token appends")
    nodes = (q, *ko.args[0], *vo.args[0], *mask)
    order = {n: j for j, n in enumerate(gm.graph.nodes)}
    consumers = [set(x.users) - {site} for x in (ko, vo)]
    publish = tuple(map(bool, consumers))
    anchor = min({site}.union(*consumers), key=order.get)
    if max(order[n] for n in nodes) >= order[anchor] or set(nodes) & {ko, vo}:
        return skip("an append input does not dominate its earliest live consumer")
    xs = []
    refs = []
    partials = []
    full = []
    for batch in (example, audit):
        i = fx.Interpreter(gm, garbage_collect_values=False)
        full.append(i.run(*batch))
        xs.append(tuple(i.env[n] for n in nodes))
        refs.append(
            tuple(
                i.env[n] for n in (site,) + tuple(n for n, live in zip((ko, vo), publish) if live)
            )
        )
        partials.append(region.op._first(*(i.env[n] for n in site.args))[0])
    _require(
        tuple(map(_geometry, xs[0])) == tuple(map(_geometry, xs[1])),
        "observed segmented inputs differ in shape/stride",
    )
    fused = SegmentedRegion(region, xs[0], publish)
    for args, ref, partial in zip(xs, refs, partials, strict=True):
        y = fused(*args)
        p = fused.dispatch(*args)[0]
        _require(
            _bits(p, partial)
            and all(_bits(a, b) for a, b in zip(y, ref, strict=True))
            and _signature(y, args) == _signature(ref, args),
            "segmented CUDA differs in partials, output bits or ownership",
        )
    label = site.target + "_segmented"
    _require(not hasattr(gm, label), "module name collision")
    gm.add_module(label, fused)
    with gm.graph.inserting_before(anchor):
        pair = gm.graph.call_module(label, nodes)
        repls = []
        for j, old in enumerate((site,) + tuple(n for n, live in zip((ko, vo), publish) if live)):
            z = gm.graph.call_function(operator.getitem, (pair, j))
            z.meta = dict(old.meta)
            repls.append((old, z))
    for old, z in repls:
        old.replace_all_uses_with(z)
    gm.graph.erase_node(site)
    gm.delete_submodule(site.target)
    for n in (ko, vo):
        assert not n.users
        gm.graph.erase_node(n)
    gm.graph.lint()
    gm.recompile()
    for batch, ref in zip((example, audit), full):
        out = gm(*batch)
        _require(
            _signature(out, batch) == _signature(ref, batch)
            and all(
                _bits(a, b)
                for a, b in zip(tree.tree_leaves(out), tree.tree_leaves(ref), strict=True)
            ),
            "whole partition changed observable bits or aliases",
        )
    rec["compact_operands"] = rec["operands"]
    rec["operands"] = [n.name for n in nodes]
    rec["module"] = label
    rec["segmented"] = dict(
        appends=[ko.name, vo.name],
        published=list(publish),
        input_strides=[list(x.stride()) for x in xs[0]],
        private_output_bytes_saved=(2 - sum(publish))
        * refs[0][0].element_size()
        * ks[0][0]
        * ks[0][1]
        * (ks[0][2] + 1)
        * ks[0][4],
    )
    return base
