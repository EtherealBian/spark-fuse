"""Embed one runtime-length attention program in a symbolic, functional decode layer."""

import math
import operator
from dataclasses import asdict
import sympy
import torch
from torch import fx
from torch.export._unlift import GuardsFn
from torch.utils import _pytree as tree
from .decode_attention import _require
from .decode_sites import DecodePartition, SDPA, _PURE, _options, _Region, _signature
from .ir import Unsupported
from .length_attention import LengthPlan, _build_length, _expr


def _shape(n):
    v = n.meta.get("val") if isinstance(n, fx.Node) else None
    _require(isinstance(v, torch.Tensor), "expected tensor metadata")
    return tuple(map(_expr, v.shape))


def _arg_shape(a):
    _require(isinstance(a, (tuple, list)), "shape arguments must be explicit")
    return tuple(_expr(v.meta["val"]) if isinstance(v, fx.Node) else v for v in a)


def _repeat(n):
    # Match the actual reshape and expand arguments, not just the inferred result.
    _require(
        n.op == "call_function"
        and n.target is torch.ops.aten.reshape.default
        and len(n.args) == 2
        and not n.kwargs,
        "expected reshape of repeated heads",
    )
    e, osh = n.args
    _require(
        e.op == "call_function"
        and e.target is torch.ops.aten.expand.default
        and len(e.args) == 2
        and not e.kwargs,
        "expected explicit head broadcast",
    )
    u, esh = e.args
    _require(
        u.op == "call_function"
        and u.target is torch.ops.aten.unsqueeze.default
        and len(u.args) == 2
        and u.args[1] == 2
        and not u.kwargs,
        "expected unsqueeze at head-group axis",
    )
    src = u.args[0]
    b, g, s, d = _shape(src)
    a, h, t, w = _shape(n)
    _require(
        all(type(x) is int for x in (b, g, d, a, h, w))
        and b == a
        and d == w
        and g > 0
        and h > g
        and h % g == 0
        and t == s
        and _arg_shape(osh) == (b, h, s, d)
        and _arg_shape(esh) == (b, g, h // g, s, d)
        and _shape(e) == (b, g, h // g, s, d)
        and _shape(u) == (b, g, 1, s, d),
        "not a canonical KV repeat",
    )
    return src, (n, e, u)


def _linear_consumer(n, b, h, d):
    def only(k):
        _require(len(k.users) == 1, "intermediate attention layout is observable")
        return next(iter(k.users))

    t = only(n)
    _require(
        t.op == "call_function"
        and t.target is torch.ops.aten.transpose.int
        and t.args == (n, 1, 2)
        and not t.kwargs,
        "expected attention transpose",
    )
    r = only(t)
    _require(
        r.op == "call_function"
        and r.target in (torch.ops.aten.reshape.default, torch.ops.aten.view.default)
        and len(r.args) == 2
        and r.args[0] is t
        and not r.kwargs
        and tuple(r.args[1]) in ((b, 1, -1), (b, 1, h * d)),
        "expected flattened hidden state",
    )
    l = only(r)
    _require(
        l.op == "call_function"
        and l.target is torch.ops.aten.linear.default
        and len(l.args) >= 2
        and l.args[0] is r
        and not l.kwargs
        and all(z is not r for z in l.args[1:]),
        "internal strides must not escape before linear",
    )
    return l.name


def _plan(ep, ins, opts, tile, warp_heads=0):
    q, k, v = ins[:3]
    b, h, n, d = _shape(q)
    bk, g, s, dk = _shape(k)
    _require(
        all(type(x) is int for x in (b, h, n, d, bk, g, dk))
        and 1 <= b == bk <= 32
        and 1 <= g <= h <= 256
        and h % g == 0
        and (g == h or opts["enable_gqa"])
        and n == 1
        and d == dk
        and d in (64, 128)
        and _shape(v) == _shape(k),
        "invalid symbolic GQA layout",
    )
    syms = [u for u in ep.range_constraints if isinstance(u, sympy.Symbol)]
    _require(
        len(syms) == 1
        and s in (syms[0], syms[0] + 1)
        and set(ep.range_constraints) <= set([s, syms[0]]),
        "expected KV length equal to past or past+1",
    )
    root = syms[0]
    delta = int(s - root)
    bounds = ep.range_constraints[root]
    _require(
        isinstance(bounds.lower, sympy.Integer) and isinstance(bounds.upper, sympy.Integer),
        "finite shape bounds required",
    )
    lo, hi = int(bounds.lower) + delta, int(bounds.upper) + delta
    if s in ep.range_constraints:
        exact = ep.range_constraints[s]
        _require(exact.lower == lo and exact.upper == hi, "conflicting derived KV bounds")
    _require(2 <= lo < hi <= 32768, "unsupported context range")
    m = 0
    if len(ins) == 4:
        sh = _shape(ins[3])
        m = sh[0]
        _require(type(m) is int and m in (1, b) and sh == (m, 1, 1, s), "unsupported dynamic mask")
    dtype, device = q.meta["val"].dtype, q.meta["val"].device
    _require(
        dtype in (torch.bfloat16, torch.float16) and device.type == "cuda",
        "expected narrow CUDA operands",
    )
    for x in ins:
        sh = _shape(x)
        t = x.meta["val"]
        _require(
            t.dtype == dtype
            and t.device == device
            and t.layout == torch.strided
            and not t.is_neg()
            and not t.is_conj()
            and len(sh) == 4
            and all(
                sympy.expand(_expr(stride) - math.prod(sh[j + 1 :])) == 0
                for j, stride in enumerate(t.stride())
            ),
            "need canonical input pitches",
        )
    sc = 1 / math.sqrt(d) if opts["scale"] is None else opts["scale"]
    _require(type(sc) in (float, int) and math.isfinite(sc) and 0 < sc <= 32, "unsupported scale")
    return LengthPlan(lo, hi, b, h, g, d, m, dtype, device, sc, tile, warp_heads), delta


def partition_symbolic_decode(
    ep,
    example,
    audit,
    *,
    tile=32,
    warp_heads=0,
    domain="checked",
    host_backend="native",
    max_workspace_bytes=32 << 20,
):
    """Lower internal decode attention; never discard a live cache, shape guard or repeat."""
    batches = tuple(map(tuple, (example, audit)))
    n = len(batches[0])
    _require(
        len(batches[1]) == n
        and ep.range_constraints
        and ep.call_spec.in_spec == tree.tree_flatten((tuple(range(n)), {}))[1]
        and all(x.kind.name == "USER_OUTPUT" for x in ep.graph_signature.output_specs)
        and torch.is_inference_mode_enabled()
        and not torch.is_autocast_enabled("cuda")
        and domain in ("bounded", "checked")
        and host_backend in ("native", "ctypes"),
        "expected positional symbolic inference export",
    )
    _require(
        all(type(x) is torch.Tensor and x.device.type == "cuda" for xs in batches for x in xs)
        and {x.untyped_storage().data_ptr() for x in batches[0]}.isdisjoint(
            x.untyped_storage().data_ptr() for x in batches[1]
        ),
        "two independent CUDA traces required",
    )
    gm = ep.module()
    sites = []
    for n in gm.graph.nodes:
        if n.op in ("placeholder", "get_attr", "output"):
            continue
        if (
            n.op == "call_module"
            and n.target == "_guards_fn"
            and type(gm.get_submodule(n.target)) is GuardsFn
        ):
            continue
        if (
            n.op == "call_function"
            and n.target is operator.add
            and isinstance(n.meta.get("val"), torch.SymInt)
            and all(
                type(x) is int
                or (isinstance(x, fx.Node) and isinstance(x.meta.get("val"), torch.SymInt))
                for x in n.args
            )
        ):
            continue
        _require(
            n.op == "call_function"
            and isinstance(n.target, torch._ops.OpOverload)
            and n.target.namespace == "aten"
            and not n.target._schema.is_mutable,
            "effectful surrounding op",
        )
        if n.target is SDPA:
            arg = n.args[4] if len(n.args) > 4 else n.kwargs.get("dropout_p", 0.0)
            _require(type(arg) in (float, int) and arg == 0, "dropout would consume RNG")
            sites.append(n)
        else:
            _require(
                str(n.target).removeprefix("aten.") in _PURE
                or n.target is torch.ops.aten.sym_size.int,
                "unsupported surrounding op: " + str(n.target),
            )
    if not sites:
        return DecodePartition(gm, [], [])
    envs = []
    original = []
    for xs in batches:
        visitor = fx.Interpreter(gm, garbage_collect_values=False)
        original.append(visitor.run(*xs))
        envs.append(visitor.env)
    ready = []
    skip = []
    for n in sites:
        try:
            ins, opts = _options(n)
            k, kchain = _repeat(ins[1])
            v, vchain = _repeat(ins[2])
            _require(k is not v and kchain[0] is not vchain[0], "distinct KV branches required")
            ins = [ins[0], k, v, *ins[3:]]
            opts["enable_gqa"] = True
            p, delta = _plan(ep, ins, opts, tile, warp_heads)
            sink = _linear_consumer(n, p.batch, p.heads, p.width)
            xs, ys = [tuple(e[x] for x in ins) for e in envs]
            fn = _build_length(
                p,
                xs,
                ys,
                reference=lambda j, batch: envs[j][n].clone(memory_format=torch.contiguous_format),
                host_backend=host_backend,
                max_workspace_bytes=max_workspace_bytes,
            )
            ready.append((n, ins, fn, kchain, vchain, sink, delta))
        except (Unsupported, ValueError) as exc:
            skip.append(dict(name=n.name, reason=str(exc)))
    records = []
    dead = []
    for n, ins, fn, kc, vc, sink, delta in ready:
        name = f"_sparkfuse_symbolic_attention_{len(records)}"
        _require(not hasattr(gm, name), "name collision")
        gm.add_module(name, _Region(fn, domain == "checked"))
        with gm.graph.inserting_before(n):
            new = gm.graph.call_module(name, tuple(ins))
        # Do not copy the old SDPA FakeTensor's potentially different singleton strides.
        n.replace_all_uses_with(new)
        gm.graph.erase_node(n)
        p = fn.plan
        rec = dict(
            name=n.name,
            module=name,
            consumer=sink,
            operands=[i.name for i in ins],
            domain=domain,
            host_backend=host_backend,
            offset_from_symbol=delta,
            plan={**asdict(p), "device": str(p.device), "dtype": str(p.dtype)},
            bypassed_repeats=[kc[0].name, vc[0].name],
            erased_repeats=[],
        )
        records.append(rec)
        dead.append((kc, vc, rec))
    for depth in range(3):
        for kc, vc, rec in dead:
            for chain in (kc, vc):
                old = chain[depth]
                if len(old.users) == 0 and not old._erased:
                    if not depth:
                        rec["erased_repeats"].append(old.name)
                    gm.graph.erase_node(old)
    if ready:
        gm.graph.lint()
        gm.recompile()
        for xs, ref in zip(batches, original):
            y = gm(*xs)
            _require(
                _signature(y, xs) == _signature(ref, xs), "output storage or public shape changed"
            )
            for a, b in zip(tree.tree_leaves(y), tree.tree_leaves(ref), strict=True):
                at, rt = (
                    (0.012, 0.02)
                    if b.dtype == torch.bfloat16
                    else ((0.002, 0.004) if b.dtype == torch.float16 else (0, 0))
                )
                try:
                    torch.testing.assert_close(a, b, atol=at, rtol=rt, check_stride=True)
                except AssertionError as exc:
                    raise Unsupported("whole graph failed numerical qualification") from exc
    return DecodePartition(gm, records, skip)
