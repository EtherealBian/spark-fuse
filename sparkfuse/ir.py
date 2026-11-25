"""Small, closed IR: one dense iteration domain and at most one last-axis reduction."""

from dataclasses import dataclass
import math
import torch
from torch.fx import Node


class Unsupported(ValueError):
    pass


DT = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}


@dataclass(frozen=True)
class Val:
    op: str
    shape: tuple
    dt: str
    deps: tuple = ()
    imm: object = None


@dataclass
class Plan:
    vals: list
    inputs: list
    outputs: list
    domain: tuple
    red: int | None

    @property
    def width(self):
        return self.domain[-1]

    @property
    def rows(self):
        return math.prod(self.domain[:-1])

    def explain(self):
        return dict(
            domain=self.domain,
            reduction=self.red,
            input_ids=self.inputs,
            output_ids=self.outputs,
            values=[vars(x) for x in self.vals],
            launches=1,
            global_intermediate_bytes=0,
        )


def lower(ep):
    vs = []
    ids = {}
    ins = []
    outs = []
    red = None
    aliases = {}  # Alias roots for no-copy casts; observable input/output aliasing cannot be erased.

    def put(op, shape, dt, deps=(), imm=None):
        vs.append(Val(op, shape, dt, tuple(deps), imm))
        return len(vs) - 1

    def arg(v):
        if isinstance(v, Node):
            return ids[v]
        if type(v) in (int, float) and math.isfinite(v) and abs(v) <= 3.4028234e38:
            return put("const", (), "f32", imm=float(v))
        raise Unsupported(f"unsupported scalar: {v!r}")

    def shape_dtype(n):
        t = n.meta.get("val")
        if not isinstance(t, torch.Tensor) or t.dtype not in DT:
            raise Unsupported(f"{n.name}: expected floating tensor")
        if any(type(s) is not int or s <= 0 for s in t.shape):
            raise Unsupported("only positive static extents supported")
        if t.layout != torch.strided or not t.is_contiguous():
            raise Unsupported(f"{n.name}: requires contiguous layout")
        return tuple(t.shape), DT[t.dtype]

    for n in ep.graph.nodes:
        if n.op == "output":
            if not isinstance(n.args[0], (list, tuple)) or not all(
                isinstance(o, Node) for o in n.args[0]
            ):
                raise Unsupported("expected flat tensor graph outputs")
            outs = [ids[o] for o in n.args[0]]
            continue
        if n.op == "call_function" and str(n.target) == "aten._assert_tensor_metadata.default":
            if (
                len(n.args) != 1
                or n.users
                or any(
                    key not in ("dtype", "device", "layout")
                    or getattr(n.args[0].meta["val"], key) != v
                    for key, v in n.kwargs.items()
                )
            ):
                raise Unsupported("unproven metadata assertion")
            continue
        sh, dt = shape_dtype(n)
        if n.op == "placeholder":
            ids[n] = put("input", sh, dt, imm=len(ins))
            ins.append(ids[n])
            continue
        if n.op != "call_function":
            raise Unsupported(f"unsupported FX node kind: {n.op}")
        op = str(n.target)
        a = n.args
        k = n.kwargs
        if op in (
            "aten.add.Tensor",
            "aten.mul.Tensor",
            "aten.sub.Tensor",
            "aten.div.Tensor",
            "aten.add.Scalar",
            "aten.mul.Scalar",
            "aten.sub.Scalar",
        ):
            if any(key != "alpha" or value != 1 for key, value in k.items()) or len(a) != 2:
                raise Unsupported(f"{op}: nondefault arguments")
            idx = put(op.split(".")[1], sh, dt, [arg(a[0]), arg(a[1])])
        elif op in (
            "aten.silu.default",
            "aten.rsqrt.default",
            "aten.relu.default",
            "aten.neg.default",
            "aten.sigmoid.default",
            "aten.tanh.default",
            "aten.square.default",
        ):
            idx = put(op.split(".")[1], sh, dt, [arg(a[0])])
        elif op == "aten.pow.Tensor_Scalar" and a[1] == 2:
            idx = put("square", sh, dt, [arg(a[0])])
        elif op in (
            "aten.to.dtype",
            "aten._to_copy.default",
            "aten.to.dtype_layout",
            "prims.convert_element_type.default",
        ):
            if any(
                key not in ("dtype", "non_blocking", "copy", "memory_format") for key in k
            ) or k.get("memory_format") not in (
                None,
                torch.preserve_format,
                torch.contiguous_format,
            ):
                raise Unsupported("unsupported cast options")
            source = arg(a[0])
            idx = put("cast", sh, dt, [source])
            if op in ("aten.to.dtype", "aten.to.dtype_layout") and dt == vs[source].dt:
                force_copy = a[3] if len(a) > 3 else k.get("copy", False)
                if not force_copy:
                    aliases[idx] = aliases.get(source, source)
        elif op == "aten.rms_norm.default":
            src = arg(a[0])
            shape = vs[src].shape
            if red is not None or tuple(a[1]) != (shape[-1],):
                raise Unsupported("requires one last-axis reduction")
            eps = a[3] if len(a) > 3 else k.get("eps")
            eps = torch.finfo(torch.float32).eps if eps is None else float(eps)
            if not math.isfinite(eps) or eps < 0:
                raise Unsupported("invalid epsilon")
            sq = put("square", shape, "f32", [src])
            red = put("mean", (*shape[:-1], 1), "f32", [sq])
            e = arg(eps)
            adj = put("add", vs[red].shape, "f32", [red, e])
            inv = put("rsqrt", vs[red].shape, "f32", [adj])
            z = put("mul", shape, "f32", [src, inv])
            w = a[2] if len(a) > 2 else k.get("weight")
            idx = put("cast", sh, dt, [z]) if w is None else put("mul", sh, dt, [z, arg(w)])
        elif op == "aten.mean.dim":
            src = arg(a[0])
            shape = vs[src].shape
            keep = a[2] if len(a) > 2 else k.get("keepdim", False)
            if red is not None or list(a[1]) not in ([-1], [len(shape) - 1]) or not keep:
                raise Unsupported("mean must reduce last axis with keepdim=True")
            red = idx = put("mean", sh, dt, [src])
        else:
            raise Unsupported(f"unsupported operator: {op}")
        ids[n] = idx
    if not ins or not outs:
        raise Unsupported("empty graph")
    domain = max((vs[i].shape for i in ins), key=lambda s: (len(s), math.prod(s)))
    if not domain or math.prod(domain) >= 2**31:
        raise Unsupported("requires a nonempty, 32-bit index domain")
    shapes = {domain, (domain[-1],), (), (*domain[:-1], 1)}
    if any(v.shape not in shapes for v in vs):
        raise Unsupported("incompatible broadcast domains")
    if red is not None and (
        vs[vs[red].deps[0]].shape != domain or vs[red].shape != (*domain[:-1], 1)
    ):
        raise Unsupported("reduction must span the iteration domain")
    if any(vs[i].shape not in (domain, (*domain[:-1], 1)) for i in outs):
        raise Unsupported("unsupported output shape")
    roots = [aliases.get(i, i) for i in outs]
    if any(vs[i].op == "input" for i in roots) or len(set(roots)) != len(roots):
        raise Unsupported("aliased outputs unsupported")
    return Plan(vs, ins, outs, domain, red)
